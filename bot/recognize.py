"""
Perception layer: turns a screenshot of the iPhone Mirroring window into a
GameState (see game_state.py). Must run in real Terminal.app (needs
window_capture.py, which needs Quartz/screencapture).

STATUS: every implemented read_* function (gold, round/phase/time, board
capacity, shop costs, own traits) reads correctly against the calibration
images, via OpenCV template matching (match_icon()/match_digits()) for the
small stylized badge numbers pytesseract couldn't handle, and pytesseract
OCR for plain flat-background text (round/phase label, board-capacity
"4/4"). read_bench()/read_board() still raise NotImplementedError -- see
bot-plan.md in the Claude project for the full up-to-date plan.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "capture"))

from game_state import GameState, Unit, ShopSlot, OpponentInfo

import cv2
import numpy as np
# pytesseract is optional -- only ocr_text()/ocr_digits() need it (used for
# round/phase text and board-capacity reading; everything else -- gold, shop
# costs, trait counts, time-left -- uses template matching and doesn't touch
# this). Importing it lazily/optionally means a broken or missing pytesseract
# install (e.g. a different python3/pip than the one requirements.txt was run
# with) doesn't crash the whole module -- it only breaks the two functions
# that actually need it, with a clear error saying how to fix it.
try:
    import pytesseract
except ImportError:
    pytesseract = None


# ---- regions ---------------------------------------------------------
# Measured directly from data/calib_deploy_phase.png and
# data/calib_opponent_view.png (both live 784x1722px captures via
# window_capture.py, Round 3 Deploy Phase). Format: (x0, y0, x1, y1) as
# fractions (0..1) of the captured frame -- resolution-independent, same
# convention input_control.py's tap_fraction() uses for taps.
#
# RESOLVED from the earlier video-frame investigation: the small badge on
# each shop card IS a real per-card elixir cost (seen here as 4, 2, 3 on
# three different cards in the same shop) -- it varies per card and per
# refresh. The "0, 0, 0" seen identically on all three cards in some video
# frames was a different, not-yet-understood game state (maybe a promo/
# free-card event) -- not a bug in this reading, just not the common case.
# Still TODO: board_hexes (hex grid cell centers) -- not needed until we
# actually place units, lower priority than shop/bench/HUD reading.
REGIONS = {
    "gold_readout": (0.7844, 0.9059, 0.8929, 0.9438),
    "round_phase_text": (0.0, 0.2061, 0.4082, 0.2497),
    "time_left_text": (0.854, 0.229, 0.893, 0.253),   # tight crop of just the
        # number itself (re-measured this session -- the original box captured
        # the "Time left:" label plus only the TOP half of the digits, which is
        # why ocr_digits() kept missing it; found via contour bounding box on a
        # generously tall crop, see data/ocr_debug notes in the project doc)
    "own_trait_badges": (0.0, 0.6737, 0.1658, 0.7898),   # stack of up to 3 badges
    "own_trait_count_badges": [   # tight crop of just the "N/M" pill per badge --
        # NOT evenly spaced within the 3 own_trait_badges slots (measured
        # directly per-badge from data/calib_deploy_phase.png; don't try to
        # derive these from own_trait_badges + a shared fractional offset).
        (0.0929, 0.7043, 0.1360, 0.7116),
        (0.0929, 0.7387, 0.1360, 0.7465),
        (0.0929, 0.7728, 0.1360, 0.7805),
    ],
    "board_capacity_text": (0.3571, 0.4181, 0.6633, 0.4646),  # "4/4" style readout
    "player_hud_row": (0.0, 0.0406, 1.0, 0.1917),        # all 4 players; split into
                                                           # 4 equal-width columns
    "shop_slots": [
        (0.2105, 0.8612, 0.3929, 0.9709),
        (0.4184, 0.8612, 0.5995, 0.9709),
        (0.6250, 0.8612, 0.8061, 0.9709),
    ],
    "shop_cost_badges": [   # top-left corner of each shop_slots box, same order.
        # Re-measured this session -- the original guesses (top-left 30%x25%
        # of each card box) were too high/misaligned and clipped the badge
        # entirely for the middle card. These were checked by eye against
        # the calibration image (see data/ocr_debug/shop{2,3,4}_raw*.png).
        (0.2050, 0.8720, 0.2450, 0.8920),
        (0.3930, 0.8680, 0.4450, 0.8930),
        (0.5810, 0.8680, 0.6330, 0.8930),
    ],
    "shop_merge_badges": [   # top-right corner of each shop_slots box -- the
        # game shows a green up-arrow here when buying that card would
        # complete a merge. NOT measured against a live full-frame capture
        # yet (2026-09-21) -- derived by mirroring shop_cost_badges' offset
        # from the card's left edge onto the right edge, calibrated against
        # a user-provided cropped screenshot (pixel-measured: badge and
        # arrow sit at the same height band, roughly mirror-image x
        # positions -- see bot-plan.md for the actual pixel math). Good
        # enough to ship and verify live rather than guess-and-wait; if
        # merge detection misses or false-positives once there's real live
        # data, re-measure this directly the way every other region here
        # was measured.
        (0.3584, 0.8720, 0.3984, 0.8920),
        (0.5729, 0.8680, 0.6249, 0.8930),
        (0.7981, 0.8680, 0.8501, 0.8930),
    ],
    "bench_slots": [   # 5 slots on the wooden tray above the shop
        (0.1467, 0.7492, 0.2831, 0.8362),
        (0.2831, 0.7492, 0.4196, 0.8362),
        (0.4196, 0.7492, 0.5561, 0.8362),
        (0.5561, 0.7492, 0.6926, 0.8362),
        (0.6926, 0.7492, 0.8291, 0.8362),
    ],
    "board_hexes": [],   # TODO -- hex grid cell centers, needed for placement only
    # when viewing an opponent's board (see input_control.BACK_BUTTON to return):
    "own_board_area": (0.11, 0.42, 0.88, 0.745),   # the player's OWN battlefield
        # during Deploy Phase (own units standing on hexes, not the opponent
        # board above) -- a single bounding rect, not per-hex cells, unlike
        # board_hexes above. Measured 2026-09-22 (see read_board() below and
        # bot-plan.md) via connected-component analysis of the grass-green
        # HSV mask on a real Deploy Phase frame (data/gameplay_runs/
        # run_20260922_013429/frame_091.png) -- the largest green blob's
        # bounding box, which lines up with board_capacity_text just above
        # it and bench_slots just below. One rect rather than a hex grid
        # because that's what read_board() actually needs: it finds
        # occupied positions by detecting non-grass blobs inside this
        # rect, not by matching pre-known hex centers.
    "opponent_name_teamsize": (0.0, 0.2875, 0.2551, 0.3223),
    "opponent_overlay_traits": (0.0, 0.3659, 0.1658, 0.4587),
    "back_button": (0.3125, 0.8944, 0.6888, 0.9466),   # also a tap target
}

REFERENCE_CARDS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "reference_cards", "unlabeled")
REFERENCE_TRAITS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "reference_traits")
REFERENCE_DIGITS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "reference_digits")
# match_digits() segments out an isolated digit GLYPH (contour-cropped, mostly
# just the ink) and needs templates cropped the same way -- different framing
# than REFERENCE_DIGITS_DIR's whole-badge crops (which match_icon uses
# directly, unsegmented, for the shop cost badges). Keeping them in separate
# directories rather than trying to make one template style serve both.
REFERENCE_DIGITS_ISOLATED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "reference_digits_isolated")
# Trait-count badges ("1/2", "2/4" on the dark pill under each trait icon)
# are a third distinct rendering: digit color varies (dim gray, white, or
# green depending on some in-game state not yet understood -- see
# data/reference_trait_digits_isolated/ filenames), and they sit on a dark
# rounded-pill background next to a green background, not the flat/plain
# background gold_readout and shop costs use. Kept as its own directory/
# mask preset in match_digits() rather than reusing either existing style.
REFERENCE_TRAIT_DIGITS_ISOLATED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "reference_trait_digits_isolated")
REFERENCE_PLACEMENT_DIGITS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "reference_placement_digits")


def crop(img, region):
    """region = (x0, y0, x1, y1) as fractions of image width/height."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = region
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def capture_frame():
    """Grab the current iPhone Mirroring window as a BGR numpy array.
    Imports window_capture lazily (needs Quartz -- real macOS Terminal
    only) so the rest of this module can still be imported/unit-tested
    against a static image anywhere (e.g. the sandboxed dev shell)."""
    from window_capture import screenshot_window
    tmp_path = "/tmp/_recognize_capture.png"
    screenshot_window(out_path=tmp_path)
    img = cv2.imread(tmp_path)
    if img is None:
        raise RuntimeError("Failed to read captured screenshot")
    return img


# ---- OCR (numbers/text) -------------------------------------------------

def ocr_digits(img, region, whitelist="0123456789", psm=7, upscale=6):
    """Read a small numeric badge. v1 preprocessing: upscale, blur, Otsu
    threshold, then flip polarity if the "ink" ends up being the majority
    color (we assume digits are the minority pixels in a badge crop).

    Known-good on: round_phase_text, board_capacity_text (clean text on a
    flat/plain background).
    Known-unreliable on: gold_readout, shop_cost_badges, time_left_text --
    these sit on textured/colored art (coin, clock, banner) that leaks
    into the threshold. NEXT STEPS if this keeps misreading live:
      1. Tighten the region further (crop just the digit, not the whole
         badge icon) -- the REGIONS boxes were fit to the badge as a
         *tap-adjacent visual landmark*, not tuned per-pixel for OCR.
      2. Try the HSV saturation-mask approach instead of Otsu (isolate
         near-white, low-saturation pixels specifically -- digits are a
         white fill, badges are colored/golden) -- prototyped, promising
         but not finished.
      3. If both remain unreliable, switch to per-digit template matching
         (crop clean digit glyphs from live captures once we have several,
         match via match_icon() below instead of OCR) -- this UI uses a
         small fixed font, so it's a good candidate for templates over OCR.
    """
    if pytesseract is None:
        raise RuntimeError(
            "pytesseract not installed in this Python -- run "
            "`python3 -m pip install -r requirements.txt` from the repo root "
            "with the SAME python3 you're using to run this script (a "
            "different python3/pip than requirements.txt was installed with "
            "is the usual cause). If the import succeeds but this still fails, "
            "the tesseract CLI binary itself may be missing -- `brew install "
            "tesseract`."
        )
    c = crop(img, region)
    c = cv2.resize(c, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (otsu == 255).mean() > 0.5:
        otsu = cv2.bitwise_not(otsu)
    cfg = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
    txt = pytesseract.image_to_string(otsu, config=cfg)
    return txt.strip()


def match_digits(img, region, template_dir=None, threshold=0.55, upscale=6,
                  white_v_min=190, white_s_max=60, include_green=False,
                  trim_right_frac=None):
    """Read a badge number via per-digit template matching instead of OCR --
    built after ocr_digits() proved unreliable on this game's stylized
    badge font (thick outline, small size, colored/textured backgrounds
    like a coin or clock that a generic threshold can't cleanly separate
    from the digit).

    Method: isolate the digit's ink color inside the region (near-white by
    default -- white_v_min/white_s_max tune that; include_green=True also
    ORs in a green-hue mask, needed for trait-count badges whose digits
    are sometimes rendered green), find the digit-shaped connected
    components, sort them left-to-right, and match each one against every
    template in template_dir via normalized cross-correlation (same
    technique as match_icon()).

    template_dir defaults to REFERENCE_DIGITS_ISOLATED_DIR (gold reading).
    Pass REFERENCE_TRAIT_DIGITS_ISOLATED_DIR for trait-count badges.

    Templates may be named "<digit>.png" or "<digit>_<variant>.png" (e.g.
    "2_gray.png", "2_green.png") -- multiple variant templates per digit
    are all tried and the best score wins; the digit value is always just
    the part of the filename before the first "_" (or the whole stem if
    there's no "_").

    trim_right_frac: if set, zero out the mask beyond this fraction of the
    crop's width before contour detection. Needed for trait-count badges,
    whose dark pill has a rounded right edge that the relaxed mask
    otherwise picks up as a spurious 4th "glyph" (not an issue for
    gold_readout/shop costs, which don't sit on a rounded pill).

    KNOWN GAP: reference_digits_isolated/ only has "3" (gold reading);
    reference_trait_digits_isolated/ has 1, 2, 4 (from the two
    calibration captures). A digit missing a template comes back as "?"
    rather than a wrong guess -- e.g. a real "17" might read as "?7"
    until "1" has a gold-style template. Treat a result containing "?" as
    unreliable/incomplete, not a confirmed reading.
    """
    if template_dir is None:
        template_dir = REFERENCE_DIGITS_ISOLATED_DIR
    sub = crop(img, region)
    sub = cv2.resize(sub, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, white_v_min), (180, white_s_max, 255))
    if include_green:
        green = cv2.inRange(hsv, (35, 80, 80), (85, 255, 255))
        mask = cv2.bitwise_or(mask, green)
    if trim_right_frac is not None:
        cut = int(sub.shape[1] * trim_right_frac)
        mask[:, cut:] = 0
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    # digit-shaped: tall enough relative to the crop, not a sliver
    min_h = sub.shape[0] * 0.25
    boxes = [b for b in boxes if b[3] >= min_h and b[2] >= 5]
    boxes.sort(key=lambda b: b[0])  # left to right

    if not os.path.isdir(template_dir):
        return ""

    templates = {}  # digit -> list of grayscale template arrays (variants)
    for fname in os.listdir(template_dir):
        if fname.lower().endswith(".png"):
            digit = fname.rsplit(".", 1)[0].split("_")[0]
            tmpl = cv2.imread(os.path.join(template_dir, fname), cv2.IMREAD_GRAYSCALE)
            if tmpl is not None:
                templates.setdefault(digit, []).append(tmpl)

    result = []
    for (bx, by, bw, bh) in boxes:
        pad = 6
        glyph = sub[max(0, by-pad):by+bh+pad, max(0, bx-pad):bx+bw+pad]
        glyph_gray = cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY)
        best_digit, best_score = "?", 0.0
        for digit, tmpls in templates.items():
            for tmpl in tmpls:
                for scale in (0.7, 0.85, 1.0, 1.15, 1.3):
                    th, tw = int(tmpl.shape[0]*scale), int(tmpl.shape[1]*scale)
                    if th < 8 or tw < 8 or th > glyph_gray.shape[0] or tw > glyph_gray.shape[1]:
                        continue
                    resized = cv2.resize(tmpl, (tw, th))
                    res = cv2.matchTemplate(glyph_gray, resized, cv2.TM_CCOEFF_NORMED)
                    _, score, _, _ = cv2.minMaxLoc(res)
                    if score > best_score:
                        best_score, best_digit = score, digit
        result.append(best_digit if best_score >= threshold else "?")

    return "".join(result)


def ocr_text(img, region, psm=6):
    """Read a short text label (not digit-whitelisted) -- e.g. 'Round 3' /
    'Deploy Phase'. More reliable than ocr_digits() in testing so far,
    since these regions are plain text on a flat background."""
    if pytesseract is None:
        raise RuntimeError(
            "pytesseract not installed in this Python -- run "
            "`python3 -m pip install -r requirements.txt` from the repo root "
            "with the SAME python3 you're using to run this script."
        )
    c = crop(img, region)
    c = cv2.resize(c, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    txt = pytesseract.image_to_string(gray, config=f"--psm {psm}")
    return txt.strip()


# ---- icon matching (cards, traits) --------------------------------------

def match_icon(img, region, template_dir, threshold=0.6, upscale=1, color=True):
    """Match a cropped region against every template in template_dir via
    normalized cross-correlation, at a few scales (templates may have been
    captured at a slightly different size than the live crop). Returns
    (best_name, score) or (None, best_score) if nothing clears threshold.

    upscale: multiply the cropped region's resolution before matching --
    needed when templates were saved from an upscaled crop (e.g.
    data/reference_digits/, saved at 6-8x) so their pixel size is in the
    same ballpark as the target; otherwise every scale variant ends up
    larger than the (native-resolution) target and nothing can match.

    color: match on the BGR image directly instead of converting to
    grayscale first (2026-09-22 -- discovered live, see bot-plan.md's card
    library writeup). Grayscale NCC turned out to be a bad discriminator
    for the card portraits specifically -- character art differs mostly in
    HUE (purple hair vs blonde vs green skin) rather than in luminance
    structure, and several grayscale-identical-looking cards (same rough
    face/hair silhouette, different color) were winning false matches at
    scores as high as 0.9+, well above any reasonable threshold. Matching
    on color directly fixed most of these in spot checks without hurting
    the badge/digit matching this same function is also used for --
    cv2.imread() always returns 3-channel BGR regardless of whether the
    source PNG was saved as grayscale, so those templates match exactly
    as before (3 identical channels correlates the same as 1). Kept as an
    opt-out flag rather than ripping out the grayscale path, in case some
    future template set turns out to need luminance-only matching."""
    if not os.path.isdir(template_dir):
        return None, 0.0
    target = crop(img, region)
    if upscale != 1:
        target = cv2.resize(target, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
    target_cmp = target if color else cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    best_name, best_score = None, 0.0
    for fname in os.listdir(template_dir):
        if not fname.lower().endswith(".png"):
            continue
        tmpl = cv2.imread(os.path.join(template_dir, fname), cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
        if tmpl is None:
            continue
        for scale in (0.8, 0.9, 1.0, 1.1, 1.2):
            th, tw = int(tmpl.shape[0] * scale), int(tmpl.shape[1] * scale)
            if th < 8 or tw < 8 or th > target_cmp.shape[0] or tw > target_cmp.shape[1]:
                continue
            resized = cv2.resize(tmpl, (tw, th))
            res = cv2.matchTemplate(target_cmp, resized, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)
            if score > best_score:
                best_score, best_name = score, fname.rsplit(".", 1)[0]

    if best_score >= threshold:
        return best_name, best_score
    return None, best_score


# ---- field readers --------------------------------------------------------

def read_gold(img) -> int:
    txt = match_digits(img, REGIONS["gold_readout"])
    return int(txt) if txt.isdigit() else None


# Confirmed live (2026-09-21): tesseract confuses this UI's bold "1" glyph
# for a pipe/exclamation mark surprisingly often ("Round 1" -> "Round |" or
# "Round !"), and "Battle"/"Battle Phase" renders in a lower-contrast color
# than "Deploy Phase" does, so it sometimes reads as "Battie" or worse. Both
# get normalized here before parsing rather than fixed in ocr_text() itself,
# since these are specific to this one text's font/coloring, not general OCR
# preprocessing fixes.
_ROUND_DIGIT_FIXUPS = str.maketrans({"|": "1", "!": "1", "I": "1", "l": "1"})
_PHASE_CHAR_FIXUPS = str.maketrans({"i": "l", "1": "l", "|": "l"})  # battie -> battle


def read_round_phase_time(img) -> tuple:
    """Returns (round_num, phase, time_left) -- any of which may be None
    if that field failed to parse.

    KNOWN GAP (2026-09-21): during pure Battle Phase (troops actively
    fighting, not the "Prepare to Battle!" transition), the "Battle Phase"
    label sometimes renders too low-contrast for tesseract to read at all
    -- confirmed against live captures (see data/live_debug/ investigation
    in bot-plan.md). round_num/phase legitimately come back None in that
    case. Not chased further for now: the bot only needs to *act* during
    Deploy Phase, and a None/unrecognized phase is the safe default (no
    action) rather than a bug that blocks anything -- revisit only if
    something downstream actually needs to distinguish "battle" from
    "unknown" instead of just "not deploy"."""
    rp_txt = ocr_text(img, REGIONS["round_phase_text"])
    round_num = None
    phase = None
    for line in rp_txt.splitlines():
        line = line.strip()
        if line.lower().startswith("round"):
            fixed = line.translate(_ROUND_DIGIT_FIXUPS)
            digits = "".join(ch for ch in fixed if ch.isdigit())
            round_num = int(digits) if digits else None
        else:
            norm = line.lower().translate(_PHASE_CHAR_FIXUPS)
            if "deploy" in norm:
                phase = "deploy"
            elif "battle" in norm or "prepare" in norm:
                # covers both the "Prepare to Battle!" transition and the
                # "Battle Phase" label itself -- bot can't act in either,
                # so they're not distinguished further right now
                phase = "battle"

    # same bright-white/low-saturation rendering as the gold digits (confirmed
    # by eye), just larger -- match_digits()'s default preset (gold's template
    # dir + mask) works directly, no new preset needed
    time_txt = match_digits(img, REGIONS["time_left_text"])
    time_left = int(time_txt) if time_txt.isdigit() else None
    return round_num, phase, time_left


def read_board_capacity(img) -> tuple:
    txt = ocr_digits(img, REGIONS["board_capacity_text"], whitelist="0123456789/")
    if "/" in txt:
        used, _, maxc = txt.partition("/")
        if used.isdigit() and maxc.isdigit():
            return int(used), int(maxc)
    return None, None


def read_own_traits(img) -> dict:
    """Match each of the (up to 3) own-trait badges against the trait icon
    library, paired with its count/threshold read from the tight
    own_trait_count_badges crop via template matching -- same technique as
    read_gold()/read_shop(), but with its own match_digits() mask preset:
    include_green=True and a low white_v_min=90, because trait-count
    digits render dim-gray, white, OR green depending on some in-game
    state not yet understood (see data/reference_trait_digits_isolated/
    filenames), unlike the single bright-white style gold/shop costs use.
    trim_right_frac=0.8 drops the dark pill's rounded right edge, which
    the relaxed mask would otherwise pick up as a spurious 4th glyph.

    match_digits() also matches the "/" separator as if it were a digit
    (no template scores above threshold for it, so it comes back "?") --
    this only keeps the first and last character of the raw result, e.g.
    "1?2" -> "1/2". Returns {trait_name_or_unmatched_id: "count/threshold"}.

    The trait icon library currently only has the handful of icons seen in
    the two calibration captures (see data/reference_traits/) -- unmatched
    icons come back keyed "unmatched_<slot index>"."""
    icon_region = REGIONS["own_trait_badges"]
    x0, y0, x1, y1 = icon_region
    n_slots = 3
    slot_h = (y1 - y0) / n_slots
    count_regions = REGIONS["own_trait_count_badges"]
    traits = {}
    for i in range(n_slots):
        icon_sub = (x0, y0 + i * slot_h, x1, y0 + (i + 1) * slot_h)
        name, score = match_icon(img, icon_sub, REFERENCE_TRAITS_DIR)
        raw = match_digits(
            img, count_regions[i],
            template_dir=REFERENCE_TRAIT_DIGITS_ISOLATED_DIR,
            white_v_min=90, include_green=True, trim_right_frac=0.8,
        )
        if not raw:
            continue  # empty slot
        count_txt = f"{raw[0]}/{raw[-1]}" if len(raw) >= 2 else raw
        key = name or f"unmatched_{i}"
        traits[key] = count_txt
    return traits


def _green_arrow_present(img, region, min_green_frac=0.08) -> bool:
    """True if `region` is substantially covered by the game's vivid green
    "will merge" up-arrow. Same green HSV range match_digits()'s
    include_green option already uses elsewhere in this file, for
    consistency -- this game's UI green appears to be one consistent hue
    across contexts (trait-count digits, this arrow)."""
    sub = crop(img, region)
    if sub.size == 0:
        return False
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 80, 80), (85, 255, 255))
    frac = float(np.count_nonzero(mask)) / mask.size
    return frac >= min_green_frac


def _shop_affordability(img) -> list:
    """Per shop slot: True if it looks affordable, False if visibly
    darkened/desaturated (the game's own "you can't afford this" cue),
    None if the crop couldn't be read at all.

    Calibrated 2026-09-21 against a single user-provided screenshot (one
    confirmed-unaffordable card, two confirmed-affordable cards -- see
    bot-plan.md for the actual HSV numbers) -- not yet confirmed against a
    live full-frame capture. Uses HSV mean brightness (V), both an
    absolute floor (a genuinely affordable card was never this dim in the
    one example seen) and a relative check against the brightest shop slot
    this same tick (guards against a whole-frame lighting/theme change
    making every card look uniformly dim, which would otherwise flag
    everything as unaffordable)."""
    ABS_V_FLOOR = 0.40
    REL_FRAC = 0.8

    values = []
    for region in REGIONS["shop_slots"]:
        sub = crop(img, region)
        if sub.size == 0:
            values.append(None)
            continue
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        values.append(float(hsv[:, :, 2].mean()) / 255.0)

    known = [v for v in values if v is not None]
    max_v = max(known) if known else None

    result = []
    for v in values:
        if v is None or max_v is None:
            result.append(None)
        else:
            darkened = (v < ABS_V_FLOOR) and (v < REL_FRAC * max_v)
            result.append(not darkened)
    return result


def read_shop(img) -> list:
    """Template-match each shop slot against data/reference_cards/, plus
    OCR the cost badge. Card identification will be weak/mostly "unknown"
    until the icon library (Task #2) has real coverage -- currently only
    seeded with noisy auto-extracted crops, not properly labeled yet.

    Two additional per-slot signals (2026-09-21), read directly off the
    game's UI instead of depending on card_id/cost being known at all --
    see ShopSlot in game_state.py and bot-plan.md's "Live gameplay
    feedback" section for why these exist:
      - will_merge: the game's own green "will merge" arrow (see
        _green_arrow_present()) -- much more reliable than trying to
        infer a merge from card_id, since card_id is mostly unrecognized
        right now.
      - affordable: the game's own darkened/desaturated "can't afford"
        cue (see _shop_affordability()) -- a fallback for when cost or
        gold fails to read numerically, which was causing missed buys.
    """
    affordability = _shop_affordability(img)
    slots = []
    for i, (card_region, cost_region, merge_region) in enumerate(
        zip(REGIONS["shop_slots"], REGIONS["shop_cost_badges"], REGIONS["shop_merge_badges"])
    ):
        # threshold raised from the match_icon() default 0.6 -- with only one
        # template per card and busy, similarly-composed fantasy portraits,
        # 0.6 let visually-different cards win false matches at scores as high
        # as 0.9 in spot checks (2026-09-22, see bot-plan.md). 0.8 is a rough
        # compromise, not a calibrated value -- still lets some wrong matches
        # through, still rejects some right ones; revisit once there are
        # multiple labeled examples per card to average over.
        name, score = match_icon(img, card_region, REFERENCE_CARDS_DIR, threshold=0.8)
        # shop costs are always a single digit -- direct whole-badge template
        # match (upscale=6 to match how reference_digits/ was captured) beats
        # match_digits()'s contour segmentation here, which was unreliable on
        # these particular small badges (see project doc for why).
        cost_digit, cost_score = match_icon(img, cost_region, REFERENCE_DIGITS_DIR, threshold=0.7, upscale=6)
        cost = int(cost_digit) if cost_digit and cost_digit.isdigit() else None
        will_merge = _green_arrow_present(img, merge_region)
        slots.append(ShopSlot(
            card_id=name, cost=cost,
            will_merge=will_merge, affordable=affordability[i],
        ))
    return slots


# Cached lazily: the 5 bench_slots crops from the calibration image, which
# is confirmed empty in both calibration screenshots -- used as a per-slot
# "what does nothing look like" reference so occupied/empty can be told
# apart by simple pixel difference instead of a hand-tuned color/texture
# threshold. Whichever bench_slots crop the live frame differs from this
# reference the least is presumably still empty.
_BENCH_EMPTY_REF = None

# How different a live bench-slot crop needs to be from the empty reference
# before it's called "occupied". Calibrated (2026-09-21) against every known-
# EMPTY bench slot available at the time (both calibration images plus 15
# live frames from a real match, all bench-empty in that clip) -- observed
# diff scores topped out around 26 (mostly a lighting/shadow quirk specific
# to slot 0, not real content). NOT yet verified against a real occupied
# slot -- no live capture with a bench unit has been available yet. Treat
# this threshold as a first guess to be corrected once we can see it
# misclassify (or not) against a real bench purchase.
BENCH_OCCUPIED_DIFF_THRESHOLD = 35


def _bench_empty_reference():
    global _BENCH_EMPTY_REF
    if _BENCH_EMPTY_REF is None:
        calib_path = os.path.join(os.path.dirname(__file__), "..", "data", "calib_deploy_phase.png")
        calib_img = cv2.imread(calib_path)
        if calib_img is None:
            _BENCH_EMPTY_REF = [None] * len(REGIONS["bench_slots"])
        else:
            _BENCH_EMPTY_REF = [crop(calib_img, r) for r in REGIONS["bench_slots"]]
    return _BENCH_EMPTY_REF


def read_bench(img) -> list:
    """Read the 5 bench slots. Occupied/empty is decided by diffing each
    slot against the known-empty reference crop from the calibration image
    (see BENCH_OCCUPIED_DIFF_THRESHOLD above) -- simpler and more grounded
    in real data than a hand-tuned color/texture heuristic, since we have
    an actual confirmed-empty reference for this exact region.

    KNOWN GAPS (both first-pass placeholders, not yet backed by a real
    occupied-bench capture -- see bot-plan.md):
      - star_level is hardcoded to 1 for every occupied slot. The star-badge
        mechanic is confirmed (N overlapping star icons near the unit, see
        bot-plan.md) but its exact position within a BENCH slot (as opposed
        to a board unit, which sits at an arbitrary, uncalibrated position)
        hasn't been measured against the live pipeline yet. This mostly
        doesn't break the shop-merge-trigger logic in decide.py, which only
        needs to know about owned 1-star copies to fire a merge buy -- it
        would just misjudge a bench unit that's already 2-star+.
      - elixir_spent is hardcoded to 0 (no real per-unit spend ledger yet).
        Only used as a last-resort tiebreak in decide.py's forced-sell
        logic, so this just makes that tiebreak a no-op for now rather than
        producing a wrong answer.
    card_id comes from match_icon() against the same (still mostly
    unlabeled/noisy) reference_cards library read_shop() uses, so expect a
    lot of None/low-confidence matches until that library is built out.
    """
    empty_refs = _bench_empty_reference()
    units = []
    for slot_region, empty_ref in zip(REGIONS["bench_slots"], empty_refs):
        live_crop = crop(img, slot_region)
        if empty_ref is not None:
            resized_ref = cv2.resize(empty_ref, (live_crop.shape[1], live_crop.shape[0]))
            diff = cv2.absdiff(live_crop, resized_ref).astype(np.float32).mean()
            occupied = diff > BENCH_OCCUPIED_DIFF_THRESHOLD
        else:
            occupied = False  # no reference available -- assume empty rather than guess
        if not occupied:
            continue
        # see the threshold=0.8 comment in read_shop() above -- same reasoning
        name, score = match_icon(img, slot_region, REFERENCE_CARDS_DIR, threshold=0.8)
        units.append(Unit(
            card_id=name,
            star_level=1,       # placeholder -- see KNOWN GAPS above
            elixir_spent=0,     # placeholder -- see KNOWN GAPS above
            traits=_card_traits_tuple(name),
            position=None,
        ))
    return units


def _card_traits_tuple(card_id):
    """Local import-free wrapper -- decide.py owns CARD_TRAITS, but
    recognize.py shouldn't import decide.py (perception layer staying
    independent of decision logic), so this just returns () for now.
    Filling in real per-card traits happens in decide.py's CARD_TRAITS
    table once the card icon library is built out; recognize.py's job is
    only to name the card, not know its trait."""
    return ()


# HSV range for the board's grass/hex-tile floor -- covers both the
# lighter hex fill and the darker hex-outline shade seen in a real capture
# (data/gameplay_runs/run_20260922_013429/frame_091.png). Anything that
# ISN'T this color inside own_board_area is assumed to be a standing unit
# (or its health bar/star badge/ability VFX) -- see read_board() below.
_BOARD_GRASS_HSV = ((30, 25, 40), (75, 255, 255))

# Board-crop-relative blobs smaller than this fraction of the crop's area
# are dropped as noise (ability sparkle particles, hex-grid seams the mask
# didn't fully catch, JPEG-ish compression speckle). Picked by eye against
# the one real calibration frame -- see bot-plan.md; not swept against a
# larger sample the way BENCH_OCCUPIED_DIFF_THRESHOLD was.
BOARD_BLOB_MIN_AREA_FRAC = 0.004

# A blob taller than this multiple of its own width is assumed to be an
# edge/wall rendering artifact bleeding into own_board_area rather than a
# real unit -- confirmed live (2026-09-22): two tall, narrow (roughly
# 15-100px wide, 400-560px tall) blobs showed up at the crop's extreme
# left/right edges in the one calibration frame, matching the arena
# wall/border decoration, not anything unit-shaped. Real units in that
# same frame were all under 2x taller than wide.
BOARD_BLOB_MAX_ASPECT = 2.5


def read_board(img) -> list:
    """Detect units standing on the player's OWN battlefield during Deploy
    Phase, by finding "not grass-colored" blobs inside own_board_area
    rather than by matching a calibrated hex grid (board_hexes above is
    still empty/TODO -- it's for a different purpose, reading an
    OPPONENT's board after navigating into it, which nothing calls yet).
    That's a deliberate shortcut: this only needs each unit's approximate
    on-screen position (to drag it to the sell zone -- see
    input_control.SELL_ZONE and main.py's execute_sell()), not which
    named hex it occupies, and blob-centroid detection gets there without
    needing to model the hex grid's exact row/column offset geometry.

    Returns a list of Unit (see game_state.py), each with:
      - position: (fx, fy) as FULL-FRAME fractions (not region-local) --
        this is what main.py needs to drag-to-sell a board unit directly,
        unlike bench units which sell via a fixed per-slot center
        (input_control.BENCH_SLOT_CENTERS).
      - card_id: best-effort match_icon() against REFERENCE_CARDS_DIR on a
        small crop around the blob centroid. LOW CONFIDENCE in practice --
        confirmed live (2026-09-22) that board units render as 3D
        isometric character models, visually quite different from the
        flat portrait-card art REFERENCE_CARDS_DIR's templates were
        cropped from (shop_slots crops) -- expect a lot of None/wrong
        matches until there's a template library actually built from
        board captures. Kept anyway rather than skipped, since a
        correct match is still useful when it happens and a wrong one
        just means an unknown-trait sell candidate, not a bad tap
        (position comes from the blob, not from card_id).
      - star_level: hardcoded 1, same placeholder as read_bench() and for
        the same reason -- no confirmed 2-star+ board capture existed to
        calibrate a star-badge reader against as of this session (see
        bot-plan.md). decide_profit_sell() stays inert for board units
        until this lands, exactly like it does for bench units.
      - elixir_spent: hardcoded 0, same placeholder as read_bench().

    NOTE: only meaningful during Deploy Phase -- during Battle Phase the
    same screen region shows units mid-fight (moving, overlapping, VFX-
    heavy), which this hasn't been tested against and is not expected to
    handle well. main.py only calls this when phase == "deploy" anyway
    (see decide_and_act()'s existing Deploy-Phase-only guard)."""
    h, w = img.shape[:2]
    region = REGIONS["own_board_area"]
    x0, y0, x1, y1 = region
    board_crop = crop(img, region)
    bh, bw = board_crop.shape[:2]
    if bh == 0 or bw == 0:
        return []

    hsv = cv2.cvtColor(board_crop, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, *_BOARD_GRASS_HSV)
    not_grass = cv2.bitwise_not(grass_mask)

    kernel = np.ones((3, 3), np.uint8)
    not_grass = cv2.morphologyEx(not_grass, cv2.MORPH_OPEN, kernel, iterations=1)
    not_grass = cv2.morphologyEx(not_grass, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(not_grass, connectivity=8)
    units = []
    for i in range(1, n):
        bx, by, bbw, bbh, area = stats[i]
        area_frac = area / (bw * bh)
        if area_frac < BOARD_BLOB_MIN_AREA_FRAC:
            continue
        long_side, short_side = max(bbw, bbh), max(min(bbw, bbh), 1)
        if long_side / short_side > BOARD_BLOB_MAX_ASPECT:
            continue  # edge/wall artifact, not a unit -- see the constant's docstring

        cx, cy = centroids[i]
        # full-frame fractions -- what execute_sell() actually needs to drag from
        fx = x0 + (cx / bw) * (x1 - x0)
        fy = y0 + (cy / bh) * (y1 - y0)

        # small fixed-size crop around the centroid for card_id matching --
        # reuses match_icon() against the same (low-confidence-here, see
        # docstring above) shop-card template library.
        half = 0.06  # ~half of a shop_slots box's width, in full-frame fractions
        icon_region = (
            max(0.0, fx - half), max(0.0, fy - half * 1.3),
            min(1.0, fx + half), min(1.0, fy + half * 1.3),
        )
        name, score = match_icon(img, icon_region, REFERENCE_CARDS_DIR, threshold=0.8)

        units.append(Unit(
            card_id=name,
            star_level=1,        # placeholder -- see docstring above
            elixir_spent=0,      # placeholder -- see docstring above
            traits=_card_traits_tuple(name),
            position=(fx, fy),
        ))
    return units


def _find_wreath_bbox(img, y_band=(0.08, 0.68)):
    """Locate the green laurel-wreath placement badge shown on the
    end-of-match results screen. Calibrated 2026-09-21 from two real
    examples: the user's own results-screen screenshot (2nd place,
    data/results_screen_2nd_place.png -- NOT itself a live-pipeline
    capture, see below) and a genuine live-pipeline frame that happened
    to be a 4th-place loss screen
    (data/gameplay_runs/run_20260921_204410/frame_013.png). Both screen
    variants (1st/2nd "podium" with Play Again/OK, 3rd/4th "loss" with
    Spectate/Leave) show the same wreath asset, just with a different
    number inside and different buttons below.

    Deliberately NOT a fixed REGIONS box: measured wreath vertical center
    at ~0.47 of frame height on the loss screen but ~0.30 on the podium
    screen (the podium layout has an extra reward-icons row below the
    wreath, which apparently pushes the wreath itself higher rather than
    being purely bottom-anchored) -- so this scans a wide band and finds
    the wreath by color instead of assuming one fixed position.

    FOUND A REAL FALSE-POSITIVE DURING TESTING, fixed before shipping: a
    plain global bbox over every green pixel in the band also matched the
    ordinary Deploy Phase battlefield -- the hex grid's grass is the same
    green hue range as the wreath leaves (color alone can't tell them
    apart, confirmed by direct pixel sampling), and it's large enough to
    clear a simple min-height floor too
    (data/gameplay_runs/run_20260921_204309/frame_015.png was the
    catch -- a "Round 2" transition banner sitting over a normal green
    battlefield). What actually separates them is SHAPE, not color: the
    wreath is two hollow, narrow (~half as wide as tall) leaf-clusters
    with a lot of black gap between individual leaves (~50-55% of their
    own bounding box is actually green), while the battlefield is one
    large, wide, nearly-solid green rectangle (~89% fill). So this uses
    connected-component analysis instead of a single global bbox, and
    only accepts components shaped like a wreath half -- confirmed this
    correctly rejects the battlefield frame while still finding both real
    examples (each resolves into exactly two matching components, the
    left and right leaf clusters split by the number in between).

    Returns a pixel bbox (x0, y0, x1, y1) covering every matching
    component (i.e. both leaf clusters plus the number gap between them),
    or None if nothing wreath-shaped is found in this band."""
    h, w = img.shape[:2]
    y0, y1 = int(y_band[0] * h), int(y_band[1] * h)
    sub = img[y0:y1]
    if sub.size == 0:
        return None
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 60, 60), (85, 255, 255))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    frame_area = h * w
    matches = []
    for i in range(1, n):  # label 0 is background
        cx, cy, cw, ch, area = stats[i]
        if cw <= 0 or ch <= 0:
            continue
        area_frac = area / frame_area
        aspect = cw / ch
        fill = area / (cw * ch)
        # Ranges measured from the two real examples (~0.019-0.026 area
        # fraction, ~0.50-0.51 aspect, ~0.53-0.55 fill) with generous
        # margin -- the battlefield false positive measured 0.18 area
        # fraction, 1.25 aspect, 0.89 fill, well outside all three.
        if 0.008 <= area_frac <= 0.05 and 0.30 <= aspect <= 0.75 and 0.35 <= fill <= 0.70:
            matches.append((cx, cy, cw, ch))

    if not matches:
        return None
    bx0 = min(m[0] for m in matches)
    by0 = min(m[1] for m in matches)
    bx1 = max(m[0] + m[2] for m in matches)
    by1 = max(m[1] + m[3] for m in matches)
    return (bx0, by0 + y0, bx1, by1 + y0)


def read_placement(img) -> int:
    """Read the final 1st/2nd/3rd/4th place from the end-of-match results
    screen. Returns 1-4, or None if this frame isn't a results screen (no
    wreath found), the number inside it couldn't be isolated, or it didn't
    match a known digit template confidently -- callers should treat None
    as "not this frame, try another" rather than an error, since this gets
    called against every saved frame of a run hunting for the one results
    screen among many gameplay frames (see bot/rl_train.py), and also every
    live tick now (see main.py) to detect + auto-continue past this screen.

    CALIBRATED 2026-09-21 from two real examples -- see
    data/reference_placement_digits/ (2.png, 4.png) and
    _find_wreath_bbox()'s docstring for provenance. **Only 2 of 4 possible
    placements have a template so far (2 and 4)** -- 1st and 3rd place
    will read as None (not a wrong guess, a real gap) until a screenshot
    of either shows up, the same "fix opportunistically from real
    gameplay" approach used for every other digit-template gap in this
    project. The "2" template's source screenshot was NOT captured by
    window_capture.py's own pipeline (different aspect ratio, no phone
    bezel -- see bot-plan.md), so the wreath-finding step above already
    handles that source's positioning without needing pixel-exact
    alignment; only the digit template's own pixels are used for the
    matching step below, so that mismatch doesn't carry through to the
    live-pipeline reads this function actually gets called with.

    Approach: find the wreath (see above), then within it isolate the
    number itself via a "bright and not green" mask (the number is a
    light lavender/silver gradient, distinctly less green and brighter
    than the leaves) -- same style of mask used to pull the two reference
    templates out of their source screenshots -- then template-match that
    crop against data/reference_placement_digits/ (multi-scale, same
    technique as match_icon() above, inlined here since the crop is a
    dynamically-found pixel region rather than one of REGIONS' fixed
    fractional boxes)."""
    wreath = _find_wreath_bbox(img)
    if wreath is None:
        return None
    x0, y0, x1, y1 = wreath
    sub = img[y0:y1, x0:x1]
    if sub.size == 0:
        return None

    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (30, 40, 40), (95, 255, 255))
    bright_mask = hsv[:, :, 2] > 120
    number_mask = (bright_mask & (green_mask == 0)).astype(np.uint8) * 255
    number_mask = cv2.morphologyEx(number_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nys, nxs = np.where(number_mask > 0)
    if len(nxs) == 0:
        return None
    pad = 4
    ny0, ny1 = max(0, int(nys.min()) - pad), int(nys.max()) + pad
    nx0, nx1 = max(0, int(nxs.min()) - pad), int(nxs.max()) + pad
    number_crop = sub[ny0:ny1, nx0:nx1]
    if number_crop.size == 0:
        return None
    target_gray = cv2.cvtColor(number_crop, cv2.COLOR_BGR2GRAY)

    best_name, best_score = None, 0.0
    if os.path.isdir(REFERENCE_PLACEMENT_DIGITS_DIR):
        for fname in os.listdir(REFERENCE_PLACEMENT_DIGITS_DIR):
            if not fname.lower().endswith(".png"):
                continue
            tmpl = cv2.imread(os.path.join(REFERENCE_PLACEMENT_DIGITS_DIR, fname), cv2.IMREAD_GRAYSCALE)
            if tmpl is None:
                continue
            for scale in (0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4):
                th, tw = int(tmpl.shape[0] * scale), int(tmpl.shape[1] * scale)
                if th < 8 or tw < 8 or th > target_gray.shape[0] or tw > target_gray.shape[1]:
                    continue
                resized = cv2.resize(tmpl, (tw, th))
                res = cv2.matchTemplate(target_gray, resized, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(res)
                if score > best_score:
                    best_score, best_name = score, fname.rsplit(".", 1)[0]

    if best_name is not None and best_score >= 0.5 and best_name.isdigit():
        return int(best_name)
    return None


def find_highlighted_leaderboard_row(img):
    """Detect the OTHER end-of-match screen: the full 4-player leaderboard
    ("Merge Tactics / Defeated!" or presumably "Victory!") shown after the
    Spectate/Leave interim prompt is dismissed -- a completely different
    layout from the wreath screens read_placement() handles above (no
    green wreath at all; instead 4 stacked player-row cards, each with a
    small ribbon badge in the corner, one row visually highlighted).

    Discovered 2026-09-21: the user reported the bot "keeps getting stuck"
    after correctly tapping Leave on a real 3rd/4th-place wreath screen
    live (confirmed working twice in the same session via saved frames --
    see bot-plan.md) -- sent a screenshot of what it was stuck on, and it
    turned out to be this entirely different screen, which
    _find_wreath_bbox() naturally can't find (there's no wreath -- each
    row instead has a small gold ribbon-shaped badge). So read_placement()
    correctly returned None here, but main.py had no fallback for "this
    IS an end-of-match screen, just not the wreath kind" and just sat
    there every tick with nothing to do. The real flow for a loss appears
    to be: wreath+Spectate/Leave interim screen -> tap Leave -> THIS
    leaderboard screen -> tap Play Again. (A win, per the user's original
    2nd-place screenshot, goes straight to a wreath+rewards+Play-Again/OK
    screen and skips this one entirely -- see read_placement()'s own
    docstring -- so this path has, so far, only ever been observed after
    a loss.)

    The player's own row is rendered in a distinct warm gold/orange
    highlight (the other 3 rows are neutral light grey) -- confirmed via
    direct HSV sampling of two real examples so far
    (data/results_screen_4th_defeated.png and a second live-captured
    instance): H~25-65 (yellow/orange), S~95-160, well separated from
    every other row's H~70-115, S~25-80.

    IMPORTANT false-positive history: an earlier version of this function
    matched on the highlighted row alone (wide, that hue/saturation range,
    single-row height) and it looked clean against the two real examples,
    but a full sweep of every saved frame turned up two real false
    positives once actually deployed: (1) the golden trophy/podium under
    the hero on the "Searching for opponent..." lobby screen -- same
    hue/saturation, and initially wide enough too (fixed by raising the
    width-fraction floor from 0.6 to 0.75, since the podium tops out
    around 0.6 and both real leaderboard rows are 0.81+); (2) an orange/
    gold ATTACK VFX flash during live Battle Phase (a spell-cast graphic
    lighting up across most of the screen width) -- same hue/saturation
    range AND wide enough to survive fix (1) too. That second one is the
    dangerous kind: misfiring mid-battle would mean tapping "Play Again"
    (or whatever's at that screen position) *during a live match*, not
    just idling on a menu. Fixed by requiring BOTH the row-shaped
    component AND a second, smaller, button-shaped component further
    down the frame in the same hue/saturation range (the real "Play
    Again" button) -- neither real false positive has anything
    button-shaped that low in the frame (checked directly), so requiring
    both to co-occur is a much stronger signal than either alone. Comes
    at basically no real cost: this screen always shows its Play Again
    button right below the highlighted row anyway.

    Returns the row's (x0, y0, x1, y1) bbox, or None if not found. Does
    NOT attempt to read the placement digit from the row's small ribbon
    badge -- that badge is rendered at a very different scale/style than
    the big wreath numbers read_placement() is calibrated for, and there
    are only two real examples to calibrate against so far. Not
    release-blocking: the match's placement still gets read correctly
    from the earlier wreath+Spectate/Leave frame in the same run (see
    rl_train.py), this function only needs to answer "is this an
    end-of-match screen at all" so main.py knows to tap Play Again."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (10, 80, 100), (45, 255, 255))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    frame_area = h * w

    row_bbox = None
    has_button = False
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if cw <= 0 or ch <= 0:
            continue
        area_frac = area / frame_area
        width_frac = cw / w
        height_frac = ch / h
        y_center_frac = (y + ch / 2) / h

        if (row_bbox is None and area_frac >= 0.03 and width_frac >= 0.75
                and 0.05 <= height_frac <= 0.22
                and 0.50 <= y_center_frac <= 0.85):
            row_bbox = (x, y, x + cw, y + ch)

        if (0.20 <= width_frac <= 0.55 and 0.03 <= height_frac <= 0.10
                and y_center_frac >= 0.85):
            has_button = True

    if row_bbox is not None and has_button:
        return row_bbox
    return None


def recognize(elixir_ledger: dict = None) -> GameState:
    """Top-level entry point. elixir_ledger tracks elixir_spent per unit
    across ticks (recognize.py can't see history from a single frame --
    the caller/decide loop must persist this itself, updated whenever a
    buy/merge/sell action is taken). NOT fully wired yet -- read_bench/
    read_board still raise; calling this end-to-end will fail until those
    are filled in. Individual read_* functions can be called and tested
    on their own already."""
    img = capture_frame()
    gold = read_gold(img)
    round_num, phase, time_left = read_round_phase_time(img)
    used, maxc = read_board_capacity(img)
    own_traits = read_own_traits(img)
    shop = read_shop(img)
    bench = read_bench(img)   # will raise for now
    board = read_board(img)   # will raise for now
    return GameState(
        round=round_num, phase=phase, time_left=time_left, gold=gold,
        board_capacity=(used, maxc), shop=shop, bench=bench, board=board,
        own_traits=own_traits, opponents={},
    )


if __name__ == "__main__":
    from window_capture import window_bounds
    b = window_bounds()
    print("Window bounds:", b)
    img = capture_frame()
    print("Captured frame shape:", img.shape)
    print("gold:", read_gold(img))
    print("round/phase/time:", read_round_phase_time(img))
    print("board_capacity:", read_board_capacity(img))
    print("own_traits:", read_own_traits(img))
    print("shop:", read_shop(img))
