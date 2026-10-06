"""
The actual play loop: capture -> recognize -> decide -> act, repeated.

Must run in real Terminal.app (needs Quartz via window_capture.py and
pyautogui via input_control.py) -- see README.md for one-time setup.

STATUS (2026-09-21): v1 -- takes real actions (buy / forced-sell) during
Deploy Phase, using decide.py's rule-based logic plus one pragmatic
bootstrap rule added here (see rule_based_choice() below). `board` is
always [] -- read_board() isn't implemented yet (needs board_hexes
calibration + dynamic unit detection, see bot-plan.md), so merge/
forced-sell decisions currently only see bench units, not units already
placed on the board. No lobby-trait commitment is wired in yet either
(decide_lobby_trait() exists in decide.py but nothing here calls it) --
buys are purely merge-opportunity + cheap-filler driven for now.

SAFETY: real taps only happen when CRMT_ACT=1 is set in the environment.
By default this is a DRY RUN -- it reads the game, computes what it would
buy/sell, and logs it, without touching the real account. Once a dry run's
decisions look sane, re-run with `CRMT_ACT=1 python3 main.py` to let it
actually tap.

REWARD/PUNISHMENT (2026-09-21): every tick's candidate actions are now
featurized (rl.py) and logged to <run_dir>/trajectory.jsonl, regardless of
CRMT_ACT. decide.py's rule-based logic still makes every real tap/sell by
default -- this is Phase 1 (data collection + warm-start), not yet a
behavior change. See rl.py's module docstring and bot-plan.md for the full
plan, including CRMT_RL_ACT (Phase 2, off by default) which would let the
learned policy actually choose the action instead.

END OF MATCH / AUTO-CONTINUE (2026-09-21): recognize.py's read_placement()
is now real (was a stub) -- every tick checks whether the current frame is
the end-of-match results screen (1st/2nd "podium" or 3rd/4th "loss", see
bot-plan.md) and, if so, logs the placement and skips the normal
buy/sell decision for that tick. With CRMT_AUTO_CONTINUE=1 (a separate
opt-in on top of CRMT_ACT, since this changes the operating mode from
"run N ticks and stop" to "queue new matches indefinitely, unattended"),
it also taps through to the next match (Play Again on 1st/2nd, Leave on
3rd/4th) and starts a fresh <run_dir> for the new match, so
bot/rl_train.py's one-episode-per-run-folder assumption still holds.
"""
import sys
import os
import json
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "capture"))

import cv2
import recognize as R
import rl
from rl import LinearPolicy
from game_state import GameState, ShopSlot
from decide import (
    decide_buys, decide_forced_sell, decide_profit_sell,
    pick_committed_trait, CARD_TRAITS,
)
from window_capture import window_bounds
import input_control as IC


# Was a hardcoded 2.0s; user asked for faster ticks (2026-09-21). Now
# CRMT_POLL_INTERVAL-tunable, default 0.5s. Caution: this is the *sleep*
# between ticks, not the tick period -- capture+recognize+decide still take
# their own time on top of it, so the real period is however long that
# processing takes plus this sleep, not exactly 500ms. Also watch for
# misreads that weren't happening at 2s: shop/bench UI mid-animation
# (buy pop, merge flash) can now get captured more often, which could
# produce a bad read that settles a moment later. If decisions start looking
# noisy/wrong that didn't before, try bumping this back up first
# (CRMT_POLL_INTERVAL=1.0 python3 main.py) before assuming it's a logic bug.
POLL_INTERVAL_SECONDS = float(os.environ.get("CRMT_POLL_INTERVAL", "0.5"))

# Every run auto-saves everything it captures -- no flag needed. Each run
# gets its own timestamped folder under data/gameplay_runs/ (no frame cap --
# this is meant to build up a real library of gameplay captures across runs
# for calibration/debugging, the same way the manual calibration screenshots
# and the mined gameplay video were used earlier in this project). Frames
# add up (roughly 1-1.5MB each at 2s intervals), so it's worth clearing out
# old data/gameplay_runs/ subfolders once they're no longer useful.
#
# Set CRMT_DEBUG_DIR to a specific folder path instead if you want a single
# named, capped folder (e.g. for a focused bug report) rather than a fresh
# uncapped timestamped one every run.
DEBUG_DIR = os.environ.get("CRMT_DEBUG_DIR")
DEBUG_MAX_FRAMES = 20  # only applies to the CRMT_DEBUG_DIR path -- auto-save is uncapped


def _new_auto_save_dir():
    """A fresh timestamped data/gameplay_runs/run_<...>/ path. Was a
    module-level constant computed once at import time; now a function so
    auto-continue (see execute_continue()/watch_loop()) can start a new
    one per match -- otherwise every match after the first would keep
    appending to the SAME trajectory.jsonl, conflating multiple episodes
    into one file and breaking bot/rl_train.py's one-episode-per-run-
    folder assumption."""
    return os.path.join(
        os.path.dirname(__file__), "..", "data", "gameplay_runs",
        "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

# Real taps/drags only fire when this is set. Defaults OFF (dry run) --
# see SAFETY note above. `CRMT_ACT=1 python3 main.py` to go live.
ACT = os.environ.get("CRMT_ACT") == "1"

# Phase 2 of the reward/punishment system (see rl.py) -- when set, the
# learned linear policy chooses the action each tick instead of
# rule_based_choice()'s hardcoded priority order. Defaults OFF: leave this
# off until data/rl_weights.json has been trained on a real batch of
# matches via bot/rl_train.py and its choices have been sanity-checked in
# dry-run logs first.
RL_ACT = os.environ.get("CRMT_RL_ACT") == "1"
RL = LinearPolicy.load()

# Whether to actually tap through the end-of-match screen and keep playing.
# Defaults OFF -- without it, main.py still detects + logs the placement
# (useful on its own: unblocks bot/rl_train.py) but just sits there once a
# match ends, same as before, until the user manually starts a new one.
# `CRMT_AUTO_CONTINUE=1 CRMT_ACT=1 python3 main.py` for real unattended
# continuous play -- see execute_continue() below. Requires CRMT_ACT=1 too
# (this alone won't tap anything real, matching every other action here).
AUTO_CONTINUE = os.environ.get("CRMT_AUTO_CONTINUE") == "1"

# Wired up 2026-09-22 -- see _maybe_commit_trait() below. Starts None
# (no commitment yet) and gets set at most once per run, the first tick
# build_state() reports round >= LOBBY_TRAIT_COMMIT_ROUND with at least
# one trait visible in read_own_traits(). Real opponent scouting still
# doesn't exist (decide_lobby_trait() in decide.py wants it and stays
# unused for now) -- this uses pick_committed_trait()'s own-progress
# fallback instead. Module-level so enumerate_candidates()/
# rule_based_choice() calls elsewhere keep working unchanged.
COMMITTED_TRAIT = None

# Merge Tactics reveals round 3 as the first "pick a trait" checkpoint in
# the versions this bot has been run against (see bot-plan.md) -- not
# reconfirmed independently this session, just carried over from
# decide_lobby_trait()'s existing docstring ("At round 3 (and optionally
# re-checked later)"). Committing once and never re-checking is a
# deliberate simplification -- re-committing mid-run would strand
# already-bought off-trait units with no sell logic reacting to the
# switch.
LOBBY_TRAIT_COMMIT_ROUND = 3

# Set once per run by watch_loop(); each tick's candidates + chosen action
# get appended here as one JSON line (see _log_trajectory_step()).
TRAJECTORY_PATH = None


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        return f"<error: {e}>"


def build_state(img):
    """Best-effort GameState from the current frame. Any field that fails
    to read comes back None (or an empty/placeholder collection) rather
    than raising, so one bad reading doesn't take down the whole tick --
    decide_and_act() is responsible for treating unreliable fields as
    "don't act" rather than guessing.

    board is always [] -- read_board() isn't implemented yet."""
    rpt = _safe(R.read_round_phase_time, img)
    round_num, phase, time_left = rpt if isinstance(rpt, tuple) else (None, None, None)

    gold = _safe(R.read_gold, img)
    if not isinstance(gold, int):
        gold = None

    board_cap = _safe(R.read_board_capacity, img)
    used, maxc = board_cap if isinstance(board_cap, tuple) else (None, None)

    shop = _safe(R.read_shop, img)
    if not isinstance(shop, list):
        shop = [ShopSlot(None, None)] * 3

    bench = _safe(R.read_bench, img)
    if not isinstance(bench, list):
        bench = []

    own_traits = _safe(R.read_own_traits, img)
    if not isinstance(own_traits, dict):
        own_traits = {}

    # Only meaningful during Deploy Phase -- see read_board()'s own
    # docstring for why (Battle Phase units are moving/overlapping/VFX-
    # heavy, untested). Skipping the call outside deploy also avoids
    # spending the per-tick cost of blob detection on a phase where
    # nothing would use the result anyway (decide_and_act() only acts
    # during deploy).
    if phase == "deploy":
        board = _safe(R.read_board, img)
        if not isinstance(board, list):
            board = []
    else:
        board = []

    return GameState(
        round=round_num, phase=phase, time_left=time_left, gold=gold,
        board_capacity=(used, maxc), shop=shop, bench=bench, board=board,
        own_traits=own_traits, opponents={},
    )


def execute_buy(slot_index, cost):
    label = f"BUY shop slot {slot_index} (cost {cost})"
    if not ACT:
        print(f"  [DRY RUN] would {label}")
        return
    print(f"  [ACT] {label}")
    IC.tap_fraction(*IC.SHOP_CARD_CENTERS[slot_index])


def execute_sell(detail):
    """detail: an enumerate_candidates() sell-candidate detail dict, either
    {"source": "bench", "bench_index": i, "unit": u} or
    {"source": "board", "board_index": i, "unit": u} (2026-09-22 -- board
    selling is new this session, see decide_profit_sell()'s docstring in
    decide.py for why). Bench drags from a fixed per-slot center
    (input_control.BENCH_SLOT_CENTERS); board drags from the unit's own
    detected (fx, fy) (read_board() in recognize.py) since board units
    don't sit at fixed screen positions. Both drag to the same
    IC.SELL_ZONE -- same underlying gesture, only the source point
    differs. NOT yet exercised against a real board unit live (no CRMT_ACT
    run has had a board-sell candidate actually get chosen yet as of this
    session) -- the drag mechanics are identical to the already-working
    bench path, but treat the first live board sell as a thing to watch
    closely, same caveat main.py gives every other freshly-wired action."""
    source = detail.get("source", "bench")
    unit = detail["unit"]
    if source == "board":
        label = f"SELL board unit at {unit.position} ({unit.card_id or 'unknown card'})"
        fx, fy = unit.position
    else:
        bench_index = detail["bench_index"]
        label = f"SELL bench slot {bench_index} ({unit.card_id or 'unknown card'})"
        fx, fy = IC.BENCH_SLOT_CENTERS[bench_index]
    if not ACT:
        print(f"  [DRY RUN] would {label}")
        return
    print(f"  [ACT] {label}")
    IC.drag_fraction(fx, fy, *IC.SELL_ZONE)


def execute_continue(label, button, placement=None):
    """Tap through an end-of-match screen so a new match can start.

    Takes the button to tap directly rather than inferring it from
    `placement` -- an earlier version picked Play Again for placement
    1/2 and Leave for 3/4, on the assumption that which SCREEN you see is
    determined by your placement. That's wrong: which screen appears
    depends on where you are in the end-of-match FLOW, not your final
    placement. A loss (3rd/4th) actually goes through it in two steps --
    first the wreath-only "Spectate"/"Leave" interim screen
    (recognize.py's read_placement(), still fires as soon as you're
    eliminated, match ongoing for the other players), THEN, after
    tapping Leave, a completely different full leaderboard "Defeated!"
    screen with its own "Play Again"/"OK" pair
    (recognize.py's find_highlighted_leaderboard_row()) -- and a win
    (1st/2nd) goes straight to a wreath+rewards "Play Again"/"OK" screen,
    skipping the interim step entirely (see both those functions'
    docstrings). Placement alone can't tell these apart -- e.g. this
    function got called with placement=4 for BOTH the interim
    Spectate/Leave screen (frame_013.png) and, in live testing right
    after, the leaderboard Defeated screen -- the caller in watch_loop()
    below now decides the actual button from which detector matched,
    always Leave for the wreath-interim screen and always Play Again for
    the leaderboard screen, and passes both straight through here.
    `placement` is now just for logging (None when it came from the
    leaderboard screen, since that path doesn't try to read the exact
    digit -- see find_highlighted_leaderboard_row()'s docstring).

    Gated in two layers: CRMT_ACT (same real-tap gate as execute_buy()/
    execute_sell()) and, on top of that, CRMT_AUTO_CONTINUE (see its own
    comment above) -- this is a bigger behavior change than one buy/sell
    tap, so it gets its own explicit opt-in rather than riding along with
    CRMT_ACT alone."""
    if not AUTO_CONTINUE:
        print(f"  [end of match] placement={placement} -- would tap {label!r} to continue "
              f"(CRMT_AUTO_CONTINUE=1 to actually keep playing)")
        return False
    if not ACT:
        print(f"  [DRY RUN] end of match, placement={placement} -- would tap {label!r}")
        return False
    print(f"  [ACT] end of match, placement={placement} -- tapping {label!r}")
    IC.tap_fraction(*button)
    return True


def _is_buyable(slot, gold):
    """True if this shop slot can be bought right now. Prefers hard
    numbers (cost <= gold) whenever both are known -- only falls back to
    the game's own visual "can afford" cue (recognize.py's
    ShopSlot.affordable, from _shop_affordability()) when a number is
    missing, e.g. from a gold/cost digit-template gap. Deliberately NOT an
    OR of both checks: verified against data/calib_deploy_phase.png that
    the visual heuristic alone isn't confidently calibrated enough to
    override a confident numeric "no" (that frame has cost=4 > gold=3 but
    the darkening heuristic reads it as affordable) -- so numbers win
    whenever both are known, and the visual cue only fills in for missing
    numbers, never contradicts present ones."""
    if slot.cost is not None and gold is not None:
        return slot.cost <= gold
    return slot.affordable is True


def enumerate_candidates(state, committed_trait=None):
    """Every action the reward/punishment policy (rl.py) considers this
    tick -- a full alternative set, not just whichever one gets executed.
    Always includes a no-op so there's a fallback candidate even when
    nothing affordable/sellable exists. Each candidate:
    {"kind": "buy"|"sell"|"noop", "detail": {...}, "feat": {...}}.

    Deliberately mirrors rule_based_choice()'s own affordability logic
    (cost <= state.gold, no card_id requirement) so every action the rules
    could pick is guaranteed to appear here too."""
    candidates = []
    merge_buys = decide_buys(state, committed_trait=committed_trait)
    merge_set = set(merge_buys)

    for i, s in enumerate(state.shop):
        if not _is_buyable(s, state.gold):
            continue
        detail = {"slot_index": i, "cost": s.cost, "is_merge": i in merge_set}
        candidates.append({
            "kind": "buy", "detail": detail,
            "feat": rl.featurize(state, "buy", detail),
        })

    for i, u in enumerate(state.bench):
        detail = {"source": "bench", "bench_index": i, "unit": u}
        candidates.append({
            "kind": "sell", "detail": detail,
            "feat": rl.featurize(state, "sell", detail),
        })

    # Board-sourced sell candidates (2026-09-22, user request: selling
    # already-merged units directly off the battlefield, not just the
    # bench, for the same buy->merge->sell profit loop -- see
    # decide_profit_sell()'s docstring in decide.py). "board_index" isn't
    # a stable slot number the way bench_index is (read_board() re-detects
    # blobs fresh every tick, in whatever order connectedComponentsWithStats
    # happens to label them) -- execute_sell() below uses the unit's own
    # position instead of this index for anything that matters (the drag
    # target); it's kept only for logging/debugging symmetry with bench.
    for i, u in enumerate(state.board):
        detail = {"source": "board", "board_index": i, "unit": u}
        candidates.append({
            "kind": "sell", "detail": detail,
            "feat": rl.featurize(state, "sell", detail),
        })

    candidates.append({
        "kind": "noop", "detail": {},
        "feat": rl.featurize(state, "noop"),
    })
    return candidates


def _find_sell_candidate(state, candidates, unit):
    """Locate `unit` (as returned by decide_profit_sell()/decide_forced_sell(),
    which pool state.bench + state.board -- see decide.py) among this
    tick's sell candidates, checking bench first then board. Unit doesn't
    define __eq__, so `in`/`.index()` fall back to identity comparison,
    which is exactly right here -- these are the SAME Unit objects
    build_state() put in state.bench/state.board and enumerate_candidates()
    then wrapped into candidate dicts, not reconstructed copies."""
    if unit in state.bench:
        bench_index = state.bench.index(unit)
        for idx, c in enumerate(candidates):
            if c["kind"] == "sell" and c["detail"].get("source") == "bench" \
                    and c["detail"]["bench_index"] == bench_index:
                return idx
    if unit in state.board:
        board_index = state.board.index(unit)
        for idx, c in enumerate(candidates):
            if c["kind"] == "sell" and c["detail"].get("source") == "board" \
                    and c["detail"]["board_index"] == board_index:
                return idx
    return None


def rule_based_choice(state, candidates, committed_trait=None):
    """Index into `candidates` -- same priority order main.py has always
    used, just reframed to pick from the shared candidate list instead of
    deciding+acting in one step (so both the rule-based path and the RL
    path in decide_and_act() go through the same candidates/logging).

    Priority:
      1. A buy that completes a merge (decide_buys()'s own top priority).
      2. NEW (2026-09-21): sell any bench unit that's already been merged
         (star_level >= 2) -- decide.py's decide_profit_sell(). This is the
         actual economic point of this ruler (buy->merge->sell for profit),
         so it's realized as soon as it's noticed, ahead of routine filler
         buying. Currently inert live until bench star_level reading is
         wired up -- see decide_profit_sell()'s docstring.
      3. PRAGMATIC V1 ADDITION (not from decide.py): if nothing merges and
         the bench has room, buy the cheapest affordable card as filler --
         decide.py's rule-based logic assumes a trait gets committed
         around round 3 and doesn't otherwise say what to buy before that,
         so without this the bot would just sit on gold round 1-2 doing
         nothing. This is a simple placeholder, not tested strategy --
         expect to revisit once we see how it plays out.
      4. If none of the above applies and the bench is full, force-sell
         the worst-fit unit to free a slot (decide.py's own logic; skipped
         if the bench isn't actually full, since the game auto-merges
         duplicates on a bench-full buy -- see bot-plan.md -- so "full"
         doesn't necessarily mean "can't buy").
      5. Otherwise, no-op.
    """
    merge_buys = decide_buys(state, committed_trait=committed_trait)
    remaining_gold = state.gold
    affordable_merge = []
    for i in merge_buys:
        slot = state.shop[i]
        if not _is_buyable(slot, remaining_gold):
            continue
        affordable_merge.append(i)
        if slot.cost is not None and remaining_gold is not None:
            remaining_gold -= slot.cost
        # else: cost unknown (this merge buy only cleared via the visual
        # afford cue) -- can't decrement a number we don't have. Harmless:
        # only affordable_merge[0] ever gets acted on this tick anyway.

    if affordable_merge:
        target = affordable_merge[0]
        for idx, c in enumerate(candidates):
            if c["kind"] == "buy" and c["detail"]["slot_index"] == target:
                return idx

    profit_sell = decide_profit_sell(state)
    if profit_sell is not None:
        idx = _find_sell_candidate(state, candidates, profit_sell)
        if idx is not None:
            return idx

    bench_room = 5 - len(state.bench)
    if bench_room > 0:
        buy_candidates = [
            (idx, c["detail"]["cost"]) for idx, c in enumerate(candidates)
            if c["kind"] == "buy"
        ]
        if buy_candidates:
            # Known costs sort first (cheapest wins among them); a
            # visual-only buy (cost unknown) is still a valid candidate,
            # just deprioritized below any candidate with a known price.
            idx, _cost = min(
                buy_candidates,
                key=lambda pair: (pair[1] is None, pair[1] if pair[1] is not None else 0),
            )
            return idx

    if bench_room <= 0:
        # decide_forced_sell()'s pool is state.bench + state.board (see
        # decide.py) -- before 2026-09-22 this branch only ever matched a
        # bench result (checked "sold in state.bench" and looked up
        # bench_index specifically), so a board pick from decide_forced_sell
        # silently fell through to noop even though decide.py had already
        # chosen it. _find_sell_candidate() below handles either source.
        sold = decide_forced_sell(state, committed_trait=committed_trait)
        if sold is not None:
            idx = _find_sell_candidate(state, candidates, sold)
            if idx is not None:
                return idx

    for idx, c in enumerate(candidates):
        if c["kind"] == "noop":
            return idx
    return len(candidates) - 1  # candidates always has a noop, but just in case


def _log_trajectory_step(candidates, chosen_index):
    """Append this tick's full candidate set + which one was taken to
    <run_dir>/trajectory.jsonl -- raw material for bot/rl_train.py once a
    match's placement can be read (see recognize.py's read_placement()).
    Written every tick (not buffered) so a Ctrl+C or crash mid-match still
    leaves a usable partial trajectory on disk."""
    if TRAJECTORY_PATH is None:
        return
    record = {
        "kinds": [c["kind"] for c in candidates],
        "candidates": [c["feat"] for c in candidates],
        "chosen_index": chosen_index,
    }
    try:
        with open(TRAJECTORY_PATH, "a") as fp:
            fp.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"  [trajectory log failed] {e}")


_AVAILABLE_TRAITS = sorted({t for traits in CARD_TRAITS.values() for t in traits})


def _maybe_commit_trait(state):
    """Set the module-level COMMITTED_TRAIT once, at the first tick that
    qualifies -- see the LOBBY_TRAIT_COMMIT_ROUND comment above for why
    this fires once rather than every tick. No-ops (returns without
    touching COMMITTED_TRAIT) if the round isn't known yet, is too early,
    or a trait was already committed."""
    global COMMITTED_TRAIT
    if COMMITTED_TRAIT is not None:
        return
    if not isinstance(state.round, int) or state.round < LOBBY_TRAIT_COMMIT_ROUND:
        return
    if not _AVAILABLE_TRAITS:
        return
    chosen = pick_committed_trait(state.own_traits, _AVAILABLE_TRAITS)
    if chosen is not None:
        COMMITTED_TRAIT = chosen
        print(f"  [trait commit] round={state.round} own_traits={state.own_traits} "
              f"-> committing to {chosen!r}")


def decide_and_act(state):
    """At most one action per tick -- keeps this observable/debuggable
    rather than firing several taps off what might already be a stale
    read (the next tick will re-read and re-decide anyway).

    Which action gets chosen: rule_based_choice() by default, or (only if
    CRMT_RL_ACT=1) the learned reward/punishment policy from rl.py -- see
    that module's docstring for why this defaults off. Either way, the
    full candidate set gets logged for later training."""
    # REVERTED 2026-09-21 (same day as the attempt): the user reported the
    # game lets you buy/sell/merge during Battle Phase, not just Deploy, so
    # this briefly allowed acting in both. Checked it against a real Battle
    # Phase frame before shipping it live, though -- REGIONS["shop_slots"]
    # doesn't show shop cards during Battle Phase at all in that capture;
    # it shows bridge/environment art in that screen position (the bench
    # tray shifts into roughly the old bench spot, empty). So a real tap
    # there during battle would land on the wrong thing, not on a card.
    # Deploy-Phase-only for real actions until a live capture shows what
    # actually needs to be tapped to buy mid-battle (a separate shop
    # button? does the layout shift entirely?) -- see bot-plan.md.
    if state.phase != "deploy":
        print(f"  phase={state.phase!r} -- not deploy, no action this tick "
              f"(battle-phase buying needs its own UI investigation first)")
        return
    if state.gold is None:
        print("  gold unreadable this tick -- buy affordability will fall back to "
              "the visual afford cue where possible (see _is_buyable())")

    _maybe_commit_trait(state)
    candidates = enumerate_candidates(state, committed_trait=COMMITTED_TRAIT)

    if RL_ACT:
        chosen_index = RL.select([c["feat"] for c in candidates])
    else:
        chosen_index = rule_based_choice(state, candidates, committed_trait=COMMITTED_TRAIT)

    chosen = candidates[chosen_index]
    if chosen["kind"] == "buy":
        execute_buy(chosen["detail"]["slot_index"], chosen["detail"]["cost"])
    elif chosen["kind"] == "sell":
        execute_sell(chosen["detail"])
    else:
        print("  nothing to do this tick (no affordable buy, nothing to sell)")

    _log_trajectory_step(candidates, chosen_index)


def watch_loop(max_iterations=None):
    """Poll the game state, decide, and act. Ctrl+C to stop. Pass
    max_iterations for a bounded test run instead of forever (this counts
    ticks, not matches -- a match boundary via auto-continue doesn't reset
    it)."""
    global TRAJECTORY_PATH

    if not window_bounds():
        raise RuntimeError(
            "iPhone Mirroring window not found -- open it and get into a "
            "match (Deploy Phase) before running this."
        )

    save_dir = DEBUG_DIR or _new_auto_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    TRAJECTORY_PATH = os.path.join(save_dir, "trajectory.jsonl")
    if DEBUG_DIR:
        print(f"[debug] CRMT_DEBUG_DIR set -- saving up to {DEBUG_MAX_FRAMES} "
              f"frames + raw OCR text to {save_dir}")
    else:
        print(f"[auto-save] saving every captured frame this run to {save_dir}")

    mode = "ON -- WILL TAP/DRAG THE REAL GAME" if ACT else "off (dry run -- logging only)"
    rl_mode = "ON -- learned policy is choosing actions" if RL_ACT else "off (rule-based governs; RL only logging/learning in the background)"
    continue_mode = "ON -- will tap through results screens and keep playing" if AUTO_CONTINUE else "off (stops at the results screen until you continue it manually)"
    print(f"Watching... ACT={mode} RL_ACT={rl_mode} AUTO_CONTINUE={continue_mode} (Ctrl+C to stop)")
    print(f"[reward system] trajectory -> {TRAJECTORY_PATH} "
          f"(run bot/rl_train.py afterward once a placement's been read)")
    i = 0
    while max_iterations is None or i < max_iterations:
        try:
            img = R.capture_frame()
        except Exception as e:
            print(f"[capture failed] {e}")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if DEBUG_DIR:
            if i < DEBUG_MAX_FRAMES:
                cv2.imwrite(os.path.join(save_dir, f"frame_{i:03d}.png"), img)
        else:
            cv2.imwrite(os.path.join(save_dir, f"frame_{i:03d}.png"), img)

        placement = _safe(R.read_placement, img)
        end_of_match = None  # (label, button, placement) once decided below
        if isinstance(placement, int):
            # Wreath-based screen. 1st/2nd never got an interim
            # elimination prompt (see execute_continue()'s docstring), so
            # a wreath screen showing 1/2 is always the win rewards
            # screen -- Play Again. 3rd/4th, seen here, is always the
            # interim Spectate/Leave prompt (the OTHER loss screen, the
            # full "Defeated!" leaderboard, has no wreath at all and is
            # handled by the elif below instead) -- Leave.
            if placement in (1, 2):
                end_of_match = ("Play Again", IC.PLAY_AGAIN_BUTTON, placement)
            elif placement in (3, 4):
                end_of_match = ("Leave", IC.LEAVE_BUTTON, placement)
            else:
                print(f"  [end of match] placement={placement!r} -- not 1-4, don't know what to tap")
        elif _safe(R.find_highlighted_leaderboard_row, img) is not None:
            # No wreath, but this is the OTHER end-of-match screen (full
            # leaderboard, "Merge Tactics Defeated!") -- see
            # find_highlighted_leaderboard_row()'s docstring. Exact
            # placement isn't read on this path yet, but the button is
            # unambiguous: this screen only ever shows Play Again/OK.
            end_of_match = ("Play Again", IC.PLAY_AGAIN_BUTTON, None)

        if end_of_match is not None:
            label, button, end_placement = end_of_match
            print(f"round=? phase='end_of_match' placement={end_placement}")
            continued = execute_continue(label, button, end_placement)
            if continued and not DEBUG_DIR:
                # Real match boundary -- start a fresh run folder so the
                # NEXT match's trajectory.jsonl doesn't get appended onto
                # this one's (see _new_auto_save_dir()'s docstring).
                time.sleep(3.0)  # let the requeue/loading transition pass
                save_dir = _new_auto_save_dir()
                os.makedirs(save_dir, exist_ok=True)
                TRAJECTORY_PATH = os.path.join(save_dir, "trajectory.jsonl")
                print(f"[auto-save] new match -- now saving to {save_dir}")
                i = 0
            i += 1
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        state = build_state(img)
        print(
            f"round={state.round} phase={state.phase} time_left={state.time_left} "
            f"gold={state.gold} board_cap={state.board_capacity} "
            f"shop={[(s.card_id, s.cost) for s in state.shop]} "
            f"bench={[(u.card_id, u.star_level) for u in state.bench]}"
        )
        try:
            decide_and_act(state)
        except Exception as e:
            print(f"  [decide/act error] {e}")

        i += 1
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else None
    watch_loop(max_iterations=max_iter)
