"""
Click/drag on the iPhone Mirroring window using pyautogui.

Must be run in a real Terminal.app process (Accessibility permission granted
to Terminal), NOT through Cowork's device shell.

IMPORTANT coordinate-space note: window_bounds() (from Quartz) reports the
window's size in macOS POINTS, but window_capture.py's screenshots (via
`screencapture`) are saved at native PIXEL resolution -- 2x the point size
on this Retina MacBook Air (confirmed: calib_deploy_phase.png / calib_
opponent_view.png both came back 784x1722px). Mixing pixel-measured
coordinates directly into a pyautogui click (which operates in points)
would land everything at 2x the intended position.

Fix: every tap target below is stored as a FRACTION (0..1) of the window's
width/height, measured from the 784x1722 calibration screenshots. Fractions
are resolution/scale independent -- tap_fraction() multiplies them by the
CURRENT window bounds (in points) at call time, so this works regardless of
backing-scale-factor or the window being resized/moved.
"""
import time
import subprocess
import pyautogui
from window_capture import window_bounds

pyautogui.PAUSE = 0.05  # small delay between pyautogui calls


def focus_window(app_name="iPhone Mirroring"):
    """Bring the iPhone Mirroring window frontmost before tapping. Without
    this, a click at the right coordinates can land as just a
    focus/activate click on the window itself rather than passing through
    as a tap on the phone -- suspected cause of a tap landing on the right
    spot but not registering in-game. Cheap to call before every action."""
    try:
        subprocess.run(
            ["osascript", "-e", f'tell application "{app_name}" to activate'],
            check=True, capture_output=True, timeout=3,
        )
        time.sleep(0.15)  # give the window manager a moment to raise it
    except Exception as e:
        print(f"[warn] could not focus {app_name}: {e}")


# ---- tap targets, as (fx, fy) fractions of the window, measured from ----
# ---- data/calib_deploy_phase.png and data/calib_opponent_view.png     ----
# (both 784x1722px live captures, Round 3 Deploy Phase / opponent scouting)

SHOP_CARD_CENTERS = [
    (0.3018, 0.9161),  # leftmost shop slot
    (0.5089, 0.9161),  # middle shop slot
    (0.7156, 0.9161),  # rightmost shop slot
]

BENCH_SLOT_CENTERS = [
    (0.2149, 0.7927),
    (0.3514, 0.7927),
    (0.4879, 0.7927),
    (0.6245, 0.7927),
    (0.7610, 0.7927),
]

BACK_BUTTON = (0.5006, 0.9204)  # returns from opponent-scouting view

# Where to drop a unit to sell it. Confirmed the mechanic is "drag a troop
# back to the shop" (bot-plan.md) and separately spotted a live frame
# mid-drag showing a "Drag troops here to sell" zone replacing the shop
# row -- this reuses the middle shop-card position as that drop target,
# since the zone appears to occupy roughly the same area. NOT yet
# confirmed with a real drag-to-sell -- first live test will tell us if
# this needs adjusting.
SELL_ZONE = SHOP_CARD_CENTERS[1]

# Top HUD row, 4 equal columns (player portraits -- tap to scout that player).
# Column order on screen is whatever the match assigned, NOT fixed to a
# specific opponent -- caller must OCR/compare the name label above each
# portrait (see recognize.py) to know who's being tapped, and skip the
# column showing your own name.
OPPONENT_PORTRAIT_CENTERS = [
    (0.1250, 0.1161),
    (0.3750, 0.1161),
    (0.6250, 0.1161),
    (0.8750, 0.1161),
]

# ---- end-of-match continue buttons (2026-09-21) -------------------------
# Two screen variants after a match ends (see bot-plan.md): 1st/2nd place
# shows a "Play Again" / "OK" button pair; 3rd/4th shows "Spectate" /
# "Leave". Same left=orange/right=blue layout both times. main.py picks
# which one to tap from recognize.py's read_placement() (1st/2nd -> Play
# Again, 3rd/4th -> Leave, never Spectate/OK) -- see main.py's
# execute_continue().

# Measured directly from data/gameplay_runs/run_20260921_204410/frame_013.png,
# a genuine live-pipeline capture of the 3rd/4th "loss" screen -- same
# capture pipeline as every other coordinate in this file, so this one is
# trustworthy without further live verification.
LEAVE_BUTTON = (0.7143, 0.9167)

# Measured from the user's own results-screen screenshot
# (data/results_screen_2nd_place.png) -- NOT itself a window_capture.py
# pipeline capture (different aspect ratio, no phone bezel margin visible;
# see bot-plan.md), so the raw fraction measured in that image doesn't
# directly line up with this file's coordinate space. Remapped using the
# bezel inset measured from frame_013.png above (content spans roughly
# x:[2.0%,98.0%] y:[4.4%,99.1%] of the real pipeline's captured window) --
# cross-checked by remapping that same screenshot's button-row Y position
# and comparing it to the Leave/Spectate row's independently-measured Y
# above: 0.9216 (remapped) vs 0.9167 (native) is a good enough match to
# trust the X position too, but this one hasn't been independently
# confirmed against a live "Play Again" tap yet -- watch the first live
# CRMT_AUTO_CONTINUE run closely.
PLAY_AGAIN_BUTTON = (0.3539, 0.9216)


def to_screen_coords(rel_x, rel_y, app_name="iPhone Mirroring"):
    """Convert PIXEL-space (rel_x, rel_y) -- e.g. a coordinate measured
    directly off a window_capture.py screenshot -- into absolute screen
    points for pyautogui. Kept for one-off debugging; prefer tap_fraction()
    for anything durable, since this assumes the screenshot pixel size
    matches the window's current backing scale."""
    bounds = window_bounds(app_name)
    if not bounds:
        raise RuntimeError(f"{app_name} window not found")
    wx, wy, _, _ = bounds
    return (wx + rel_x, wy + rel_y)


def to_screen_coords_fraction(fx, fy, app_name="iPhone Mirroring"):
    """Convert a (fx, fy) fraction of the window (0..1 each) into absolute
    screen points. This is scale-independent -- works whether the capture
    pipeline is 1x or 2x, and survives the window being resized."""
    bounds = window_bounds(app_name)
    if not bounds:
        raise RuntimeError(f"{app_name} window not found")
    wx, wy, ww, wh = bounds
    return (wx + fx * ww, wy + fy * wh)


def tap(rel_x, rel_y):
    """Pixel-space tap -- see to_screen_coords() caveat above."""
    focus_window()
    x, y = to_screen_coords(rel_x, rel_y)
    pyautogui.click(x, y)


def tap_fraction(fx, fy):
    """Preferred: tap at a (fx, fy) fraction of the window, e.g.
    tap_fraction(*SHOP_CARD_CENTERS[0])."""
    focus_window()
    x, y = to_screen_coords_fraction(fx, fy)
    pyautogui.click(x, y)


def drag(rel_x1, rel_y1, rel_x2, rel_y2, duration=0.25):
    focus_window()
    x1, y1 = to_screen_coords(rel_x1, rel_y1)
    x2, y2 = to_screen_coords(rel_x2, rel_y2)
    pyautogui.moveTo(x1, y1)
    pyautogui.dragTo(x2, y2, duration=duration, button="left")


def drag_fraction(fx1, fy1, fx2, fy2, duration=0.25):
    focus_window()
    x1, y1 = to_screen_coords_fraction(fx1, fy1)
    x2, y2 = to_screen_coords_fraction(fx2, fy2)
    pyautogui.moveTo(x1, y1)
    pyautogui.dragTo(x2, y2, duration=duration, button="left")


if __name__ == "__main__":
    # Calibration check: tap the leftmost shop-card slot and report where
    # that landed on screen, so we can visually confirm against the phone.
    b = window_bounds()
    print("Window bounds (points):", b)
    if b:
        fx, fy = SHOP_CARD_CENTERS[0]
        x, y = to_screen_coords_fraction(fx, fy)
        print(f"Tapping leftmost shop card at fraction ({fx},{fy}) -> screen ({x:.0f},{y:.0f})")
        tap_fraction(fx, fy)
        time.sleep(0.5)
        print("Done — check the phone: did it select/highlight the leftmost shop card?")
