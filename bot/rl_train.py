"""
Offline trainer for the reward/punishment policy (rl.py).

Run this by hand any time after one or more live runs
(`python3 main.py` / `CRMT_ACT=1 python3 main.py`), e.g.:

    cd ~/git/clash-royale-merge-bot/bot
    python3 rl_train.py

For every data/gameplay_runs/run_<timestamp>/ folder with a
trajectory.jsonl (written automatically by main.py, see rl.py's module
docstring for why this is decoupled from the live loop): tries
recognize.read_placement() against its saved frames (most likely near the
end of the run, so it's searched last-frame-first) to find the match's
final placement. If found, computes the reward (rl.PLACEMENT_REWARDS) and
runs one rl.LinearPolicy.reinforce_update() over that run's trajectory,
then saves the updated weights. Each run is only ever trained on once
(tracked in data/rl_weights.json's history by run directory name), so
re-running this script after new matches is always safe/idempotent --
it'll just pick up whatever's new.

CURRENT STATUS (2026-09-21): recognize.read_placement() always returns
None -- it hasn't been calibrated yet (needs a real end-of-match results
screen, not captured so far). So right now this script will find zero
trainable runs and say so -- that's expected, not a bug. Once a live run
gets allowed to play a match all the way to actual elimination/victory,
auto-save will capture that screen, read_placement() can get filled in
against it, and this script will start finding real placements to learn
from -- including retroactively, in every already-saved run's frames.
"""
import glob
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "capture"))

import cv2
import recognize as R
import rl
from rl import LinearPolicy, PLACEMENT_REWARDS

GAMEPLAY_RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gameplay_runs")


def find_placement_for_run(run_dir):
    """Search this run's saved frames for the end-of-match results screen,
    most-recent-first (a match's placement screen is near the end of the
    run, and searching that direction means we don't waste time OCR'ing
    every early-round frame first)."""
    frames = sorted(glob.glob(os.path.join(run_dir, "frame_*.png")), reverse=True)
    for frame_path in frames:
        img = cv2.imread(frame_path)
        if img is None:
            continue
        placement = R.read_placement(img)
        if placement in (1, 2, 3, 4):
            return placement, frame_path
    return None, None


def load_trajectory(run_dir):
    path = os.path.join(run_dir, "trajectory.jsonl")
    if not os.path.exists(path):
        return None
    steps = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a truncated last line from a mid-tick crash/Ctrl+C
    return steps


def main():
    policy = LinearPolicy.load()
    already_trained = {h["run"] for h in policy.history}

    run_dirs = sorted(
        d for d in glob.glob(os.path.join(GAMEPLAY_RUNS_DIR, "run_*")) if os.path.isdir(d)
    )
    if not run_dirs:
        print(f"No runs found under {GAMEPLAY_RUNS_DIR} -- nothing to train on yet.")
        return

    trained_this_pass = 0
    skipped_no_trajectory = 0
    skipped_already_trained = 0
    skipped_no_placement = 0

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        if run_name in already_trained:
            skipped_already_trained += 1
            continue

        trajectory = load_trajectory(run_dir)
        if not trajectory:
            skipped_no_trajectory += 1
            continue

        placement, frame_path = find_placement_for_run(run_dir)
        if placement is None:
            skipped_no_placement += 1
            continue

        reward = PLACEMENT_REWARDS[placement]
        policy.reinforce_update(trajectory, reward)
        policy.history.append({
            "run": run_name,
            "placement": placement,
            "reward": reward,
            "placement_frame": os.path.basename(frame_path),
            "steps": len(trajectory),
        })
        trained_this_pass += 1
        print(f"[trained] {run_name}: placement={placement} reward={reward:+.1f} "
              f"({len(trajectory)} ticks)")

    if trained_this_pass:
        policy.save()
        print(f"\nSaved updated weights -> {os.path.abspath(rl.WEIGHTS_PATH)}")
        print(f"Total episodes trained on (all time): {policy.episodes_trained}")
    else:
        print("No new runs to train on this pass.")

    print(
        f"\nSummary: trained {trained_this_pass}, "
        f"already-trained {skipped_already_trained}, "
        f"no trajectory.jsonl {skipped_no_trajectory}, "
        f"no placement found {skipped_no_placement}"
    )
    if skipped_no_placement and not trained_this_pass and not skipped_already_trained:
        print(
            "\nNote: read_placement() isn't calibrated yet (see recognize.py), "
            "so no run can be scored yet -- this is expected until a live run "
            "plays a match all the way to the results screen and that gets "
            "wired up. Nothing wrong with your data; it's just waiting."
        )


if __name__ == "__main__":
    main()
