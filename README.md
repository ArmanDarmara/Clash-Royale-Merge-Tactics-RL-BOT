# Clash Royale Merge Tactics Bot

## How this actually runs (important)

This project's code lives here in ~/git/clash-royale-merge-bot (synced from the
Cowork session). But the GUI-automation part (screenshotting the iPhone
Mirroring window, clicking/dragging on it) CANNOT run through Cowork's device
shell -- that shell is a sandboxed Linux VM with no access to the real macOS
display/window server. It's file-I/O only.

So: Claude writes/edits the code here, but YOU run it yourself in a real
Terminal.app on this Mac. That's the only process that can actually see the
screen and move the cursor.

## One-time setup (already done)
1. `cd ~/git/clash-royale-merge-bot`
2. `python3 -m pip install -r requirements.txt` -- done, all deps installed.
3. Accessibility permission granted to Terminal -- confirmed working live
   (taps land correctly and register in-game).
4. Screen Recording permission granted to Terminal -- confirmed working live.
5. `brew install tesseract` -- done (the pytesseract Python package needs the
   actual `tesseract` CLI binary too).

## How to run it right now
`main.py` is now v1 -- it can actually buy and sell for you, not just watch.
**It defaults to a DRY RUN**: it reads the game, decides what it would buy or
sell, and prints that decision, but doesn't touch the real game until you
turn it on. Recommended: watch a dry run first, check the decisions look
sane, then go live.

1. Open iPhone Mirroring, get into a Merge Tactics match.
2. Dry run (safe, logs only):
   ```
   cd ~/git/clash-royale-merge-bot/bot
   python3 main.py
   ```
3. Once that looks reasonable, go live (it will actually tap/drag):
   ```
   CRMT_ACT=1 python3 main.py
   ```
   (either way, optionally `python3 main.py 20` / `CRMT_ACT=1 python3 main.py 20`
   to stop after 20 iterations instead of running until Ctrl+C)
   Tick interval defaults to 500ms now (was a hardcoded 2s) -- tune with
   `CRMT_POLL_INTERVAL=1.0 python3 main.py` if reads start looking noisy at
   the faster pace (see bot-plan.md).
3b. **New (2026-09-21): fully unattended continuous play.** Add
   `CRMT_AUTO_CONTINUE=1` on top of `CRMT_ACT=1` and the bot will now detect
   the end-of-match results screen on its own and keep queuing new matches
   with no input from you:
   ```
   CRMT_AUTO_CONTINUE=1 CRMT_ACT=1 python3 main.py
   ```
   A loss (3rd/4th) actually goes through TWO different screens -- first
   a wreath-only "Spectate"/"Leave" prompt the moment you're eliminated
   (taps "Leave"), then, after that, a completely different full
   leaderboard "Merge Tactics Defeated!" screen with its own "Play
   Again"/"OK" (taps "Play Again"). A win (1st/2nd) skips straight to a
   wreath+rewards "Play Again"/"OK" screen. **Fixed 2026-09-22**: an
   earlier version only recognized the first (wreath) screen and picked
   the button from placement alone, which got the second screen wrong --
   it doesn't have a wreath at all, so the bot just sat there idling on
   it forever (this is what "it doesn't play again" turned out to be,
   confirmed against a real screenshot the user sent of exactly that).
   Both screens are now detected and handled correctly -- see bot-plan.md
   for the full detector writeup (and its false-positive history: two
   other bright orange/gold UI elements, a lobby trophy and a live
   Battle-Phase attack VFX, briefly false-matched during tuning and are
   now excluded). Each real continue starts a fresh
   `data/gameplay_runs/run_<timestamp>/` folder so `rl_train.py` sees one
   episode per file. Without `CRMT_AUTO_CONTINUE`, the bot still detects
   both screens and logs the placement when it has one, but just idles
   once a match ends, same as before.
   **Watch the first live run's "Play Again" tap closely** -- its
   coordinate was derived from a user screenshot rather than a native
   capture, so it's slightly less certain than every other calibrated
   coordinate in this project (see bot-plan.md).
4. Each tick prints a state line, then either `[DRY RUN] would BUY/SELL ...`
   or `[ACT] BUY/SELL ...` (or a reason nothing happened that tick).
   Buying/selling only happens during **Deploy Phase** right now -- the user
   reported it should also be possible mid-battle, but a saved Battle Phase
   capture showed the shop-row screen position displaying bridge/environment
   art instead of cards, so that's on hold until a live capture shows the
   real tap target during battle (see bot-plan.md).
5. Report back anything that looks wrong -- a crash, a bad decision, a tap
   that lands in the wrong place, a sell that doesn't register, etc. That's
   the real signal for what to fix next.

**Every run now auto-saves its captures**, no flag needed. Each run writes
every frame it captures to its own timestamped folder,
`data/gameplay_runs/run_<YYYYMMDD_HHMMSS>/`, so there's always a record of
what the bot actually saw without you doing anything extra. (`CRMT_DEBUG_DIR`
still works as an override if you want a single named, 20-frame-capped folder
for a specific bug report instead.)

## Reward/punishment system (new, 2026-09-21)
Every tick now also feeds a small learned "reward/punishment" policy
(`bot/rl.py`) -- placement-based: a bigger reward for 1st, a smaller one for
2nd, a punishment for 3rd/4th. **It doesn't drive real actions yet** -- the
rule-based logic above still makes every real tap/sell. Right now it just
logs every tick's options to `data/gameplay_runs/run_.../trajectory.jsonl`
so there's something to learn from. Once a live run plays a match all the
way to the actual results screen (just let it keep going instead of
stopping early), run:
```
cd ~/git/clash-royale-merge-bot/bot
python3 rl_train.py
```
to train on whatever runs it can find a placement for. Full plan and status
in bot-plan.md and in `rl.py`'s own module docstring.

Two known limitations, updated 2026-09-22 (see bot-plan.md for the full
reasoning):
- **Board reading + selling is now implemented** (2026-09-22) -- `read_board()`
  detects units already placed on the field via position (blob/contour
  detection against the grass-colored background), not a calibrated hex
  grid, and the profit-sell/forced-sell logic now checks the board as well
  as the bench. Per-unit `card_id` on the board is low-confidence (board
  units render as 3D isometric models, quite different from the flat shop
  portraits the card templates were cropped from) and `star_level` is still
  a hardcoded placeholder (no confirmed 2-star+ example exists in the data
  yet, see below). It still doesn't look at the **opponent's** board at all
  -- no calibration data exists for that.
- There's now a real (21-card) card-icon library (`data/reference_cards/`),
  replacing the earlier unusable auto-extracted junk -- see the Status
  section below for its accuracy caveats. Merge detection still mostly
  doesn't need it day-to-day: the game itself shows a green up-arrow badge
  on a shop card when buying it would complete a merge, and
  `recognize.py`/`decide.py` read that directly and prioritize those buys.

## Status
- [x] iPhone Mirroring drivable via clicks/drags/scrolls -- confirmed live.
- [x] All Python deps installed and working (opencv-python, numpy, Pillow,
      pyautogui, pyobjc-framework-Quartz, pytesseract + tesseract CLI).
- [x] `capture/window_capture.py` -- grabs the iPhone Mirroring window.
- [x] `capture/input_control.py` -- fraction-based tap/drag targets, tested
      live. Includes `SELL_ZONE` now (drag-to-sell target, unconfirmed live).
- [x] `bot/recognize.py` -- HUD/shop/trait numbers all read correctly (gold,
      round/phase/time, board capacity, shop costs, own traits). `read_bench()`
      now implemented (occupied/empty detection verified against real data;
      star level and elixir-spent are still placeholders -- see bot-plan.md).
      `read_board()` still not implemented.
- [x] Shop merge-arrow + affordability detection (2026-09-21) --
      `ShopSlot.will_merge`/`affordable` now read the game's own green
      "will merge" badge and card-darkening "can't afford" cue directly, so
      merge buys are prioritized correctly and gold/cost OCR gaps no longer
      block a buy the game itself shows as affordable (numbers still win
      whenever both a number and the visual cue are available).
- [x] `bot/decide.py` -- rule-based v1, passing its smoke tests. Fixed a real
      crash bug this session (would crash on an unreadable board_capacity).
- [x] `bot/main.py` -- **v1: takes real buy/sell actions**, gated behind
      `CRMT_ACT=1` (see "How to run it right now"). Auto-saves every run's
      captures to `data/gameplay_runs/run_<timestamp>/` now, no flag needed.
      First live run found gold misreading as 1 instead of 5 (bad digit
      template) -- fixed; next step is running it again and reporting what
      happens now that gold reads correctly.
- [x] `read_board()` (2026-09-22) -- implemented via position-detection
      (blob/contour analysis against the grass-colored background), not a
      calibrated hex grid. Detects standing board units and their screen
      position well enough to drive a sell-drag; per-unit `card_id` is
      low-confidence (3D board models vs. flat shop-portrait templates) and
      `star_level` is still a hardcoded placeholder. No placement strategy
      yet (nothing chooses where a *new* unit goes on the board).
- [x] Card-icon recognition library (2026-09-22) -- 21 cards, hand-labeled
      from real Deploy-Phase gameplay frames, in `data/reference_cards/`.
      Also fixed a real `match_icon()` bug along the way: it was matching on
      grayscale, a poor discriminator for portraits that differ mainly by
      hue -- switched to color (BGR) matching, which measurably reduced (but
      did not eliminate) false-positive card matches. Threshold raised to
      0.8 for card matching specifically. Treat `card_id` reads as
      probabilistic, not ground truth, until there's a multi-template
      library.
- [x] Lobby-trait commitment (2026-09-22) -- `decide_lobby_trait()` itself
      still can't run for real (needs opponent scouting that doesn't exist),
      but `main.py` now calls a practical stand-in, `pick_committed_trait()`:
      commits to whichever trait your own shop luck has leaned into hardest,
      once at round 3 (reads `read_own_traits()`, which the bot already had).
      Found + fixed a related bug while wiring this: `decide_forced_sell()`'s
      trait-fit scoring was reading a field the recognizer never actually
      populated, so trait-fit silently never influenced forced-sells before
      this fix.

- [x] `bot/rl.py` -- reward/punishment policy (linear, trained via
      REINFORCE from match placement). Logs every tick to
      `trajectory.jsonl` now; doesn't drive real actions yet
      (`CRMT_RL_ACT=1` will, once trained -- see above and bot-plan.md).
- [x] `bot/recognize.py`'s `read_placement()` -- **now real** (2026-09-21),
      reads the end-of-match wreath badge via connected-component shape
      detection (color alone false-matched the battlefield -- fixed) +
      digit template matching. Only 2nd and 4th place have reference digit
      templates so far (`data/reference_placement_digits/`); 1st and 3rd
      still read as `None` until a screenshot of either shows up.
- [x] `main.py`'s `CRMT_AUTO_CONTINUE` (2026-09-21) -- taps through the
      results screen (Play Again / Leave) and rotates to a new run folder
      per match, for fully unattended continuous play -- see "How to run it
      right now" above.
- [x] **First real RL training happened** (2026-09-21) -- swept old saved
      runs, found 2 real episodes (both 4th place), trained
      `data/rl_weights.json` on them for real via `rl_train.py`. Still very
      little/lopsided data -- `CRMT_RL_ACT` stays off by default until
      there's more, ideally some wins too.
- [x] `decide.py`'s `decide_profit_sell()` (2026-09-21, extended 2026-09-22
      to cover the board too) -- sells any merged (star>=2) bench OR board
      unit for profit, the actual point of Spirit Empress's
      buy->merge->sell economics. **Still inert live** -- needs
      bench/board star-level reading calibrated first (both still hardcode
      star_level=1; actively searched ~3550 real bench crops this session
      for a confirmed 2-star+ example and found none -- genuinely blocked on
      data, not a skipped task, see bot-plan.md) -- activates automatically
      once a real example turns up.
- [ ] Board placement STRATEGY (choosing where a *newly bought/merged* unit
      goes on the hex grid) -- not started. `read_board()` (above) only
      reads what's already there; front-line/back-line, countering the
      opponent, etc. is still an open design question, not just a
      calibration gap -- see bot-plan.md.

Full up-to-date plan and history: the "Clash Royale Merge Tactics" Claude
project (`claude/bot-plan.md`), kept current every session -- more detail
than this file, check there first if this looks stale.
