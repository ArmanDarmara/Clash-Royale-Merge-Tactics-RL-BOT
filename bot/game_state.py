"""
Shared schema for what recognize.py produces each tick and decide.py consumes.

Kept as plain dicts/dataclasses (not tied to any perception implementation)
so decide.py's logic can be written and unit-tested right now, independent
of screen-coordinate calibration.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Unit:
    card_id: str            # e.g. "knight", "princess" -- matches a template name
    star_level: int          # 1, 2, or 3
    elixir_spent: int        # total elixir put into this unit's merge chain so far
    traits: tuple             # e.g. ("noble",) or ("undead", "clan") if dual-trait
    position: Optional[tuple] = None  # (fx, fy) full-frame fraction coords if on
        # board (2026-09-22 -- see recognize.py's read_board()), None if on bench.
        # Originally meant as a (row, col) hex coord -- changed to screen-fraction
        # coords instead, since that's what execute_sell() actually needs to drag
        # a board unit to the sell zone, and board_hexes (recognize.py REGIONS) was
        # never calibrated / is for a different purpose (reading an opponent's
        # board). Nothing else read this field before this change (grepped clean).


@dataclass
class ShopSlot:
    card_id: Optional[str]   # None if slot is empty
    cost: int                # elixir cost to buy (TBD: confirm this is really per-card cost
                              # and not something else -- see recognize.py TODO)
    # Added 2026-09-21: two signals read directly off the game's own UI,
    # deliberately NOT dependent on card_id (the icon library) or on cost
    # (the digit templates) -- see recognize.py's read_shop() for how these
    # get set.
    will_merge: bool = False        # game shows a green up-arrow on a shop
                                     # card when buying it completes a merge
    affordable: Optional[bool] = None  # game visibly darkens/desaturates a
                                     # card you can't afford -- None means
                                     # "couldn't tell" (e.g. bad crop), not
                                     # "unaffordable"


@dataclass
class OpponentInfo:
    name: str
    health: int
    traits: dict              # trait_name -> current count, from the scouted overlay


@dataclass
class GameState:
    round: int
    phase: str                # "deploy" or "battle"
    time_left: int             # seconds remaining in current phase
    gold: int                  # current elixir/gold balance
    board_capacity: tuple      # (used, max) e.g. (4, 5)
    shop: list                 # list[ShopSlot], always length 3
    bench: list                # list[Unit], units not yet placed
    board: list                # list[Unit], units placed on the hex board
    own_traits: dict           # trait_name -> count among bench+board units
    opponents: dict            # name -> OpponentInfo, only for scouted opponents
    ruler: str = "spirit_empress"


# Trait activation thresholds confirmed from the Oct 2025 patch notes research.
# Traits activate bonuses at these troop counts; used by decide.py to judge
# how close a comp is to its next breakpoint.
TRAIT_THRESHOLDS = (2, 4, 6)
