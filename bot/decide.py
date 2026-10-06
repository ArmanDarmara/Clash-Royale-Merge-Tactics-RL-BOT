"""
Rule-based decision layer (v1). Pure logic, no screen I/O -- takes a
GameState (see game_state.py) and returns actions. Deliberately kept
independent of recognize.py/input_control.py so it can be unit-tested
with synthetic states before perception is wired up.

Encodes:
  - the confirmed Spirit Empress buy->merge->sell economics
  - the forced-sell tiebreaker (rank by comp/trait fit, never by cost --
    every unmerged 1-star sale nets exactly -1 regardless of price)
  - round-3+ lobby scouting: commit to the trait least used among
    scouted opponents, to improve 3-star odds later
"""
from collections import Counter
from game_state import GameState, Unit, ShopSlot, OpponentInfo, TRAIT_THRESHOLDS


# ---- economics -------------------------------------------------------

def sell_value(unit: Unit) -> int:
    """Selling a unit always returns (total elixir spent on its merge chain) - 1,
    regardless of star level or ruler. Confirmed by user: a 2-elixir card sells
    for 1; a merged 2-star costing 4 total sells for 3."""
    return max(unit.elixir_spent - 1, 0)


def merge_bonus(ruler: str) -> int:
    """Golden elixir paid out at the moment two units merge. Universal +1,
    but Spirit Empress's own bonus overlaps/stacks for +2 total."""
    return 2 if ruler == "spirit_empress" else 1


def cycle_net_value(cost_per_copy: int, ruler: str) -> int:
    """Net elixir value of buy(2x) -> merge -> sell, for sanity-checking the
    model. Should be 0 for any non-Spirit-Empress ruler (breakeven) and +1
    for Spirit Empress (profit) -- matches the user's worked example."""
    spent = cost_per_copy * 2
    merged_unit = Unit(card_id="x", star_level=2, elixir_spent=spent, traits=())
    return -spent + merge_bonus(ruler) + sell_value(merged_unit)


# ---- buying / merging --------------------------------------------------

def find_merge_opportunities(state: GameState):
    """Return the shop slot indices that would complete a merge.

    Confirmed by the user: 2 copies of the same card at the same star level
    merge into 1 copy at the next star -- 2x 1-star -> 1x 2-star, 2x 2-star
    -> 1x 3-star (4 one-star-equivalents total), 2x 3-star -> 1x 4-star (8
    total, rare -- shop only stocks 4 copies of a given card, so this
    normally only happens via a quest reward, e.g. Spirit Empress's 3rd
    quest at 60 merges, or other game modes; reward mechanics change by
    season so don't hardcode around them). Shop cards are always bought at
    1-star, so completing a merge from the shop just needs ONE matching
    1-star unit already owned, not two."""
    owned = Counter(
        (u.card_id, u.star_level) for u in (state.bench + state.board)
    )
    merge_slots = []
    for i, slot in enumerate(state.shop):
        # Added 2026-09-21: the game itself shows a green "will merge"
        # arrow on a shop card when buying it completes a merge (see
        # recognize.py's read_shop()/_green_arrow_present()) -- far more
        # reliable than the card_id check below, since card_id is mostly
        # unrecognized right now (icon library gap). Trust it directly
        # when present; only fall back to the card_id/owned-copy check
        # when it isn't (either no arrow, or the field wasn't populated --
        # e.g. a synthetic/test ShopSlot without will_merge set).
        if getattr(slot, "will_merge", False):
            merge_slots.append(i)
            continue
        if slot.card_id is None:
            continue
        key = (slot.card_id, 1)  # shop cards are always 1-star when bought
        if owned[key] >= 1:
            merge_slots.append(i)
    return merge_slots


def decide_buys(state: GameState, committed_trait: str = None) -> list:
    """Return an ordered list of shop slot indices to buy this tick.
    Priority: (1) anything that completes a merge, (2) anything matching
    the committed comp trait if bench has room, (3) nothing -- don't buy
    off-trait filler once a trait is committed, since off-trait units are
    pure downside (they cost a bench slot and this ruler's edge is merge
    farming ON the comp, not random accumulation).
    """
    buys = list(find_merge_opportunities(state))
    if committed_trait:
        # Only computed when actually needed -- board_capacity can be
        # (None, None) on a tick where that field failed to read, and this
        # used to compute bench_room unconditionally, crashing on a None -
        # None subtraction even when committed_trait was never set and the
        # result would've gone unused. Found + fixed 2026-09-21 wiring up
        # main.py's real action loop, which calls this every tick against
        # live (sometimes-unreadable) state.
        used, maxc = state.board_capacity
        bench_room = (maxc - used - len(buys)) if (used is not None and maxc is not None) else 0
        if bench_room > 0:
            for i, slot in enumerate(state.shop):
                if i in buys or slot.card_id is None:
                    continue
                if committed_trait in _card_traits(slot.card_id):
                    buys.append(i)
                    bench_room -= 1
                if bench_room <= 0:
                    break
    return buys


def _card_traits(card_id: str) -> tuple:
    """Placeholder lookup until the card->trait table is filled in from the
    reference-icon library (Task #2). Returns () (unknown) until then."""
    return CARD_TRAITS.get(card_id, ())


# Filled in 2026-09-22 from the first labeled batch of
# data/reference_cards/unlabeled/*.png (Task #10 -- see bot-plan.md for the
# extraction/clustering story). Trait names are a BEST-EFFORT read of each
# card's small bottom-center badge icon, cross-checked against the only
# traits actually confirmed so far (data/reference_traits/ has real
# calibration icons for "clan", "forest", "goblin", "horse" -- those three
# names are used here wherever a card's badge visually matched that family;
# "undead", "guardian", and "pyro" are this session's own guesses for badge
# shapes (skull / water-droplet / campfire) that don't have a confirmed
# reference icon yet, and may not be the game's real trait names. Treat
# every value here as unverified until read_own_traits() or a real trait
# tooltip confirms it -- wrong now just means a wrong trait-synergy buy,
# not a crash, since _card_traits() already defaults unknown cards to ().
CARD_TRAITS = {
    "barbarian": ("guardian",),
    "skeleton_king": ("undead",),
    "shadow_assassin": ("forest",),
    "curly_hair_woman": ("guardian",),
    "golden_knight": ("guardian",),
    "blonde_prince": ("clan",),
    "pink_cap_scout": ("forest",),
    "goblin_goggles": ("goblin",),
    "dark_witch": ("undead",),
    "green_orc": ("goblin",),
    "skeleton_pile": ("undead",),
    "swordgirl": ("clan",),
    "fire_queen": ("pyro",),
    "mustache_man": ("clan",),
    "red_hair_archer": ("forest",),
    "baby_dragon": ("forest",),
    "goblin_blue_cap": ("goblin",),
    "goblin_green": ("goblin",),
    "pink_archer": ("guardian",),
    "wood_totem": ("goblin",),
    "lavender_helmet_girl": ("clan",),
}


# ---- profit-taking sell (merge completed -- the whole point of the ruler) ---

def decide_profit_sell(state: GameState) -> Unit:
    """Return a bench unit to sell for merge profit, or None.

    This ruler's entire economic engine is buy(dupe) -> merge -> sell: the
    merge itself pays a gold bonus, and selling then refunds
    (elixir_spent - 1) on top -- together that's a net profit for Spirit
    Empress specifically (see cycle_net_value() above; breakeven for other
    rulers). So the moment a bench unit has actually been merged
    (star_level >= 2), it should be sold right away to bank that profit --
    unlike decide_forced_sell() below, this fires regardless of whether the
    bench is full, and targets merge progress, not comp/trait fit. A merged
    unit isn't being kept for its board strength here; it's inventory to
    flip.

    NOTE (2026-09-21): confirmed via user report that merged units were
    sitting on the bench unsold. Root cause: recognize.py's read_bench()
    currently hardcodes star_level=1 for every bench unit (the star-badge
    sub-region has only been measured against BOARD units mid-battle, which
    show it next to a health bar -- see data/reference_stars/ -- bench units
    during Deploy Phase have no health bar, so that's a different, not-yet-
    calibrated screen position). So the bench half of this function is
    correct but inert in practice until star_level reading is wired up for
    the bench specifically -- see bot-plan.md. Searched ~3550 real bench
    crops for a confirmed 2-star+ example to calibrate against (2026-09-22)
    and found essentially none -- this bot's bench sits empty the huge
    majority of ticks (units seem to get placed straight onto the board),
    so there's still no real data to calibrate star_level against here.

    EXTENDED 2026-09-22 (user request: "we could also sell troops in the
    battle field, so we could merge and sell troops in the battlefield to
    gain the economic benefit") -- now also checks state.board, not just
    state.bench, for the same reason: a merged unit is inventory to flip
    regardless of which side of the wooden tray line it's standing on. This
    additionally depends on read_board() (recognize.py, new this session)
    for board_capacity, whose card_id/star_level reads are themselves
    best-effort/low-confidence -- see read_board()'s docstring. Practically:
    this half is exercised even less than the bench half right now, since
    star_level is ALSO hardcoded 1 for every board unit read_board() finds
    (no confirmed 2-star+ board capture existed to calibrate against
    either -- see bot-plan.md)."""
    candidates = [u for u in (state.bench + state.board) if u.star_level >= 2]
    if not candidates:
        return None
    # If more than one is ready in the same tick, sell the most valuable
    # first (highest sunk cost -> highest refund). Fairly arbitrary since
    # only one action fires per tick anyway -- the rest get caught next tick.
    return max(candidates, key=lambda u: u.elixir_spent)


# ---- forced sell (bench/board full, no merge available) ---------------

def decide_forced_sell(state: GameState, committed_trait: str = None) -> Unit:
    """Pick the single worst-fit unit to sell. Never rank by cost -- every
    unmerged 1-star sale nets exactly -1 regardless of price, so cost carries
    no information here. Rank by (a) does it match the committed trait,
    (b) star level (keep merged progress), (c) elixir sunk (keep the most
    invested unit, as a last-resort tiebreak only)."""
    pool = state.bench + state.board
    if not pool:
        return None

    def fit_score(u: Unit):
        # NOTE (2026-09-22): reads traits fresh via _card_traits(u.card_id)
        # rather than trusting u.traits -- recognize.py's readers
        # (read_bench()/read_board()) always hand back traits=() by
        # design (perception layer stays independent of CARD_TRAITS, see
        # recognize.py's _card_traits_tuple() docstring), so u.traits was
        # silently always empty here before this fix, meaning trait-fit
        # never actually influenced forced-sell -- it degraded to
        # star_level/elixir_spent only. _card_traits() is decide.py's own
        # lookup (same one decide_buys() already uses), so this actually
        # sees the CARD_TRAITS table filled in this session.
        matches_trait = committed_trait in _card_traits(u.card_id) if committed_trait else False
        return (
            matches_trait,          # False (0) sorts before True (1) -> worst fit first
            u.star_level,           # lower star sells before higher star
            u.elixir_spent,         # least invested sells first, all else equal
        )

    return min(pool, key=fit_score)


# ---- round 3+ lobby scouting --------------------------------------------

def pick_committed_trait(own_traits: dict, available_traits: list) -> str:
    """Practical stand-in for decide_lobby_trait() below until real
    opponent scouting exists (2026-09-22). decide_lobby_trait() needs
    per-opponent trait counts, which means navigating into each opponent's
    board mid-round and reading it -- a real UI-navigation feature (see
    REGIONS['opponent_overlay_traits'] / 'back_button' in recognize.py,
    both present but nothing in main.py drives them yet) that's out of
    scope for this pass. In the meantime, commit to whichever trait the
    player's OWN shop luck has already leaned into hardest -- read
    straight off the game's own trait-progress badges via
    read_own_traits() -- as a greedy substitute for "avoid what opponents
    are doing": reinforcing an accidental early lead is a reasonable
    default when there's no visibility into contention at all. Swap the
    call site in main.py over to decide_lobby_trait() once opponent
    scouting lands; this function can stay as the no-scouting fallback."""
    if not available_traits:
        return None
    if not own_traits:
        return available_traits[0]

    def progress(trait):
        raw = own_traits.get(trait)
        if not raw or "/" not in str(raw):
            return 0
        have, _, _need = str(raw).partition("/")
        try:
            return int(have)
        except ValueError:
            return 0

    return max(available_traits, key=progress)


def decide_lobby_trait(opponents: dict, available_traits: list) -> str:
    """At round 3 (and optionally re-checked later), commit to whichever
    available trait is least used across all scouted opponents -- improves
    odds of finding 3-star merges later since fewer players compete for the
    same shop cards."""
    usage = Counter()
    for opp in opponents.values():
        for trait, count in opp.traits.items():
            if count > 0:
                usage[trait] += 1  # count *players* using it, not their stack size
    # traits nobody has scouted yet default to 0 usage -- most attractive
    return min(available_traits, key=lambda t: usage.get(t, 0))


# ---- smoke test ----------------------------------------------------------

if __name__ == "__main__":
    # 1. economics sanity check
    assert cycle_net_value(2, ruler="knight_king") == 0, "non-Spirit-Empress should breakeven"
    assert cycle_net_value(2, ruler="spirit_empress") == 1, "Spirit Empress should net +1"
    print("[ok] economics: breakeven for other rulers, +1 for Spirit Empress")

    # 2. forced-sell should ignore cost and pick the off-trait unit
    bench = [
        Unit("cheap_offtrait", 1, elixir_spent=2, traits=("clan",)),
        Unit("expensive_ontrait", 1, elixir_spent=5, traits=("forest",)),
    ]
    state = GameState(
        round=5, phase="deploy", time_left=20, gold=3,
        board_capacity=(5, 5), shop=[ShopSlot(None, 0)] * 3,
        bench=bench, board=[], own_traits={}, opponents={},
    )
    sold = decide_forced_sell(state, committed_trait="forest")
    assert sold.card_id == "cheap_offtrait", f"expected to sell off-trait unit, got {sold.card_id}"
    print("[ok] forced sell picks worst trait-fit, not cheapest")

    # 3. lobby scouting picks least-contested trait
    opponents = {
        "p1": OpponentInfo(name="p1", health=7, traits={"clan": 2, "noble": 1}),
        "p2": OpponentInfo(name="p2", health=5, traits={"clan": 1, "undead": 3}),
    }
    pick = decide_lobby_trait(opponents, ["clan", "noble", "undead", "forest"])
    assert pick == "forest", f"expected 'forest' (0 opponents), got {pick}"
    print("[ok] lobby scouting picks the untouched trait")

    # 4. merge detection should fire with just ONE matching 1-star unit
    #    already owned (2 total = merge), not two already owned (3 total)
    bench2 = [Unit("knight", 1, elixir_spent=2, traits=("noble",))]
    state2 = GameState(
        round=3, phase="deploy", time_left=20, gold=5,
        board_capacity=(5, 5),
        shop=[ShopSlot("knight", 2), ShopSlot(None, 0), ShopSlot(None, 0)],
        bench=bench2, board=[], own_traits={}, opponents={},
    )
    merges = find_merge_opportunities(state2)
    assert merges == [0], f"expected shop slot 0 to complete a merge, got {merges}"
    print("[ok] merge detected from 1 owned + 1 in shop (2 total), not 3")

    # 5. profit-sell should target a merged (star>=2) bench unit, ignoring
    #    an unmerged one even when it's more expensive
    bench3 = [
        Unit("cheap_merged", 2, elixir_spent=4, traits=()),
        Unit("expensive_unmerged", 1, elixir_spent=5, traits=()),
    ]
    state3 = GameState(
        round=5, phase="deploy", time_left=20, gold=3,
        board_capacity=(2, 5), shop=[ShopSlot(None, 0)] * 3,
        bench=bench3, board=[], own_traits={}, opponents={},
    )
    sold3 = decide_profit_sell(state3)
    assert sold3.card_id == "cheap_merged", f"expected the merged unit, got {sold3.card_id}"
    assert decide_profit_sell(GameState(
        round=5, phase="deploy", time_left=20, gold=3, board_capacity=(1, 5),
        shop=[ShopSlot(None, 0)] * 3, bench=[Unit("x", 1, 2, ())], board=[],
        own_traits={}, opponents={},
    )) is None, "no star>=2 units on bench -- should return None"
    print("[ok] profit-sell targets a merged unit, ignores unmerged ones")

    print("All smoke tests passed.")
