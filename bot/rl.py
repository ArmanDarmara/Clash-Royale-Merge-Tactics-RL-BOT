"""
Reward/punishment learning layer (2026-09-21).

WHY THIS SHAPE: the obvious "RL" version of this ask -- a deep policy
trained by self-play -- isn't realistic here. There's one live account, no
simulator, matches take 10+ minutes each, and real play risks a ban, so we
can't generate the thousands of episodes deep RL needs. What IS realistic:
a small linear policy over hand-built features, trained via REINFORCE
(Monte-Carlo policy gradient) from the *sparse, once-per-match* reward
decide.py's placement gives us. That's real reinforcement learning, just
scoped to match how little and how slow the data actually is.

Formalization:
  - Episode = one match. Decision point = one main.py tick.
  - Action = one candidate from enumerate_candidates() in main.py: a
    specific buy, a specific sell, or no-op.
  - Policy: softmax over a linear score w . phi(state, action) across
    whatever candidates are available that tick (see featurize()).
  - Reward: terminal only, from the match's final placement (see
    PLACEMENT_REWARDS below) -- credited equally to every action taken
    that match (no shaping / discounting yet -- simplest correct version).
  - Update: standard softmax policy-gradient (REINFORCE) --
    w += lr * G * (phi(chosen) - E_pi[phi]) for every tick in the episode.

Rollout plan (see bot-plan.md for the up to date status):
  Phase 1 (now): decide.py's rule-based logic keeps making the real
  taps/sells (unchanged, still the safe default). main.py additionally
  featurizes every candidate each tick and logs which one the rule-based
  logic actually executed to <run_dir>/trajectory.jsonl. Once
  read_placement() is calibrated (needs a real end-of-match screen capture
  -- see recognize.py), bot/rl_train.py replays every saved run's
  trajectory + detected placement through reinforce_update() below. This
  warm-starts the linear weights from real rule-based play before the
  policy ever acts on its own -- it's off-policy relative to what the
  softmax policy would have chosen, but it's a reasonable and standard way
  to bootstrap a policy from an existing controller's rollouts.
  Phase 2 (later, once weights look sane and a few dozen matches have been
  trained on): set CRMT_RL_ACT=1 so main.py lets this policy actually
  choose the action each tick (real on-policy REINFORCE from then on).
"""
import json
import math
import os
import random

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rl_weights.json")

# User's ask, translated into numbers: punishment for 3rd/4th, reward for
# 2nd, bigger reward for 1st. These are a starting guess, not measured --
# easy to retune later (data/rl_weights.json stores whichever reward was
# actually used for each historical episode, so retuning doesn't silently
# invalidate past training data's meaning).
PLACEMENT_REWARDS = {1: 3.0, 2: 1.0, 3: -1.0, 4: -2.0}

# Fixed feature order -- every featurize() call returns exactly these keys
# (0.0 when a feature doesn't apply to that action kind), so scoring/
# gradient code never has to reconcile mismatched key sets across
# candidates.
FEATURE_KEYS = [
    "bias",
    "is_buy", "is_sell", "is_noop",
    "cost",
    "gold_before", "gold_after",
    "is_merge_buy",
    "bench_used_before", "bench_room_after",
    "round_num",
    "star_level",          # sell candidates only
    "trait_committed",     # 1.0 if committed_trait is set at all (buy/sell match, not just set)
]


def featurize(state, kind, detail=None) -> dict:
    """kind: 'buy' | 'sell' | 'noop'. detail:
      buy:  {'slot_index': i, 'cost': c, 'is_merge': bool}
      sell: {'bench_index': i, 'unit': Unit}
      noop: None
    Returns a dict with every FEATURE_KEYS entry present (missing -> 0.0)."""
    detail = detail or {}
    f = {k: 0.0 for k in FEATURE_KEYS}
    f["bias"] = 1.0
    f["round_num"] = float(state.round or 0)
    f["gold_before"] = float(state.gold or 0)
    f["bench_used_before"] = float(len(state.bench))

    if kind == "buy":
        cost = detail.get("cost") or 0
        f["is_buy"] = 1.0
        f["cost"] = float(cost)
        f["gold_after"] = float((state.gold or 0) - cost)
        f["is_merge_buy"] = 1.0 if detail.get("is_merge") else 0.0
        f["bench_room_after"] = float(5 - len(state.bench) - 1)
    elif kind == "sell":
        unit = detail.get("unit")
        f["is_sell"] = 1.0
        f["gold_after"] = float(state.gold or 0)
        f["star_level"] = float(getattr(unit, "star_level", 0) or 0)
        f["bench_room_after"] = float(5 - len(state.bench) + 1)
    else:  # noop
        f["is_noop"] = 1.0
        f["gold_after"] = float(state.gold or 0)
        f["bench_room_after"] = float(5 - len(state.bench))

    return f


class LinearPolicy:
    """Softmax policy over a linear score, trained by REINFORCE. Weights
    persist to WEIGHTS_PATH between runs/processes."""

    def __init__(self, weights=None, episodes_trained=0, history=None):
        self.weights = weights or {k: 0.0 for k in FEATURE_KEYS}
        self.episodes_trained = episodes_trained
        self.history = history if history is not None else []

    @classmethod
    def load(cls):
        if not os.path.exists(WEIGHTS_PATH):
            return cls()
        try:
            with open(WEIGHTS_PATH) as fp:
                data = json.load(fp)
            weights = {k: float(data.get("weights", {}).get(k, 0.0)) for k in FEATURE_KEYS}
            return cls(weights, data.get("episodes_trained", 0), data.get("history", []))
        except Exception:
            # A corrupt/partial weights file shouldn't crash the bot --
            # fall back to a fresh (all-zero) policy.
            return cls()

    def save(self):
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        with open(WEIGHTS_PATH, "w") as fp:
            json.dump({
                "weights": self.weights,
                "episodes_trained": self.episodes_trained,
                "history": self.history,
            }, fp, indent=2)

    def score(self, feat: dict) -> float:
        return sum(self.weights[k] * feat.get(k, 0.0) for k in FEATURE_KEYS)

    def softmax_probs(self, candidate_feats: list, temperature: float = 1.0) -> list:
        scores = [self.score(f) / temperature for f in candidate_feats]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]  # subtract max for numerical stability
        total = sum(exps)
        return [e / total for e in exps]

    def select(self, candidate_feats: list, temperature: float = 1.0, rng=None) -> int:
        """Sample an action index from the softmax distribution (Phase 2 --
        only used when CRMT_RL_ACT=1)."""
        rng = rng or random
        probs = self.softmax_probs(candidate_feats, temperature)
        r = rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        return len(probs) - 1  # floating-point fallback

    def reinforce_update(self, trajectory: list, reward: float, lr: float = 0.05):
        """trajectory: list of {'candidates': [feat, ...], 'chosen_index': i}
        (one entry per tick that made a real decision). Standard softmax
        policy-gradient step, same reward (the match's terminal placement
        reward) credited to every tick -- see module docstring."""
        for step in trajectory:
            cands = step["candidates"]
            chosen = step["chosen_index"]
            if not cands or chosen is None or chosen >= len(cands):
                continue
            probs = self.softmax_probs(cands)
            for k in FEATURE_KEYS:
                expected = sum(p * f.get(k, 0.0) for p, f in zip(probs, cands))
                grad = cands[chosen].get(k, 0.0) - expected
                self.weights[k] += lr * reward * grad
        self.episodes_trained += 1


if __name__ == "__main__":
    # Smoke test: a policy should learn to prefer whichever of two
    # candidates was consistently chosen right before a good outcome, and
    # avoid whichever preceded a bad outcome.
    class _FakeState:
        round = 3
        gold = 5
        bench = []

    good_feats = featurize(_FakeState(), "buy", {"cost": 2, "is_merge": True})
    bad_feats = featurize(_FakeState(), "noop")

    policy = LinearPolicy()
    good_traj = [{"candidates": [good_feats, bad_feats], "chosen_index": 0}]
    bad_traj = [{"candidates": [good_feats, bad_feats], "chosen_index": 1}]

    for _ in range(200):
        policy.reinforce_update(good_traj, reward=PLACEMENT_REWARDS[1], lr=0.05)
    for _ in range(200):
        policy.reinforce_update(bad_traj, reward=PLACEMENT_REWARDS[4], lr=0.05)

    p_good_first = policy.softmax_probs([good_feats, bad_feats])[0]
    assert p_good_first > 0.9, f"expected policy to strongly prefer the merge-buy, got {p_good_first}"
    print(f"[ok] policy learned to prefer the winning-history action (p={p_good_first:.3f})")
    print("All smoke tests passed.")
