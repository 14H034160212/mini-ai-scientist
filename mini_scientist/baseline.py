"""A non-LLM reference: random experiments + brute-force search over a small
grammar of `F if C else G`. It answers "how hard is this lab, really?"
and is what an LLM scientist should be compared against."""
from __future__ import annotations

import itertools
import random

from .lab import Lab, compile_expr

TERMS = ["a", "b", "a + b", "a - b", "b - a", "a * b", "2*a", "2*b", "a + 1", "a + 3",
         "2*a + 3", "a - b + 5", "abs(a - b)", "max(a, b)", "min(a, b)", "0"]
CONDS = ["a > b", "a >= b", "a == b", "a % 2 == 0", "b % 2 == 0", "a % 2 == b % 2",
         "(a + b) % 5 == 0", "b % 3 == 0", "a % 3 == 0"] + \
        [f"a > {k}" for k in range(1, 20)] + [f"b < {k}" for k in range(1, 21)] + \
        ["5 <= a <= 15"]


def _candidates():
    for t in TERMS:
        yield t
    for c, f, g in itertools.product(CONDS, TERMS, TERMS):
        if f != g:
            yield f"{f} if {c} else {g}"


_COMPILED = None


def enumerative_scientist(lab: Lab, batch: int = 6, seed: int = 0) -> dict:
    """Run random experiments in batches until exactly one-or-more candidates survive
    and the survivor set stops shrinking; submit the first (simplest) survivor."""
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = [(e, compile_expr(e)) for e in _candidates()]
    rng = random.Random(seed)
    alive = _COMPILED
    while True:
        obs = {(a, b): y for a, b, y in lab.observations}
        alive = [(e, g) for e, g in alive if all(g(a, b) == y for (a, b), y in obs.items())]
        if not alive or lab.used >= lab.budget:
            break
        # stop when every survivor agrees everywhere (no experiment can split them)
        if len({tuple(g(a, b) for a in range(21) for b in range(21)) for _, g in alive[:50]}) == 1:
            break
        lab.run([(rng.randint(0, 20), rng.randint(0, 20)) for _ in range(batch)])
    sub = alive[0][0] if alive else None
    score = lab.verify(sub) if sub else {"correct": False, "accuracy": 0.0}
    return {"rule": lab.rule.name, "tier": lab.rule.tier, "seed": lab.seed, "submitted": sub,
            **score, "experiments": lab.used, "survivors": len(alive)}
