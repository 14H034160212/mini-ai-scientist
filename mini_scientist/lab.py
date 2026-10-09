"""A tiny, fully verifiable "laboratory" with a hidden rule.

The machine takes two integers a, b in [0, 20] and returns an integer.
The agent can only learn the rule by running experiments. Because the
domain is finite (21 x 21 = 441 inputs), a submitted rule can be checked
for exact equivalence against the hidden one -- no LLM judge needed.
"""
from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field

LO, HI = 0, 20
DOMAIN = [(a, b) for a in range(LO, HI + 1) for b in range(LO, HI + 1)]


@dataclass(frozen=True)
class Rule:
    name: str
    tier: int          # 1 = single formula, 2 = one branch, 3 = nested / modular branch
    expr: str          # ground truth, in the same expression language the agent uses


RULES = [
    # Tier 1: one formula everywhere
    Rule("sum", 1, "a + b"),
    Rule("affine", 1, "2*a + 3"),
    Rule("product", 1, "a * b"),
    Rule("diff_plus", 1, "a - b + 5"),
    # Tier 2: a single condition splits the input space
    Rule("absdiff", 2, "a - b if a > b else b - a"),
    Rule("parity_a", 2, "a + b if a % 2 == 0 else a * b"),
    Rule("threshold_b", 2, "a * 2 if b < 10 else a + b"),
    Rule("max_plus", 2, "a + 1 if a >= b else b * 2"),
    # Tier 3: two conditions, or a modular one that is easy to miss
    Rule("nested", 3, "a - b if a > 10 else (a * 2 if b % 3 == 0 else a + b)"),
    Rule("mod5", 3, "a + b if (a + b) % 5 == 0 else a"),
    Rule("band", 3, "b if 5 <= a <= 15 else a * b"),
    Rule("parity_both", 3, "a * b if a % 2 == b % 2 else a + b"),
]
RULES_BY_NAME = {r.name: r for r in RULES}


# ---------------------------------------------------------------- safe eval
_ALLOWED = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Name, ast.Load, ast.Constant, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)
_FUNCS = {"abs": abs, "min": min, "max": max}


class ExprError(ValueError):
    pass


def compile_expr(src: str, names: tuple = ("a", "b")):
    """Compile an agent-written expression over `names`. Only arithmetic,
    comparisons, `x if c else y`, and abs/min/max are allowed."""
    if not isinstance(src, str):
        raise ExprError(f"expected an expression string, got {type(src).__name__}")
    src = src.strip().strip("`")
    if src.startswith("lambda"):
        src = src.split(":", 1)[1]
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"syntax error: {e.msg}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ExprError(f"not allowed: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in (*names, *_FUNCS):
            raise ExprError(f"unknown name '{node.id}' (only {', '.join(names)}, abs, min, max)")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise ExprError("only abs/min/max may be called")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) and 0 <= node.right.value <= 4):
                raise ExprError("exponent must be a constant 0..4")
    code = compile(tree, "<expr>", "eval")

    def f(*args):
        try:
            v = eval(code, {"__builtins__": {}}, {**dict(zip(names, args)), **_FUNCS})
        except ZeroDivisionError:
            return None
        return int(v) if isinstance(v, (int, float, bool)) else None
    return f


# ---------------------------------------------------------------- the lab
@dataclass
class Lab:
    rule: Rule
    budget: int = 30                      # max number of experiments (input pairs)
    seed: int = 0
    observations: list = field(default_factory=list)

    def __post_init__(self):
        self._f = compile_expr(self.rule.expr)
        rng = random.Random(self.seed)
        for a, b in rng.sample(DOMAIN, 3):      # three free starting observations
            self.observations.append((a, b, self._f(a, b)))

    @property
    def used(self) -> int:
        return len(self.observations) - 3

    def run(self, inputs) -> list:
        """Run experiments. Returns [(a, b, out)] for valid in-range inputs within budget."""
        out = []
        for pair in inputs:
            if self.used >= self.budget:
                break
            try:
                if isinstance(pair, dict):                   # small models sometimes write {"a": 3, "b": 4}
                    pair = (pair["a"], pair["b"])
                a, b = int(pair[0]), int(pair[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            if not (LO <= a <= HI and LO <= b <= HI):
                continue
            row = (a, b, self._f(a, b))
            self.observations.append(row)
            out.append(row)
        return out

    def check(self, expr: str) -> list:
        """The agent's 'Python tool': test a hypothesis against data it has
        already observed (never against hidden data). Returns mismatches."""
        g = compile_expr(expr)
        seen, bad = set(), []
        for a, b, y in self.observations:
            if (a, b) in seen:
                continue
            seen.add((a, b))
            if g(a, b) != y:
                bad.append((a, b, y, g(a, b)))
        return bad

    def verify(self, expr: str) -> dict:
        """Ground-truth scoring on all 441 inputs (the agent never sees this)."""
        try:
            g = compile_expr(expr)
        except ExprError as e:
            return {"correct": False, "accuracy": 0.0, "error": str(e)}
        hits = sum(g(a, b) == self._f(a, b) for a, b in DOMAIN)
        return {"correct": hits == len(DOMAIN), "accuracy": hits / len(DOMAIN)}
