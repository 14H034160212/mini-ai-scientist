"""Sanity checks that need no GPU: every ground-truth rule verifies against
itself, the sandbox rejects unsafe code, and a scripted agent works end to end."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini_scientist import Lab, RULES, Episode, run_batch, compile_expr  # noqa: E402
from mini_scientist.lab import ExprError  # noqa: E402

for r in RULES:
    assert Lab(r).verify(r.expr)["correct"], r.name

for bad in ["__import__('os')", "open('x')", "a.__class__", "[a for a in b]", "a ** 99"]:
    try:
        compile_expr(bad)
        raise AssertionError(f"accepted: {bad}")
    except ExprError:
        pass

lab = Lab(RULES[4])  # absdiff
assert not lab.verify("a - b")["correct"]
assert lab.verify("abs(a - b)")["correct"]       # equivalent form counts as correct
assert lab.verify("max(a, b) - min(a, b)")["correct"]


class Scripted:
    """Plays a fixed policy: run experiments, check a wrong rule, then the right one, submit."""
    step = 0

    def generate(self, convs):
        out, n = [], self.step
        self.step += 1
        for _ in convs:
            out.append([
                '{"hypotheses": ["a - b"], "tool": "experiment", "inputs": [[5, 2], [2, 5], [7, 7]]}',
                '{"analysis": "try \\(a-b\\)", "tool": "compute", "columns": ["out - (a - b)", "out - abs(a - b)"]}',
                'I think: {"hypotheses": ["a - b"], "tool": "check", "expr": "a - b"}',
                '```json\n{"tool": "check", "expr": "abs(a - b)"}\n```',
                '{"tool": "submit", "expr": "abs(a - b)"}',
            ][n])
        return out


res = run_batch(Scripted(), [Episode(Lab(RULES[4], seed=s)) for s in range(3)])
assert all(r["correct"] and r["experiments"] == 3 and r["checks"] == 2 and r["computes"] == 1
           and r["parse_errors"] == 0 and r["turns"] == 5 for r in res), res
e = Episode(Lab(RULES[4]))
e.step('{"tool": "compute", "columns": ["out - abs(a - b)"]}')
assert "CONSTANT (0)" in e.trace[-1]["feedback"], e.trace[-1]["feedback"]
g = Episode(Lab(RULES[4]), gate=True)                # absdiff; wrong rule must be refused
g.step('{"tool": "experiment", "inputs": [[2, 5]]}')
g.step('{"tool": "submit", "expr": "a - b"}')
assert not g.done and g.n_rejected == 1 and "REJECTED" in g.trace[-1]["feedback"]
g.step('{"tool": "submit", "expr": "abs(a - b)"}')
assert g.done and g.result()["correct"]
ng = Episode(Lab(RULES[4]))                          # default: no gate, wrong rule accepted
ng.step('{"tool": "submit", "expr": "a - b"}')
assert ng.done and not ng.result()["correct"]
from mini_scientist.agent import TRUNCATED  # noqa: E402
t = Episode(Lab(RULES[0]))
t.step(TRUNCATED)
assert not t.done and t.n_truncated == 1 and t.n_parse_errors == 0 and "cut off" in t.trace[-1]["feedback"]
d = Episode(Lab(RULES[0]))
d.step('{"tool": "experiment", "inputs": [{"a": 3, "b": 4}, {"x": 1}, "junk", [5]]}')
assert d.lab.used == 1 and d.trace[-1]["ran"] == [(3, 4, 7)]
print(json.dumps(res[0], indent=1))
print("all lab tests passed")
