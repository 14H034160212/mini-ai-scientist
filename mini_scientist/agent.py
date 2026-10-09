"""The mini AI Scientist loop: hypothesise -> experiment -> analyse -> check -> revise -> submit.

Each turn the model emits one JSON action. Four tools:
  experiment  run the hidden machine on chosen inputs (costs budget)
  compute     let Python compute derived columns (e.g. out - a - b) over all observations
  check       test a candidate rule (a Python expression) against the data seen so far
  submit      commit to a final rule; scored on all 441 inputs

Division of labour: the LLM decides what to try and what to believe; Python does
every piece of arithmetic. A "lab notebook" of tested hypotheses replaces long chat
history, so the prompt stays short enough for small models.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .lab import Lab, ExprError, LO, HI, compile_expr

TRUNCATED = "[thinking truncated]"   # what a backend returns when a reasoning model never finished thinking

INTRO = f"""You are a scientist studying a black-box machine.
The machine takes two integers a and b (each from {LO} to {HI}) and returns an integer `out`.
It follows a hidden deterministic rule. The rule may be a single formula, or it may
behave differently in different regions of the input space (if/else).
Your goal: discover the exact rule by designing experiments.

Every turn, reply with ONE JSON object and nothing else. Start it with an "analysis"
field (2-4 sentences). Then choose one tool:"""

TOOLS = {
    "experiment": """Run experiments (at most 6 input pairs per turn, costs budget):
{"analysis": "...", "tool": "experiment", "inputs": [[a, b], [a, b]]}""",
    "compute": """Ask Python to compute derived columns over ALL observations (free). Use this
instead of doing arithmetic in your head. Columns may use a, b and out:
{"analysis": "...", "tool": "compute", "columns": ["out - a - b", "out - a * b", "out % 2"]}""",
    "check": """Check a candidate rule against ALL observations (free):
{"analysis": "...", "tool": "check", "expr": "<python expression in a and b>"}""",
    "submit": """Submit your final rule (ends the study):
{"analysis": "...", "tool": "submit", "expr": "<python expression in a and b>"}""",
}

RULES_OF_SYNTAX = """Expressions use a and b, the operators + - * // % ** and comparisons, and abs/min/max.
A conditional must be written `X if CONDITION else Y` (always with else).
Examples: "3*a - b", "a + b if a > 7 else 2*b"

Good scientific practice:
- Do not guess from 3 points. Run controlled experiments: fix b and vary a, then fix a and
  vary b. Cover a > b, a < b, a == b, even and odd values, small and large values.
- Use compute to test simple formulas on all data at once: a column that is constant
  (e.g. out - a - b is always 0) reveals the formula. If a column is constant only on
  some rows, find what those rows have in common: that is the condition.
- Start from the SIMPLEST rule that fits. Only add a condition when the data forces it.
- Before submitting, check the rule and run an experiment that could refute it."""


def system_prompt(tools: list[str]) -> str:
    body = "\n\n".join(f"{i}. {TOOLS[t]}" for i, t in enumerate(tools, 1))
    return f"{INTRO}\n\n{body}\n\n{RULES_OF_SYNTAX}"


_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def parse_action(text: str):
    """Return the first JSON object in `text` that has a 'tool' key, else None.
    Tolerates invalid escapes such as LaTeX `\\(` that small models like to emit."""
    dec = json.JSONDecoder()
    for src in (text, _BAD_ESCAPE.sub(r"\\\\", text)):
        i = src.find("{")
        while i != -1:
            try:
                obj, _ = dec.raw_decode(src[i:])
                if isinstance(obj, dict) and "tool" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
            i = src.find("{", i + 1)
    return None


@dataclass
class Episode:
    lab: Lab
    max_turns: int = 15
    tools: tuple = ("experiment", "compute", "check", "submit")
    history_turns: int = 3                # chat turns kept verbatim; older ones live in the notebook
    gate: bool = False                    # refuse a submission that contradicts the observed data
    trace: list = field(default_factory=list)
    notebook: list = field(default_factory=list)   # [(expr, n_wrong, n_obs)]
    turns: int = 0
    n_checks: int = 0
    n_computes: int = 0
    n_parse_errors: int = 0
    n_rejected: int = 0
    n_truncated: int = 0
    submitted: str | None = None
    last_clean: str | None = None         # last checked rule with zero mismatches
    done: bool = False
    _chat: list = field(default_factory=list)
    _seen: set = field(default_factory=set)

    def __post_init__(self):
        self.system = system_prompt(list(self.tools))
        self._chat = [{"role": "user", "content": self._status("Initial observations:")}]

    # ------------------------------------------------------------ prompt
    @property
    def messages(self) -> list:
        keep = self._chat[-(2 * self.history_turns + 1):]
        return [{"role": "system", "content": self.system}] + keep

    def _rows(self):
        return sorted({(a, b): y for a, b, y in self.lab.observations}.items())

    def _status(self, head: str) -> str:
        table = "\n".join(f"  f({a}, {b}) = {y}" for (a, b), y in self._rows())
        msg = f"{head}\n\nAll observations (a, b -> out):\n{table}"
        if self.notebook:
            msg += "\n\nLab notebook (rules already checked):\n" + "\n".join(
                f"  {e}  ->  " + ("fits all" if w == 0 else f"WRONG on {w}/{n}") for e, w, n in self.notebook[-8:])
        left, turns_left = self.lab.budget - self.lab.used, self.max_turns - self.turns
        msg += f"\n\nExperiment budget left: {left}. Turns left: {turns_left}."
        if turns_left <= 1:
            msg += " This is your last turn: you must submit now."
        elif left <= 0:
            msg += " No experiments left: check your best rule and submit."
        return msg + "\nReply with one JSON object (analysis first)."

    def _reply(self, rec: dict, fb: str):
        rec["feedback"] = fb
        self._chat.append({"role": "user", "content": self._status(fb)})

    # ------------------------------------------------------------ tools
    def _compute(self, cols) -> str:
        if not isinstance(cols, list) or not cols:
            return 'compute needs "columns": a list of expressions in a, b, out.'
        cols, out, fns = [str(c) for c in cols[:4]], [], []
        for c in cols:
            try:
                fns.append(compile_expr(c, names=("a", "b", "out")))
            except ExprError as e:
                return f"Column `{c}` is invalid: {e}."
        lines = []
        for (a, b), y in self._rows():
            lines.append(f"  a={a:>2} b={b:>2} out={y:>3} | " + " | ".join(f"{c} = {g(a, b, y)}" for c, g in zip(cols, fns)))
        for c, g in zip(cols, fns):
            vals = {g(a, b, y) for (a, b), y in self._rows()}
            out.append(f"`{c}` is CONSTANT ({vals.pop()}) on all rows" if len(vals) == 1
                       else f"`{c}` takes {len(vals)} different values")
        return "Computed columns:\n" + "\n".join(lines) + "\nSummary: " + "; ".join(out) + "."

    def _check(self, expr: str) -> str:
        try:
            g = compile_expr(expr)
        except ExprError as e:
            return (f"Check failed: {e}. The expression must be valid Python in a and b; "
                    "every conditional needs both parts: `X if CONDITION else Y`.")
        rows = self._rows()
        wrong = [(a, b, y, g(a, b)) for (a, b), y in rows if g(a, b) != y]
        self.notebook.append((expr, len(wrong), len(rows)))
        if not wrong:
            self.last_clean = expr
            fb = f"Check `{expr}`: fits all {len(rows)} observations."
            if expr.count(" if ") > 2:
                fb += " But it has many branches and may be overfitted; look for a simpler rule."
            return fb + " Run an experiment that could refute it, or submit."
        ok = [(a, b) for (a, b), y in rows if g(a, b) == y]
        return (f"Check `{expr}`: WRONG on {len(wrong)} of {len(rows)} observations.\n"
                f"  RIGHT on: {', '.join(f'({a},{b})' for a, b in ok) or 'none'}\n"
                f"  WRONG on: " + ", ".join(f"({a},{b}) out={y} rule={r}" for a, b, y, r in wrong[:10]) +
                "\nWhat do the WRONG rows have in common? Revise.")

    def _contradicts(self, expr) -> bool:
        try:
            g = compile_expr(str(expr or ""))
        except ExprError:
            return True
        return any(g(a, b) != y for (a, b), y in self._rows())

    # ------------------------------------------------------------ loop
    def step(self, reply: str):
        try:
            self._step(reply)
        except Exception as e:                       # a malformed action must never kill the whole batch
            self.n_parse_errors += 1
            if not self.trace or self.trace[-1]["turn"] != self.turns:
                self.trace.append({"turn": self.turns, "raw": reply, "action": None})
            self.trace[-1]["error"] = repr(e)
            self._reply(self.trace[-1], f"Could not apply that action ({type(e).__name__}). "
                                        "Reply with ONE valid JSON object.")
            if not self.done and self.turns >= self.max_turns:
                self.submitted = self.last_clean
                self.done = True

    def _step(self, reply: str):
        self.turns += 1
        self._chat.append({"role": "assistant", "content": reply[:2000]})
        act = parse_action(reply)
        rec = {"turn": self.turns, "raw": reply, "action": act}
        tool = (act or {}).get("tool")
        if tool not in self.tools:
            tool = None
        key = json.dumps([tool, act and act.get("expr"), act and act.get("columns"), len(self.lab.observations)])

        if reply == TRUNCATED:
            self.n_truncated += 1
            self._reply(rec, "Your reasoning was too long and got cut off before you answered. "
                             "Think more briefly and reply with one JSON object.")
        elif tool == "submit" and self.gate and self.turns < self.max_turns and self._contradicts(act.get("expr")):
            self.n_rejected += 1
            self._reply(rec, "Submission REJECTED: a conclusion must agree with your own data.\n"
                        + self._check(str(act.get("expr", ""))))
        elif tool == "submit":
            self.submitted = str(act.get("expr") or "") or self.last_clean
            self.done = True
            rec["feedback"] = "submitted"
        elif tool == "experiment":
            inputs = act.get("inputs")
            rows = self.lab.run(inputs[:6]) if isinstance(inputs, list) else []
            rec["ran"] = rows
            self._reply(rec, "New results: " + ", ".join(f"f({a}, {b}) = {y}" for a, b, y in rows) if rows else
                        f"No experiment ran: inputs must be [[a, b], ...] with values {LO}..{HI} and budget left.")
        elif tool in ("check", "compute") and key in self._seen:
            self._reply(rec, f"You already ran this exact {tool} on this exact data; the result will not change. "
                             "Run an experiment or try something different.")
        elif tool == "compute":
            self.n_computes += 1
            self._seen.add(key)
            self._reply(rec, self._compute(act.get("columns")))
        elif tool == "check":
            self.n_checks += 1
            self._seen.add(key)
            self._reply(rec, self._check(str(act.get("expr", ""))))
        else:
            self.n_parse_errors += 1
            self._reply(rec, "Could not read a valid action. Reply with ONE JSON object whose \"tool\" is one of: "
                             + ", ".join(self.tools) + ". Use plain text in strings (no backslashes).")

        if not self.done and self.turns >= self.max_turns:
            self.submitted = self.last_clean
            self.done = True
        self.trace.append(rec)

    def result(self) -> dict:
        score = self.lab.verify(self.submitted) if self.submitted else \
            {"correct": False, "accuracy": 0.0, "error": "no submission"}
        return {
            "rule": self.lab.rule.name, "tier": self.lab.rule.tier, "seed": self.lab.seed,
            "truth": self.lab.rule.expr, "submitted": self.submitted, **score,
            "experiments": self.lab.used, "turns": self.turns, "checks": self.n_checks,
            "computes": self.n_computes, "parse_errors": self.n_parse_errors, "rejected": self.n_rejected, "truncated": self.n_truncated,
        }


def run_batch(backend, episodes: list[Episode]) -> list[dict]:
    """Step all episodes in lockstep, one batched generate() per turn."""
    while True:
        live = [e for e in episodes if not e.done]
        if not live:
            break
        replies = backend.generate([e.messages for e in live])
        for e, r in zip(live, replies):
            e.step(r)
    return [e.result() for e in episodes]
