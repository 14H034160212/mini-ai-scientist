"""Aggregate a sweep directory into a markdown table and the headline figure.

  python scripts/analyze.py results/sweep
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "Droid Sans Fallback", "DejaVu Sans"]

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
OFF, ON = "#eb6834", "#2a78d6"   # categorical slots 2 and 1: orange = thinking off, blue = thinking on


def load(d: Path):
    runs = {}
    for f in sorted(d.glob("*.jsonl")):
        runs[f.stem] = [json.loads(l) for l in open(f)]
    return runs


def own_data_conflict(r) -> bool:
    """Did the agent submit a rule that contradicts experiments it ran itself?"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mini_scientist.lab import Lab, RULES_BY_NAME, ExprError
    lab = Lab(RULES_BY_NAME[r["rule"]], seed=r["seed"])
    for t in r["trace"]:
        lab.observations.extend(tuple(x) for x in (t.get("ran") or []))
    try:
        return bool(lab.check(r["submitted"] or ""))
    except ExprError:
        return True


def summarize(rs):
    s = {"n": len(rs), "solved": sum(r["correct"] for r in rs),
         "acc": sum(r["accuracy"] for r in rs) / len(rs),
         "exp_solved": (sum(r["experiments"] for r in rs if r["correct"]) /
                        max(1, sum(r["correct"] for r in rs))),
         "parse": sum(r["parse_errors"] for r in rs) / len(rs),
         "conflict": sum(own_data_conflict(r) for r in rs if not r["correct"]),
         "fail": sum(not r["correct"] for r in rs)}
    for t in (1, 2, 3):
        x = [r for r in rs if r["tier"] == t]
        s[f"t{t}"] = f"{sum(r['correct'] for r in x)}/{len(x)}"
    return s


def table(runs) -> str:
    lines = ["| 配置 | 解出 | Tier 1 | Tier 2 | Tier 3 | 平均一致率 | 成功时平均实验次数 | 失败中与自身数据矛盾 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, rs in runs.items():
        s = summarize(rs)
        lines.append(f"| {name} | {s['solved']}/{s['n']} ({100 * s['solved'] / s['n']:.0f}%) | {s['t1']} | {s['t2']} | "
                     f"{s['t3']} | {s['acc']:.2f} | {s['exp_solved']:.1f} | {s['conflict']}/{s['fail']} |")
    return "\n".join(lines)


def figure(runs, path: Path):
    sizes = ["0.6B", "1.7B", "4B", "8B"]
    rate = lambda k: (100 * sum(r["correct"] for r in runs[k]) / len(runs[k])) if k in runs else None
    off = [rate(f"Qwen3-{s}") for s in sizes]
    on = [rate(f"Qwen3-{s}_think") for s in sizes]

    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    w, gap = 0.36, 0.02
    for i, (vals, color, label) in enumerate([(off, OFF, "关闭思考"), (on, ON, "开启思考")]):
        xs = [x + (i - 0.5) * (w + gap) for x in range(len(sizes))]
        labelled = False
        for x, v in zip(xs, vals):
            if v is None:
                continue
            ax.bar(x, v, w, color=color, label=None if labelled else label, zorder=2)
            labelled = True
            ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(sizes)), [f"Qwen3-{s}" for s in sizes], color=INK, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("精确发现的规则比例 (%)", color=MUTED, fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    ax.set_title("Mini AI Scientist：12 条隐藏规则 × 3 个种子（每组 36 个 episode）", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)


if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/sweep")
    runs = load(d)
    md = table(runs)
    (d / "summary.md").write_text(md + "\n")
    figure(runs, d / "solve_rate.png")
    print(md)
