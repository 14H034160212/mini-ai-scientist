"""Draw the case's pipeline diagram -> docs/flow.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, LINE = "#0b0b0b", "#52514e", "#c3c2b7"
LLM, PY, LAB = "#2a78d6", "#1baf7a", "#eb6834"      # who does the step: model / Python / hidden lab

fig, ax = plt.subplots(figsize=(11, 4.6), dpi=160)
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.6)
ax.axis("off")


def box(x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=color, lw=1.8))
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=10, color=INK, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8, color=MUTED)


def arrow(x0, y0, x1, y1, label=""):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    if label:
        ax.text((x0 + x1) / 2 + 0.12, (y0 + y1) / 2, label, ha="left", va="center", fontsize=8, color=MUTED)


# row 1: the research loop
box(0.2, 3.0, 1.8, 1.0, "隐藏规则实验室", "f(a,b), a,b∈[0,20]", LAB)
box(2.5, 3.0, 1.8, 1.0, "开局观测", "3 个随机 (a,b,out)", LAB)
box(4.8, 3.0, 2.0, 1.0, "LLM 提出假设", "Qwen3 + thinking", LLM)
box(7.3, 3.0, 1.6, 1.0, "设计实验", "experiment ≤6 组/轮", LLM)
box(9.2, 3.0, 1.6, 1.0, "执行实验", "预算 30 次", LAB)
arrow(2.0, 3.5, 2.5, 3.5)
arrow(4.3, 3.5, 4.8, 3.5)
arrow(6.8, 3.5, 7.3, 3.5)
arrow(8.9, 3.5, 9.2, 3.5)

# row 2: analysis and checking
box(9.2, 1.5, 1.6, 1.0, "Python 计算", "compute 派生列", PY)
box(6.9, 1.5, 2.0, 1.0, "假设检验", "check 逐条 RIGHT/WRONG", PY)
box(4.8, 1.5, 1.8, 1.0, "实验记录本", "已检验假设与结论", PY)
arrow(10.0, 3.0, 10.0, 2.5, "")
arrow(9.2, 2.0, 8.9, 2.0)
arrow(6.9, 2.0, 6.6, 2.0)
arrow(5.7, 2.5, 5.8, 3.0, "修正假设")

# row 3: conclusion
box(2.4, 0.2, 2.0, 1.0, "提交关卡（可选）", "与自身数据矛盾则拒绝", PY)
box(0.2, 0.2, 1.8, 1.0, "441 点穷举验证", "精确等价 → 正确/错误", LAB)
box(4.8, 0.2, 1.8, 1.0, "submit", "提交规则表达式", LLM)
arrow(5.7, 1.5, 5.7, 1.2)
arrow(4.8, 0.7, 4.4, 0.7)
arrow(2.4, 0.7, 2.0, 0.7)

for i, (c, t) in enumerate([(LLM, "模型决策"), (PY, "Python 执行 / 校验"), (LAB, "黑盒实验室 / 真值")]):
    y = 1.0 - i * 0.32
    ax.add_patch(FancyBboxPatch((7.4, y), 0.18, 0.18, boxstyle="round,pad=0.01", fc="white", ec=c, lw=1.8))
    ax.text(7.7, y + 0.09, t, va="center", fontsize=8, color=MUTED)

out = Path(__file__).resolve().parents[1] / "docs" / "flow.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
