"""Build mini_ai_scientist.ipynb (the case, in Chinese) following the 紫皮书 case structure.

  python scripts/build_notebook.py

The sweep table in section 6 is generated from results/sweep/*.jsonl, so the
numbers in the notebook are always the numbers on disk.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze import load, table  # noqa: E402

C = []


def _src(s):
    return s.strip().splitlines(keepends=True)


def md(s):
    C.append({"cell_type": "markdown", "metadata": {}, "source": _src(s)})


def code(s):
    C.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(s)})


sweep_dir = ROOT / "results" / "sweep"
SWEEP_TABLE = table(load(sweep_dir)) if sweep_dir.exists() else "（尚未运行 scripts/run_sweep.sh）"

# ---------------------------------------------------------------- 0. 基本信息
md("""
# Mini AI Scientist：让 4B 开源小模型自主做实验、发现隐藏规律，并自动验证对错

| 项目 | 内容 |
|---|---|
| **一句话简介** | 给开源模型一台"黑盒机器"（输入两个整数、输出一个整数，内部规则未知），让它自己**提出假设 → 设计实验 → 调用 Python 分析 → 修正假设**，最终提交一条规则；规则在全部 441 个输入上做穷举等价验证，**不需要 LLM 当裁判**。 |
| **适用对象** | 做 LLM Agent / 推理 / 评测的工程师与研究者；想把"AI Scientist"落到可量化实验的同学；需要可验证奖励环境（如 GRPO）的训练者。 |
| **难度与耗时** | 进阶。首次复现约 60 分钟（环境 10 分钟 + 下载模型 5 分钟 + Notebook 运行约 30–45 分钟，视 GPU 而定）。 |
| **运行方式** | 魔搭 Notebook（单卡 GPU，≥24GB 显存）；vLLM 0.11.0 本地推理，无需任何 API Key。 |
| **主要模型与工具** | Qwen3-4B（思考模式，主模型）；Qwen3-0.6B/1.7B/8B、Qwen2.5-Instruct 用于规模对比；vLLM、Python AST 沙箱。 |
""")

# ---------------------------------------------------------------- 1. 场景与问题
md("""
## 1. 场景与问题

**谁在什么情况下遇到了什么问题？** 我们在做"AI Scientist"类 Agent 时，常见做法是让大模型直接做开放式科研（提课题、写代码、写论文）。这对小模型太难，更关键的是**结果无法判断对错**：模型说"我发现了 X"，没有标准答案可以核对，也就无法比较不同模型、不同 Agent 设计的好坏。

**痛点**
- 开放式任务没有真值，评测只能靠 LLM 裁判或人工，成本高、口径不稳；
- 不知道小模型（0.6B–8B）到底能不能做"科学发现"，也不知道瓶颈在模型能力还是 Agent 设计；
- 失败时看不到原因：是实验设计不好、数据分析算错，还是根本没检查就下结论？

**本 Case 的输入、输出与完成边界**

| | 内容 |
|---|---|
| 输入 | 一台隐藏规则的黑盒机器（12 条规则、3 个难度等级）、实验预算 30 次、最多 15 轮交互 |
| 输出 | 模型提交的规则（Python 表达式）、完整科研过程记录（JSONL）、自动评测报告（正确率、实验次数、失败原因） |
| 完成标准 | 提交规则在全部 441 个输入上与真实规则**完全一致**才算"发现成功" |
| 边界 | 规则是两个整数输入上的确定性函数；不涉及噪声、连续变量和真实物理实验（见第 8 节拓展方向） |
""")

# ---------------------------------------------------------------- 2. 解决思路
md("""
## 2. 解决思路

核心思想：**把"科研"压缩成一个最小但完整、且可以穷举验证的闭环**，并让模型与 Python 各司其职——模型负责*决定*（提什么假设、做什么实验、何时下结论），Python 负责*执行与校验*（运行实验、计算、核对假设），黑盒实验室负责*真值*。

![流程图](docs/flow.png)

| 模块 | 负责方 | 作用 |
|---|---|---|
| 隐藏规则实验室 `lab.py` | 实验室 | 保存真实规则；执行实验；最终在 441 个输入上做等价验证 |
| Agent 循环 `agent.py` | 模型 + Python | 每轮模型输出一个 JSON 动作：`experiment` / `compute` / `check` / `submit` |
| 安全沙箱 `compile_expr` | Python | 只允许算术、比较、`X if C else Y`、`abs/min/max`，拒绝任意代码执行 |
| 实验记录本 | Python | 保留最近 3 轮对话，更早的假设与检验结论压缩成记录，控制上下文长度 |
| 提交关卡（可选） | Python | 拒绝与模型**自己已观测数据**相矛盾的结论 |
| 推理后端 `backend.py` | vLLM | 批量推理；剥离 `<think>` 思考过程，只把最终动作写回对话 |

**为什么这样选型**
- **Qwen3 系列**：同一家族覆盖 0.6B–8B，并且可以按请求开关思考模式，适合做"规模 × 思考"的对照；
- **vLLM**：思考模式每轮会生成数千 token，transformers 太慢（实测 Qwen3-8B：transformers 跑 4 个 episode 需 28 分钟，vLLM 批量跑 36 个只需 33 分钟）；
- **穷举验证而非 LLM 裁判**：输入空间只有 21×21=441，等价性可以精确判定，评测零成本、零歧义。
""")

# ---------------------------------------------------------------- 3. 前置准备
md("""
## 3. 前置准备

| 项目 | 要求 / 版本 |
|---|---|
| 平台 | 魔搭 Notebook，GPU 环境（单卡 ≥24GB 显存，如 A10 24GB）；本文实测环境为 Linux + 单卡 A100 80GB |
| 运行时 | Python 3.11 |
| 依赖 | `vllm==0.11.0`（自带 `torch==2.8.0+cu128`）、`transformers>=4.56,<5`、`modelscope`、`pandas`、`matplotlib` |
| 驱动 | NVIDIA 驱动需支持 CUDA 12.8 及以上 |
| 模型 | `Qwen/Qwen3-4B`（约 8GB，从魔搭下载）；可选 `Qwen/Qwen3-8B`（约 16GB） |
| 数据 | 无需下载：12 条隐藏规则定义在代码中，开局观测由随机种子生成，完全可复现 |
| 权限 | 无需任何 API Key / Token |

**注意事项（复现中真实遇到过的问题）**
1. **transformers 5.x 与 vLLM 0.11 不兼容**（报错 `Qwen2Tokenizer has no attribute all_special_tokens_extended`），必须安装 `transformers<5`；安装后**重启 Kernel**。
2. **CUDA 13 版本的 vLLM 需要更新的驱动**（报错 `CUDA driver version is insufficient`）。请使用本文固定的 `vllm==0.11.0`（CUDA 12.8 构建）。
3. 在 `.py` 脚本中使用 vLLM 时，入口代码必须放在 `if __name__ == "__main__":` 下（vLLM 以 spawn 方式启动子进程）；Notebook 中无此问题。
""")
code("""
# [单元格 1] 安装依赖（魔搭镜像如已满足可跳过）。安装完成后请重启 Kernel 再继续。
# !pip install -q "vllm==0.11.0" "transformers>=4.56,<5" modelscope pandas matplotlib
""")
code("""
# [单元格 2] 获取代码并设置模型来源
# !git clone https://github.com/14H034160212/mini-ai-scientist.git mini_ai_scientist
import os, sys
if os.path.isdir("mini_ai_scientist"):
    os.chdir("mini_ai_scientist")
sys.path.insert(0, ".")
os.environ.setdefault("USE_MODELSCOPE", "1")          # 1 = 从魔搭下载权重；0 = 使用 Hugging Face 缓存

import platform, torch, transformers, vllm
print("Python", platform.python_version(), "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| vllm", vllm.__version__)
print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
""")
md("""
**检查方法**：输出中 `transformers` 版本应为 4.x，`vllm` 为 0.11.0，并能看到 GPU 型号与显存。
""")

# ---------------------------------------------------------------- 4. 分步实战
md("""
## 4. 分步实战

先验证不依赖模型的模块（步骤 1–3，CPU 即可），再接入模型（步骤 4–7）。
""")

md("""
### 步骤 1：搭建隐藏规则实验室

- **目标**：构造研究对象——一台规则未知、但真值可穷举验证的黑盒机器。
- **输入与操作**：`RULES` 中 12 条规则，按难度分三档：Tier 1 单一公式；Tier 2 一个条件分支；Tier 3 嵌套条件或取模条件。`Lab(rule, seed)` 会用 `seed` 随机生成 3 个开局观测。
- **过程与输出**：打印全部规则、一个实验室的开局观测，以及一次实验的结果。
- **检查方法**：每条真实规则对自身验证必须 100% 正确；同一 `seed` 的开局观测必须完全相同（可复现）。
""")
code("""
from mini_scientist import Lab, RULES, RULES_BY_NAME

for r in RULES:
    print(f"Tier {r.tier}  {r.name:12s} {r.expr}")

lab = Lab(RULES_BY_NAME["parity_a"], seed=0)
print("\\n开局观测 (a, b, out):", lab.observations)
print("做一次实验 [(3,4), (4,4)] ->", lab.run([(3, 4), (4, 4)]), "| 已用预算", lab.used, "/", lab.budget)

# 检查
assert all(Lab(r).verify(r.expr)["correct"] for r in RULES)
assert Lab(RULES[0], seed=7).observations == Lab(RULES[0], seed=7).observations
print("\\n✅ 步骤 1 通过：12 条规则自检正确，开局观测可复现")
""")

md("""
### 步骤 2：让模型"调用 Python"的安全沙箱

- **目标**：模型写的规则与计算式要被真正执行，但不能执行任意代码。
- **输入与操作**：`compile_expr` 解析表达式的语法树，只放行算术、比较、`X if C else Y` 与 `abs/min/max`。
- **过程与输出**：对合法与恶意表达式分别打印"允许/拒绝"；演示"语义等价即正确"。
- **检查方法**：`__import__`、属性访问、推导式、超大指数必须被拒绝；`abs(a-b)` 与 `max(a,b)-min(a,b)` 都应被判为 `absdiff` 的正确答案。
""")
code("""
from mini_scientist.lab import compile_expr, ExprError

for src in ["abs(a - b)", "a * b if a % 2 else a + b", "__import__('os').system('ls')", "a.__class__", "a ** 99"]:
    try:
        compile_expr(src); print("✓ 允许:", src)
    except ExprError as e:
        print("✗ 拒绝:", src, "->", e)

absdiff = Lab(RULES_BY_NAME["absdiff"])
print("\\nmax(a,b)-min(a,b) ->", absdiff.verify("max(a, b) - min(a, b)"))
print("a - b              ->", absdiff.verify("a - b"))

# 检查
for bad in ["__import__('os')", "a.__class__", "[a for a in b]", "a ** 99"]:
    try: compile_expr(bad); raise AssertionError(bad)
    except ExprError: pass
assert absdiff.verify("max(a, b) - min(a, b)")["correct"] and not absdiff.verify("a - b")["correct"]
print("\\n✅ 步骤 2 通过：危险代码被拒绝，等价规则判为正确")
""")

md("""
### 步骤 3：Agent 工具与协议（先用"脚本模型"空跑）

- **目标**：在接入真实模型前，确认 Agent 循环、四个工具和评测流程本身没有问题。
- **输入与操作**：模型每轮输出一个 JSON 动作：

| 工具 | 作用 | 消耗实验预算 |
|---|---|---|
| `experiment` | 选择最多 6 组输入跑实验 | 是（共 30 次） |
| `compute` | 让 Python 在全部观测上计算派生列，例如 `out - a*b`，并提示哪一列是常数 | 否 |
| `check` | 用候选规则逐条核对已观测数据，列出 RIGHT / WRONG 行 | 否 |
| `submit` | 提交最终规则，结束研究 | — |

  这里用一个按固定剧本回复的 `Scripted` 假模型走一遍：做实验 → 计算 → 检验错误假设 → 检验正确假设 → 提交。
- **过程与输出**：打印系统提示词（模型实际看到的指令）与空跑结果。
- **检查方法**：空跑必须"发现成功"，并且实验 3 次、检验 2 次、计算 1 次、无格式错误。
""")
code("""
from mini_scientist import Episode, run_batch
from mini_scientist.agent import system_prompt

print(system_prompt(["experiment", "compute", "check", "submit"]))

class Scripted:
    \"\"\"按固定剧本回复的假模型，用来检验 Agent 循环本身。\"\"\"
    script = ['{"tool": "experiment", "inputs": [[5, 2], [2, 5], [7, 7]]}',
              '{"tool": "compute", "columns": ["out - (a - b)", "out - abs(a - b)"]}',
              '{"tool": "check", "expr": "a - b"}',
              '{"tool": "check", "expr": "abs(a - b)"}',
              '{"tool": "submit", "expr": "abs(a - b)"}']
    step = 0
    def generate(self, convs):
        reply = self.script[self.step]; self.step += 1
        return [reply] * len(convs)

dry = Episode(Lab(RULES_BY_NAME["absdiff"], seed=0))
run_batch(Scripted(), [dry])
for t in dry.trace:
    print(f"第{t['turn']}轮 {t['action']['tool']:10s} ->", (t.get('feedback') or str(t.get('ran')))[:110].replace("\\n", " | "))
res = dry.result()
print(res)

# 检查
assert res["correct"] and res["experiments"] == 3 and res["checks"] == 2 and res["computes"] == 1 and res["parse_errors"] == 0
print("\\n✅ 步骤 3 通过：Agent 循环与评测流程正确")
""")

md("""
### 步骤 4：加载模型（vLLM + 思考模式）

- **目标**：在 GPU 上加载 Qwen3-4B，开启思考模式。
- **输入与操作**：`VLLMBackend(MODEL, thinking=True)`。关键参数：
  - `max_tokens=8192`：每轮思考 + 回答的上限；
  - `max_model_len=20480`：上下文长度（实验记录本使对话保持很短，这个长度足够）；
  - `temperature=0.6, top_p=0.95, top_k=20`：Qwen3 官方推荐的思考模式采样参数（贪心解码会让思考模型陷入循环）；
  - `gpu_memory_utilization=0.85`：24GB 显卡可正常运行 4B；若显存不足请调小或换 `Qwen/Qwen3-1.7B`。
- **过程与输出**：首次运行会从魔搭下载约 8GB 权重，并做 CUDA Graph 编译（约 2–4 分钟）。随后发一个最简单的请求做冒烟测试。
- **检查方法**：冒烟测试的回复能被解析成带 `tool` 字段的 JSON 动作。（为避免泄露本机路径，保存的输出中已用 scripts/strip_notebook_logs.py 去掉 vLLM 启动日志。）
""")
code("""
import time
from mini_scientist.backend import VLLMBackend
from mini_scientist.agent import parse_action

MODEL = "Qwen/Qwen3-4B"                     # 显存充足可换 "Qwen/Qwen3-8B"
t0 = time.time()
backend = VLLMBackend(MODEL, thinking=True, max_tokens=8192, max_model_len=20480, gpu_memory_utilization=0.85)
print(f"模型加载用时 {time.time() - t0:.0f}s")

smoke = Episode(Lab(RULES_BY_NAME["sum"], seed=0))
reply = backend.generate([smoke.messages])[0]
print("模型第一轮回复:", reply[:400])
print("思考过程长度:", backend.think_chars[-1], "字符")

# 检查
assert parse_action(reply) is not None, "回复无法解析为 JSON 动作"
print("\\n✅ 步骤 4 通过：模型可以按协议输出动作")
""")

md("""
### 步骤 5：一次完整的科研过程

- **目标**：观察模型如何从 3 个观测出发，一步步做实验、检验、修正，直到提交。
- **输入与操作**：规则 `parity_a`（Tier 2：a 为偶数时 `a+b`，否则 `a*b`），开局种子 0。模型看不到真实规则。
- **过程与输出**：逐轮打印模型的分析、动作、实验结果和工具反馈，最后打印提交规则与 441 点验证结果。
- **检查方法**：过程完整结束并给出验证结果。由于采样具有随机性，单次结果可能成功也可能失败——**这正是第 6 节要用多条规则、多个种子统计的原因**。
""")
code("""
def show(ep):
    for t in ep.trace:
        act = t["action"] or {}
        print(f"── 第 {t['turn']} 轮  [{act.get('tool', '思考被截断' if t['raw'] == '[thinking truncated]' else '无法解析')}]")
        if act.get("analysis"): print("  分析:", act["analysis"][:300])
        if act.get("inputs"):   print("  实验:", act["inputs"])
        if act.get("columns"):  print("  计算:", act["columns"])
        if act.get("expr"):     print("  规则:", act["expr"])
        if t.get("ran"):        print("  结果:", ", ".join(f"f({a},{b})={y}" for a, b, y in t["ran"]))
        if t.get("feedback") and t["feedback"] != "submitted":
            print("  反馈:", t["feedback"].splitlines()[0][:200])
    res = ep.result()
    print(f"\\n真实规则: {res['truth']}\\n提交规则: {res['submitted']}\\n"
          f"441 个输入上一致率: {res['accuracy']:.0%}   {'✅ 发现成功' if res['correct'] else '❌ 未完全发现'}")
    return res

t0 = time.time()
ep = Episode(Lab(RULES_BY_NAME["parity_a"], seed=0))
run_batch(backend, [ep])
res = show(ep)
print(f"用时 {time.time() - t0:.0f}s")

# 检查
assert ep.done and "accuracy" in res
print("\\n✅ 步骤 5 通过：科研过程完整结束并得到验证结果")
""")

md("""
### 步骤 6：批量评测（12 条规则 × 1 个种子）

- **目标**：用统一口径衡量 Agent 的发现能力。
- **输入与操作**：12 条规则各建一个 episode（种子 0），所有 episode **同步推进**、每轮一次批量推理。
- **过程与输出**：保存完整过程到 `results/notebook/Qwen3-4B_think.jsonl`，输出按难度分组的统计表。
- **检查方法**：文件有 12 行；作为参考，我们用同一配置跑 3 个种子，每个种子解出 6–7/12、Tier 1 为 3–4/4。因此**正常范围**为：总体 ≥4/12 且 Tier 1 ≥3/4。
""")
code("""
import json, pandas as pd
from pathlib import Path

def evaluate(backend, seeds=(0,), **episode_kw):
    eps = [Episode(Lab(r, seed=s), **episode_kw) for r in RULES for s in seeds]
    t0 = time.time()
    results = run_batch(backend, eps)
    return eps, pd.DataFrame(results), time.time() - t0

def summarize(df):
    return df.groupby("tier").agg(解出=("correct", "sum"), 总数=("correct", "size"), 平均一致率=("accuracy", "mean"),
                                  平均实验次数=("experiments", "mean"), 被截断轮数=("truncated", "sum"))

eps, df, sec = evaluate(backend)
out = Path("results/notebook"); out.mkdir(parents=True, exist_ok=True)
with open(out / "Qwen3-4B_think.jsonl", "w") as f:
    for e, r in zip(eps, df.to_dict("records")):
        f.write(json.dumps({**r, "trace": e.trace}, ensure_ascii=False) + "\\n")

print(f"用时 {sec:.0f}s，总体解出 {df.correct.sum()}/{len(df)}")
display(df[["rule", "tier", "correct", "accuracy", "experiments", "submitted"]])
summarize(df)
""")
code("""
# 检查
n_lines = sum(1 for _ in open("results/notebook/Qwen3-4B_think.jsonl"))
tier1 = df[df.tier == 1].correct.sum()
print(f"结果文件 {n_lines} 行；总体 {df.correct.sum()}/12；Tier 1 {tier1}/4")
assert n_lines == 12
assert df.correct.sum() >= 4 and tier1 >= 3, "低于正常范围：请检查依赖版本与采样参数（见第 3、8 节）"
print("✅ 步骤 6 通过：结果在正常范围内")
""")

md("""
### 步骤 7：分析失败——模型在和自己的数据"唱反调"

- **目标**：弄清失败到底是"找不到规律"，还是别的原因。
- **输入与操作**：对每个失败的 episode，把模型提交的规则放回它**自己做过的全部实验数据**上逐条核对。
- **过程与输出**：统计失败中"提交规则与自身数据矛盾"的比例。
- **检查方法**：输出各失败 episode 的矛盾行数。

在完整扫描中，不开思考的模型这一比例非常高（Qwen3-8B 不开思考时 24 次失败中 23 次如此），开思考后明显下降（Qwen3-8B 思考模式 20 次失败中 10 次）。也就是说，大量失败**不是不会发现，而是没检查就下结论**——这恰好是人类科研里最基本的规范："结论必须与数据一致"。
""")
code("""
def own_data_conflicts(row, trace):
    lab = Lab(RULES_BY_NAME[row["rule"]], seed=row["seed"])
    for t in trace:
        for a, b, y in (t.get("ran") or []):
            lab.observations.append((a, b, y))
    try:
        return len(lab.check(row["submitted"] or ""))
    except ExprError:
        return -1                                   # 提交的表达式无法解析

fails = [(e.result(), e.trace) for e in eps if not e.result()["correct"]]
rows = [{"rule": r["rule"], "submitted": r["submitted"], "与自身数据矛盾的行数": own_data_conflicts(r, tr)} for r, tr in fails]
print(f"失败 {len(rows)} 个，其中提交规则与自身数据矛盾的 {sum(x['与自身数据矛盾的行数'] != 0 for x in rows)} 个（-1 表示未提交或无法解析）")
pd.DataFrame(rows)
""")

md("""
### 步骤 8：加一道"提交关卡"，再评测一次

- **目标**：把"结论必须与自己的数据一致"写进 Agent：提交时若与已观测数据矛盾，就拒绝并展示矛盾行，让模型继续研究。
- **输入与操作**：`Episode(..., gate=True)`，其余设置与步骤 6 完全相同。
- **过程与输出**：保存到 `results/notebook/Qwen3-4B_think_gate.jsonl`，与步骤 6 对比。
- **检查方法**：被拒绝的提交次数 `rejected` > 0 时，关卡确实拦截了矛盾结论；对比两次的解出数。单种子差异会受随机性影响，3 个种子的结果见第 6 节。
""")
code("""
eps_g, df_g, sec_g = evaluate(backend, gate=True)
with open(out / "Qwen3-4B_think_gate.jsonl", "w") as f:
    for e, r in zip(eps_g, df_g.to_dict("records")):
        f.write(json.dumps({**r, "trace": e.trace}, ensure_ascii=False) + "\\n")

print(f"用时 {sec_g:.0f}s；无关卡 {df.correct.sum()}/12 → 有关卡 {df_g.correct.sum()}/12；"
      f"被拦截的矛盾提交 {df_g.rejected.sum()} 次")
pd.DataFrame({"无关卡": summarize(df)["解出"], "有关卡": summarize(df_g)["解出"]})
""")

# ---------------------------------------------------------------- 5. 完整运行
md(f"""
## 5. 完整运行与最终结果

Notebook 中的步骤 6–8 是单种子的精简版。完整实验用命令行脚本一次跑完（实测单卡 A100 约 4 小时）：

```bash
python scripts/test_lab.py                    # 无需 GPU 的自检，应输出 all lab tests passed
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sweep.sh
python scripts/analyze.py results/sweep       # 生成 results/sweep/summary.md 与 solve_rate.png
```

- **输入**：12 条规则 × 3 个种子 = 每个配置 36 个 episode；
- **配置**：Qwen3-0.6B/1.7B/4B/8B × 思考开/关；提交关卡；Qwen2.5-Instruct（非推理模型对照）；去掉 `compute` / `check` 工具的消融；
- **产物**：`results/sweep/<配置>.jsonl`（每个 episode 的完整过程与评分）、`summary.md`、`solve_rate.png`。

**完整扫描结果（真实运行，36 个 episode / 配置）**

{SWEEP_TABLE}

![各规模模型开/关思考的解出率](results/sweep/solve_rate.png)

**结果解读**（每组 36 个 episode；解出率在 50% 附近时，单组的标准误约为 8 个百分点，小于此量级的差异不应过度解读）

1. **思考模式比参数量更重要。** 每个规模开启思考后解出率都明显提升：0.6B 0%→19%，1.7B 14%→39%，4B 28%→56%，8B 33%→44%。开启思考的 1.7B（39%）已超过不开思考的 8B（33%）。4B 与 8B（开思考）之间的差距在随机波动范围内——**在这个任务上，4B 已经够用**，魔搭免费 GPU 可以轻松运行。
2. **思考带来的主要收益，是"结论与自己的数据一致"。** 表格最后一列统计失败 episode 中提交规则与模型*自己做过的实验*相矛盾的比例：不开思考时几乎全部如此（8B 23/24、4B 25/26、1.7B 28/31）；开思考后大幅下降（8B 10/20、4B 7/16、1.7B 5/22）。不思考的模型失败，多半不是"找不到规律"，而是"没核对就下结论"。
3. **提交关卡能拦住矛盾结论，但能否改对取决于模型。** 关卡在 8B 思考（16→19）与 1.7B 思考（14→16）上小幅提升，在 4B 思考上持平（20→20，拦截 22 次），在 8B 不思考上无效（12→11，拦截 48 次，被拦下后反复提交错误规则直到轮数用完）。注意：为保证每个 episode 都能结束，最后一轮的提交不经过关卡，所以开启关卡后失败中仍可能出现矛盾结论。
4. **工具要匹配模型能力。** `compute` 是为不会思考、心算容易出错的模型设计的；开启思考的模型几乎不用它（8B 每个 episode 平均 0.36 次，1.7B 仅 0.06 次）。去掉 `compute`（20/36）或 `check`（17/36）后，8B 思考的成绩没有变差（完整工具 16/36）。
5. **上一代非推理模型明显更弱。** Qwen2.5-7B / 3B-Instruct 均为 5/36，1.5B 为 0/36；3B 到 7B 没有提升。
6. **Tier 3 仍是能力边界。** 嵌套或取模条件的规则，所有配置中最好的成绩是 2/12（8B 思考 + 提交关卡）。
""")

# ---------------------------------------------------------------- 6. 评测与验收
md("""
## 6. 评测与验收

**功能验收**
- 步骤 1–3 的断言全部通过（无需 GPU）；
- 步骤 6、8 生成 `results/notebook/*.jsonl`，每行包含提交规则、441 点一致率、实验次数与完整过程；
- `python scripts/test_lab.py` 输出 `all lab tests passed`。

**质量验收**（指标与口径）
- **发现成功率**：提交规则在 441 个输入上与真实规则完全一致的比例，按难度分层报告；
- **一致率**：提交规则与真实规则一致的输入比例（部分正确时的连续指标）；
- **实验效率**：成功 episode 平均使用的实验次数；
- **正常范围**：Qwen3-4B 思考模式单种子总体 ≥4/12、Tier 1 ≥3/4（依据第 5 节 3 个种子的实测）。

**对照基线：不用 LLM 的穷举搜索**——随机做实验，并在约 1.2 万条 `F if C else G` 候选规则中排除与数据不符者：
""")
code("""
from mini_scientist.baseline import enumerative_scientist
bl = pd.DataFrame([enumerative_scientist(Lab(r, seed=s), seed=s) for r in RULES for s in range(3)])
print(f"穷举基线：{bl.correct.sum()}/{len(bl)}")
bl.groupby("tier").agg(解出=("correct", "sum"), 总数=("correct", "size"), 平均实验次数=("experiments", "mean"))
""")
md("""
穷举基线的两个局限，正好说明了 LLM 科学家的价值与差距：**表达能力受限于语法**（两层嵌套的 `nested` 不在候选空间，永远发现不了）；**实验是盲目的**（即使 `a + b` 也几乎用光 30 次预算，而 LLM 通常 4–6 次即可）。

**异常验收**（至少一个典型失败或边界输入，以及系统如何处理）

| 异常情况 | 真实出现在 | 系统表现 |
|---|---|---|
| 思考过长被截断（未输出 `</think>`） | Qwen3 思考模式约 10–20% 的轮次（8B 24/194，4B 31/158） | 不把半截思考当答案写回对话；提示"想得更简短一些"，并计入 `truncated` |
| 输出格式不合规（如 LaTeX 转义 `\\(`、输入写成 `{"a":3,"b":4}`） | 小模型常见 | 宽松解析；无法解析时计入 `parse_errors` 并提示重试，单个 episode 出错不会中断整批评测 |
| 重复同一检验、原地打转 | Qwen2.5-7B | 拦截重复的 `check`/`compute`，提示去做新实验 |
| 提交与自身数据矛盾的结论 | 各规模模型普遍 | 开启提交关卡时拒绝并列出矛盾行 |
| 恶意代码 | 演示用 | AST 白名单拒绝（步骤 2） |
""")
code("""
# 异常验收演示：截断、畸形输入、关卡拒绝（无需 GPU）
from mini_scientist.agent import TRUNCATED
e = Episode(Lab(RULES_BY_NAME["absdiff"]), gate=True)
e.step(TRUNCATED);                                                          print("截断   ->", e.trace[-1]["feedback"])
e.step('{"tool": "experiment", "inputs": [{"a": 2, "b": 5}, "junk"]}');    print("畸形输入 ->", e.trace[-1]["ran"])
e.step('{"tool": "submit", "expr": "a - b"}');                              print("矛盾提交 ->", e.trace[-1]["feedback"].splitlines()[0])
assert e.n_truncated == 1 and e.n_rejected == 1 and not e.done
print("✅ 异常处理符合预期")
""")

# ---------------------------------------------------------------- 7. 限制、排错与迁移
md("""
## 7. 已知限制、排错与可复用方向

**已知限制**
- 规则空间较小（两个整数输入、确定性输出），离真实科研仍有距离；
- 采样带来随机性：单次结果会波动，结论应以多种子统计为准（本文每个配置 36 个 episode）；
- Tier 3（嵌套 / 取模条件）对 8B 以内的模型仍然很难；
- 思考模式的计算开销约为非思考模式的 10–14 倍，且有一部分轮次因思考过长被截断。

**常见问题**

| 现象 | 原因 | 解决 |
|---|---|---|
| `Qwen2Tokenizer has no attribute all_special_tokens_extended` | transformers 5.x 与 vLLM 0.11 不兼容 | `pip install "transformers>=4.56,<5"` 后重启 Kernel |
| `CUDA driver version is insufficient for CUDA runtime version` | 安装了 CUDA 13 构建的 vLLM | 固定 `vllm==0.11.0` |
| `Free memory on device ... is less than desired GPU memory utilization` | 显存被其他进程占用 | 调小 `gpu_memory_utilization`，或结束占用进程 |
| 脚本中报 `An attempt has been made to start a new process before ... bootstrapping` | vLLM 用 spawn 启动子进程 | 入口放在 `if __name__ == "__main__":` 下 |
| `decoder prompt is longer than the maximum model length` | 截断的思考被当成答案写回对话 | 已在 `split_thinking` 中处理；如自行修改后端请保留此逻辑 |

**可复用方向**
- 换成带噪声的观测、三元输入或浮点输出，考察"重复实验"与统计推断；
- 环境自带可验证奖励（441 点精确匹配），可直接用于 GRPO 等强化学习训练小模型的"科研策略"；
- 同样的"模型决策 + Python 执行 + 真值验证"结构可迁移到：数据分析 Agent 的自动评测、SQL / 公式推断、系统辨识与故障定位。
""")

# ---------------------------------------------------------------- 8. 代码数据参考
md("""
## 8. 代码、数据与参考资料

- 代码仓库：`https://github.com/14H034160212/mini-ai-scientist`（包含 `mini_scientist/` 源码、`scripts/`、本 Notebook 与 `results/sweep/` 实测结果）
- 数据：无外部数据；12 条规则定义于 `mini_scientist/lab.py`，开局观测由随机种子生成
- 模型：[Qwen/Qwen3-4B](https://modelscope.cn/models/Qwen/Qwen3-4B)、[Qwen/Qwen3-8B](https://modelscope.cn/models/Qwen/Qwen3-8B)、[Qwen/Qwen3-1.7B](https://modelscope.cn/models/Qwen/Qwen3-1.7B)、[Qwen/Qwen3-0.6B](https://modelscope.cn/models/Qwen/Qwen3-0.6B)（Apache-2.0）
- 推理框架：[vLLM](https://github.com/vllm-project/vllm)（Apache-2.0）
- 相关工作：Sakana AI *The AI Scientist*；BoxingGym（实验设计基准）；Popper（LLM Agent 的可证伪检验）
""")

nb = {"cells": C, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}}
(ROOT / "mini_ai_scientist.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", ROOT / "mini_ai_scientist.ipynb", f"({len(C)} cells)")
