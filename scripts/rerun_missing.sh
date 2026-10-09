#!/bin/bash
# After run_sweep.sh: rerun every sweep config whose result file is missing (e.g. crashed).
cd "$(dirname "$0")/.."
[ -f scripts/vllm_env.sh ] && source scripts/vllm_env.sh
VLLM_PY=${VLLM_PY:-python}
while pgrep -f "[r]un_sweep.sh" >/dev/null; do sleep 30; done
OUT=results/sweep
check() { local tag=$1; shift; [ -f $OUT/$tag.jsonl ] && return
  echo "== $(date +%T) rerun $tag"; $VLLM_PY scripts/run_eval.py --backend vllm --seeds 3 --out $OUT "$@" 2>&1 | grep -E "solved|tier|Traceback|Error"; }
for m in Qwen3-8B Qwen3-4B Qwen3-1.7B Qwen3-0.6B; do
  check ${m}_think --model Qwen/$m --think; check $m --model Qwen/$m; done
for m in Qwen3-8B Qwen3-4B Qwen3-1.7B; do check ${m}_think_gate --model Qwen/$m --think --gate; done
check Qwen3-8B_gate --model Qwen/Qwen3-8B --gate
for m in Qwen2.5-7B-Instruct Qwen2.5-3B-Instruct Qwen2.5-1.5B-Instruct; do check $m --model Qwen/$m; done
check Qwen3-8B_nocompute_think --model Qwen/Qwen3-8B --think --no-compute
check Qwen3-8B_nocheck_think --model Qwen/Qwen3-8B --think --no-check
echo "== $(date +%T) reruns done"
