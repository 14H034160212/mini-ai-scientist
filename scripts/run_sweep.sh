#!/bin/bash
# Full sweep for the case study: model size x thinking, plus tool ablations.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_sweep.sh   (~4 h on one A100)
cd "$(dirname "$0")/.."
[ -f scripts/vllm_env.sh ] && source scripts/vllm_env.sh   # optional, machine-specific: export VLLM_PY=...
VLLM_PY=${VLLM_PY:-python}
OUT=results/sweep; mkdir -p $OUT
run() { echo "== $(date +%T) $*"; $VLLM_PY scripts/run_eval.py --backend vllm --seeds 3 --out $OUT "$@" 2>&1 | grep -E "solved|tier|Traceback|Error"; }
for m in Qwen3-8B Qwen3-4B Qwen3-1.7B Qwen3-0.6B; do
  run --model Qwen/$m --think
  run --model Qwen/$m
done
# submit gate: a conclusion that contradicts the agent's own data is refused
for m in Qwen3-8B Qwen3-4B Qwen3-1.7B; do
  run --model Qwen/$m --think --gate
done
run --model Qwen/Qwen3-8B --gate
for m in Qwen2.5-7B-Instruct Qwen2.5-3B-Instruct Qwen2.5-1.5B-Instruct; do
  run --model Qwen/$m
done
run --model Qwen/Qwen3-8B --think --no-compute
run --model Qwen/Qwen3-8B --think --no-check
echo "== $(date +%T) sweep done"
