#!/bin/bash
# vllm-rig.sh -- start vLLM for the cross-card follow-up (FOLLOWUP-vllm.md), the same flags on every card.
#
#   tools/vllm-rig.sh start [extra vllm args...]   apply the Turing clamp, start vLLM on PORT, wait for /health
#   tools/vllm-rig.sh stop
#
# Model: xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4 (W4A16, text only, the T4 rig's checkpoint).
# One request at a time (--max-num-seqs 1), prefix caching off so every prompt is computed in full,
# float16 (Turing has no bfloat16 path), seed 0. --max-model-len 2048 covers prompt + 1,024 tokens.
# The Turing clamp is gemma4-dev/gpu-vllm-t4-2b-w4a16/patch_triton_turing.py, copied here unchanged:
# vLLM forces Triton attention for Gemma 4, whose 512-wide heads overflow Turing's shared memory.
set -e
PY=${PY:-python3}
PORT=${PORT:-8090}
MODEL=${MODEL:-xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4}
LOG=${LOG:-$HOME/vllm-rig.log}
here=$(cd "$(dirname "$0")" && pwd)
case "$1" in
start)
  shift
  f=$($PY -c 'import importlib.util as u; print(u.find_spec("vllm.v1.attention.ops.triton_unified_attention").origin)')
  $PY "$here/patch_triton_turing.py" --check "$f" || $PY "$here/patch_triton_turing.py" "$f"
  $PY -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)' | tee "$LOG.versions"
  nohup $PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name e2b \
    --host 127.0.0.1 --port "$PORT" --dtype float16 --seed 0 --max-num-seqs 1 --max-model-len 2048 \
    --no-enable-prefix-caching --gpu-memory-utilization "${GPU_UTIL:-0.90}" "$@" > "$LOG" 2>&1 &
  echo $! > "$LOG.pid"
  for i in $(seq 1 900); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "healthy after ${i}s"; exit 0; }
    kill -0 "$(cat "$LOG.pid")" 2>/dev/null || { echo "vLLM exited; tail of $LOG:"; tail -30 "$LOG"; exit 1; }
    sleep 1
  done
  echo "not healthy after 900 s"; exit 1 ;;
stop)
  [ -f "$LOG.pid" ] && kill "$(cat "$LOG.pid")" 2>/dev/null; rm -f "$LOG.pid"; echo stopped ;;
*) sed -n 2,6p "$0"; exit 2 ;;
esac
