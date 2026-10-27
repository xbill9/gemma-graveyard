# CLAUDE.md

Hacktoberfest Week 4 entry: which GSM8K answers Gemma 4 E2B loses at each quantization rung, from BF16
down to Q2_K. `SPEC.md` is the working plan; its checks are go/no-go gates, run in order.
`RULES-week4.md` is the challenge text, `TEMPLATE-week4.md` the required post headings.

## Deadline

2026-10-31. Any commit after it must be noted in `README.md`. Credit reused code (`lazy_eval.py` and the
GSM8K split come from `xbill9/lazy-gemma`) in the README and the post.

## Measurement rules

- Correctness is graded by code, never by a model. Deaths, ghosts and every other count come from
  `graveyard.py bury`, never from reading rows.
- Every rung runs on the same machine, llama.cpp build, questions and sampling. Speed and memory are
  compared only within one machine.
- The BF16 reference is the checkpoint's own dtype. F16 is a rung, not a reference.
- A tombstone counts only after check 4 (determinism) has passed on that machine.
- Anything not measured here is marked as unmeasured.

## Gemma 4 + llama-server

Thinking goes to `reasoning_content`; only `content` is graded. Use `/v1/chat/completions`. Thinking is
off by default (`--think` turns it on), so `max_tokens` 1024 is enough.

## Host: the T4 VM (chosen 2026-10-27)

Facts below are from `~/gemma4-dev/gpu-vllm-t4-2b/tpu.env` (measured 2026-09-17/18). Re-check each one.

- GCE `n1-standard-2`, `us-west2-b`: Tesla T4, 15360 MiB, compute capability 7.5, 2 vCPU, 7.8 GB RAM.
- **Turing has no bfloat16 datapath.** BF16 stays the accuracy reference, but its speed on this card is
  not comparable with the other rungs. Record whether llama.cpp's CUDA backend offloads the BF16 GGUF at
  all, and report BF16 speed separately. F16 is the fastest 16-bit rung here.
- **Disk: work on `/opt1`** (~249 GB free). `/` has ~4 GB and `/tmp` ~3.9 GB: the BF16 checkpoint, the
  nine GGUFs and a llama.cpp build do not fit there. Put the HF download, `~/graveyard-gguf` and any
  llama.cpp build under `/opt1` (`~/.cache` already links there).
- **Swap is not persistent.** No swap as provisioned; the 16 GB swapfile is not in fstab. If
  `swapon --show` is empty: `sudo fallocate -l 16G /opt1/swapfile && sudo chmod 600 /opt1/swapfile &&
  sudo mkswap /opt1/swapfile && sudo swapon /opt1/swapfile`. Conversion of a 10 GB checkpoint on 7.8 GB RAM
  needs it.
- **Python:** `/usr/bin/python3.13` with `PYTHONUSERBASE=/opt1/pyuser PIP_CACHE_DIR=/opt1/pipcache
  TMPDIR=/opt1/tmp`; `pip install --user --break-system-packages`. The default `python3` (pyenv 3.12)
  has its site-packages on the full root disk. `convert_hf_to_gguf.py` needs llama.cpp's `gguf-py`
  requirements; `f16_range.py` needs numpy only.
- **llama.cpp is built at `/opt1/llama.cpp`** (`b11699`, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=75`,
  checked 2026-10-27). There is no `~/llama.cpp`, which both scripts default to: pass `--llama /opt1/llama.cpp`
  and the same path as `ladder.sh`'s third argument. Rebuild with `-j2` (2 vCPU).
- `swapon` is `/sbin/swapon` (not on `PATH`). 2026-10-27: no swap active, `/` at 94% with 1.9 GB free.
- vLLM rigs share this GPU on port 8000 (`pgrep -af vllm`). Stop them before running; the graveyard
  uses port 8090.

## Layout

Scripts at the root, `Makefile` with `test` and `lint` (`ruff check` on an explicit file list: add each
new module), runs under `runs/<date>-<name>/` with a `REPORT.md`; `runs/*/` is gitignored, so commit a
report with `git add -f`. `ladder.sh` skips any GGUF that already exists, so delete a partial file before
rebuilding. `graveyard.py run` launches its own `llama-server` (`-ngl 99 --seed 0`); stop any other
`llama-server` on the GPU first. Python: the machine's own `python3`, no virtualenvs. GGUF files live
outside the repo (`*.gguf` is ignored).
