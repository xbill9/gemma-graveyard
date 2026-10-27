#!/usr/bin/env python3
"""graveyard.py -- which GSM8K answers each quantization rung of Gemma 4 E2B loses, scored in code.

  graveyard.py run <rung> <model.gguf> <data.jsonl> <out-dir> [--llama DIR] [--port P] [--limit N]
                   [--think] [--max-tokens M] [--ctx C]
  graveyard.py bury <out-dir> <rung> [<rung> ...]

run starts a llama-server on one GGUF, asks every question once (greedy, one request at a time,
no prompt cache) and writes <out-dir>/<rung>.jsonl plus <rung>.meta.json (file size, load time,
GPU memory after load, server command). bury reads the rungs in the order given; the first one is
the reference. A question dies at the first rung that gets it wrong after the reference got it right;
it is a ghost at any rung that gets it right after the reference got it wrong. Every count is
computed here, never by a model.

Prompt, answer extraction and grading are lazy_eval.py's (copied from the Week 3 repo, which copied
them from gemma4-dev/jev-tpu-v5e1/gen_eval.py).
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request

from lazy_eval import extract, load, post, prompt, right


def gpu_used_mib():
    if not shutil.which("nvidia-smi"):
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False
    )
    return int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout.strip() else None


def wait_healthy(url, proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError("llama-server not healthy in time")


def ask(upstream, rec, a):
    body = {
        "temperature": 0.0,
        "max_tokens": a.max_tokens,
        "messages": [{"role": "user", "content": prompt(rec["question"])}],
        "chat_template_kwargs": {"enable_thinking": a.think},
        "cache_prompt": False,
    }
    out = {"id": rec["id"], "answer": rec["answer"]}
    t0 = time.time()
    try:
        r = post(upstream, body)
    except (OSError, ValueError) as e:  # recorded, scored wrong, counted separately
        out.update(error=f"{type(e).__name__}: {e}"[:300], right=False, seconds=round(time.time() - t0, 2))
        return out
    msg = r["choices"][0]["message"]
    t = r.get("timings") or {}
    out.update(
        text=msg.get("content"),
        reasoning=msg.get("reasoning_content"),
        finish=r["choices"][0].get("finish_reason"),
        completion_tokens=r["usage"]["completion_tokens"],
        prompt_tokens=r["usage"]["prompt_tokens"],
        decode_tps=t.get("predicted_per_second"),
        prefill_tps=t.get("prompt_per_second"),
        seconds=round(time.time() - t0, 2),
        extracted=extract(msg.get("content")),
        right=right(msg.get("content"), rec["answer"]),
    )
    return out


def run(a):
    os.makedirs(a.out_dir, exist_ok=True)
    recs = load(a.data)[: a.limit] if a.limit else load(a.data)
    cmd = [
        os.path.join(a.llama, "build", "bin", "llama-server"),
        "-m", a.model, "-ngl", "99", "-c", str(a.ctx), "-np", "1", "--port", str(a.port), "--seed", "0",
    ]  # fmt: skip
    url = f"http://127.0.0.1:{a.port}"
    before = gpu_used_mib()
    with open(os.path.join(a.out_dir, f"{a.rung}.server.log"), "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            load_s = wait_healthy(url, proc)
            meta = {
                "rung": a.rung,
                "model": os.path.abspath(a.model),
                "bytes": os.path.getsize(a.model),
                "load_seconds": round(load_s, 1),
                "gpu_mib_before": before,
                "gpu_mib_loaded": gpu_used_mib(),
                "think": a.think,
                "max_tokens": a.max_tokens,
                "cmd": cmd,
                "n": len(recs),
            }
            rows = []
            with open(os.path.join(a.out_dir, f"{a.rung}.jsonl"), "w") as f:  # written as it goes
                for rec in recs:
                    row = ask(url, rec, a)
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
            meta["gpu_mib_after"] = gpu_used_mib()
        finally:
            proc.terminate()
            proc.wait(timeout=60)
    with open(os.path.join(a.out_dir, f"{a.rung}.meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({"rung": a.rung, **rung_summary(rows)}))


def rung_summary(rows):
    ok = [r for r in rows if "error" not in r]
    tps = [r["decode_tps"] for r in ok if r.get("decode_tps")]
    return {
        "n": len(rows),
        "right": sum(r["right"] for r in rows),
        "errors": len(rows) - len(ok),
        "truncated": sum(r.get("finish") == "length" for r in ok),
        "median_decode_tps": round(statistics.median(tps), 1) if tps else None,
        "mean_tokens": round(statistics.mean(r["completion_tokens"] for r in ok), 1) if ok else None,
    }


def bury_rows(by_rung, rungs):
    """by_rung: {rung: {id: row}}. Returns per-rung table and per-question tombstones/ghosts."""
    ref = by_rung[rungs[0]]
    ids = sorted(ref)
    for r in rungs[1:]:
        missing = set(ids) - set(by_rung[r])
        if missing:
            raise ValueError(f"{r} is missing {len(missing)} question(s) the reference has, e.g. {min(missing)}")
    table, tombstones, ghosts = [], [], []
    for r in rungs:
        rows = by_rung[r]
        deaths = [i for i in ids if ref[i]["right"] and not rows[i]["right"]]
        risen = [i for i in ids if not ref[i]["right"] and rows[i]["right"]]
        changed = [i for i in ids if rows[i].get("extracted") != ref[i].get("extracted")]
        table.append(
            {
                "rung": r,
                **rung_summary([rows[i] for i in ids]),
                "dead": len(deaths),
                "ghosts": len(risen),
                "answer_changed": len(changed),
            }
        )
    for i in ids:
        if ref[i]["right"]:
            died = next((r for r in rungs[1:] if not by_rung[r][i]["right"]), None)
            if died:
                after = [r for r in rungs[rungs.index(died) + 1 :] if by_rung[r][i]["right"]]
                tombstones.append(
                    {
                        "id": i,
                        "died_at": died,
                        "answer": ref[i]["answer"],
                        "said": by_rung[died][i].get("extracted"),
                        "right_again_at": after,
                    }
                )
        else:
            back = [r for r in rungs[1:] if by_rung[r][i]["right"]]
            if back:
                ghosts.append({"id": i, "reference_said": ref[i].get("extracted"), "right_at": back})
    return {"reference": rungs[0], "table": table, "tombstones": tombstones, "ghosts": ghosts}


def bury(a):
    by_rung = {r: {row["id"]: row for row in load(os.path.join(a.out_dir, f"{r}.jsonl"))} for r in a.rungs}
    g = bury_rows(by_rung, a.rungs)
    for r in g["table"]:
        meta = os.path.join(a.out_dir, f"{r['rung']}.meta.json")
        if os.path.exists(meta):
            with open(meta) as f:
                m = json.load(f)
            r.update(bytes=m["bytes"], gpu_mib_loaded=m.get("gpu_mib_loaded"), gpu_mib_before=m.get("gpu_mib_before"))
    with open(os.path.join(a.out_dir, "graveyard.json"), "w") as f:
        json.dump(g, f, indent=1)
    for r in g["table"]:
        print(json.dumps(r))
    deaths = {}
    for t in g["tombstones"]:
        deaths[t["died_at"]] = deaths.get(t["died_at"], 0) + 1
    print(json.dumps({"first_deaths_by_rung": deaths, "tombstones": len(g["tombstones"]), "ghosts": len(g["ghosts"])}))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("rung")
    r.add_argument("model")
    r.add_argument("data")
    r.add_argument("out_dir")
    r.add_argument("--llama", default=os.path.expanduser("~/llama.cpp"))
    r.add_argument("--port", type=int, default=8090)
    r.add_argument("--limit", type=int)
    r.add_argument("--think", action="store_true")
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--ctx", type=int, default=4096)
    b = sub.add_parser("bury")
    b.add_argument("out_dir")
    b.add_argument("rungs", nargs="+")
    a = p.parse_args()
    run(a) if a.cmd == "run" else bury(a)


if __name__ == "__main__":
    sys.exit(main())
