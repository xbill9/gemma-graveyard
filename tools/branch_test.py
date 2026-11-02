#!/usr/bin/env python3
"""branch_test.py <rung> <gguf> <out.jsonl> [--llama DIR] [--threads N]: the causal test. For every pair of cards whose answers to one
question differ in grade at this rung, take the two recorded texts up to their first difference, add each branch's next 12
characters, and let one llama-server on one machine finish both greedily. If the graded result follows the forced text, the
fork decides the answer and the card only picks the branch. Same sampling as graveyard.py run; the grader is lazy_eval.py's."""
import argparse
import itertools
import json
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, ".")
from lazy_eval import extract, right

D = {"l4": "runs/2026-10-28-l4", "1650": "runs/2026-10-29-1650", "3050": "runs/2026-10-29-3050"}
PORT = 8092


def rows(g, r):
    with open(f"{D[g]}/{r}.jsonl") as f:
        return {x["id"]: x for x in map(json.loads, f)}


def fork(a, b):
    n = min(len(a), len(b))
    return next((k for k in range(n) if a[k] != b[k]), n)


def done(prompt, prefix):
    body = {"prompt": prompt + prefix, "n_predict": 1024, "temperature": 0, "cache_prompt": False, "special": False}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rung")
    p.add_argument("gguf")
    p.add_argument("out")
    p.add_argument("--llama", default="llama.cpp")
    p.add_argument("--threads", default="10")
    a = p.parse_args()
    with open("runs/2026-11-01-flips/prompts.json") as f:
        prompts = json.load(f)
    g = {k: rows(k, a.rung) for k in D}
    seen = set()
    srv = subprocess.Popen([f"{a.llama}/build/bin/llama-server", "-m", a.gguf, "-c", "4096", "-np", "1", "--port", str(PORT),
                            "--seed", "0", "-t", a.threads], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
                break
            except OSError:
                time.sleep(1)
        with open(a.out, "a") as f:
            for x, y in itertools.combinations(g, 2):
                for i in g[x]:
                    u, v = g[x][i], g[y][i]
                    if u["right"] == v["right"]:
                        continue
                    s, t = (u, v) if u["right"] else (v, u)  # s is the right branch
                    k = fork(s["text"], t["text"])
                    key = (i, s["text"][:k + 12], t["text"][:k + 12])
                    if key in seen:
                        continue
                    seen.add(key)
                    res = {}
                    for name, text in (("right", s["text"]), ("wrong", t["text"])):
                        pre = text[:k + 12]
                        c = done(prompts[i]["p"], pre)
                        full = pre + c["content"]
                        res[name] = {"extracted": extract(full), "right": right(full, s["answer"]), "stop": c.get("stop_type")}
                    f.write(json.dumps({"rung": a.rung, "id": i, "pair": [x, y], "fork": k, "right_branch": s["text"][k:k + 12],
                                        "wrong_branch": t["text"][k:k + 12], "gold": s["answer"], "recorded_right": s["extracted"],
                                        "recorded_wrong": t["extracted"], **res}) + "\n")
                    f.flush()
    finally:
        srv.terminate()
        srv.wait(timeout=30)


if __name__ == "__main__":
    main()
