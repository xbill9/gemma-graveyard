#!/usr/bin/env python3
"""replay_input.py <prompts.json> <rows.jsonl> <tag> <limit> > in.tsv: one replay.cpp input line per recorded answer."""
import json
import sys

rows, tag, limit = sys.argv[2], sys.argv[3], int(sys.argv[4])
with open(sys.argv[1]) as f:
    prompts = json.load(f)
with open(rows) as f:
    lines = f.readlines()
for k, line in enumerate(lines):
    x = json.loads(line)
    if k < limit and x.get("text"):
        print(f"{tag}:{x['id']}\t{prompts[x['id']]['p'].encode().hex()}\t{x['text'].encode().hex()}")
