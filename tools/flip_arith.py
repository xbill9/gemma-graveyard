#!/usr/bin/env python3
"""flip_arith.py: every plain 'a op b = c' in the recorded answers, computed. Chained, percent and rounded-on-purpose
expressions are skipped; claims that stay false after a 0.6% tolerance are counted per rung, three cards."""
import json
import re

D={'l4':'runs/2026-10-28-l4','1650':'runs/2026-10-29-1650','3050':'runs/2026-10-29-3050'}
R=["BF16","F16","Q8_0","Q6_K","Q5_K_M","Q4_K_M","Q4_0","Q3_K_M","Q2_K"]
N=r'\\?\$?(-?\d[\d,]*(?:\.\d+)?)\\?%?'
EQ=re.compile(r'(?:^|(?<=[\s$(:]))'+N+r'\s*(\+|-|\\times|×|\*|/|\\div)\s*'+N+r'\s*=\s*'+N+r'(?=[\s$.,;)\\]|$)',re.MULTILINE)
def num(s): return float(s.replace(',',''))
def check(t):
    good=bad=0;ex=[]
    for line in (t or '').split('\n'):
      for m in EQ.finditer(line):
        pre=line[:m.start()].rstrip()
        if pre[-1:] in '+-*/×=' or pre.endswith(('\\times', '\\div', ')')) : continue
        post=line[m.end():].lstrip()
        if '%' in m[0] or post[:6].startswith(('\\times','\\div','\\text')) or post[:1] in ('+','-','*','/','×','(','\\'): continue
        if m[4].endswith('.') : continue
        a,op,b,c=num(m[1]),m[2],num(m[3]),num(m[4])
        v=a+b if op=='+' else a-b if op=='-' else a*b if op in('\\times','×','*') else (a/b if b else None)
        if v is None: continue
        if abs(v-c)>max(0.0051,abs(v)*0.006): bad+=1;ex.append(m[0])
        else: good+=1
    return good,bad,ex
print('rung, checked claims, false, per 1000 (3 cards), answers with any false: right/wrong')
for r in R:
    g=b=0;ar=aw=nr=nw=0;ex=[]
    for d in D.values():
        with open(f'{d}/{r}.jsonl') as fh:
            lines = fh.readlines()
        for l in lines:
            x=json.loads(l);G,B,E=check(x.get('text'));g+=G;b+=B;ex+=E[:1]
            if x['right']: nr+=1;ar+=B>0
            else: nw+=1;aw+=B>0
    print(r,g+b,b,round(1000*b/max(g+b,1),2),f'{ar}/{nr}',f'{aw}/{nw}',ex[:6])
