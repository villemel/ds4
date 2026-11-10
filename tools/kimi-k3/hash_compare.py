#!/usr/bin/env python3
"""Bit-level comparison of two runs from their DS4_K3_TRACE_HASH lines: logits per position and, when logged, attention result, routed output and hidden state per layer."""
import re, sys
def load(f):
    s = open(f, errors="replace").read()
    L = {int(a): b for a, b in re.findall(r"K3 hash logits pos (\d+) ([0-9a-f]{16})", s)}
    H = {(w, int(p), int(i)): h for w, p, i, h in re.findall(r"K3 hash layer (\w+) pos (\d+) il (\d+) ([0-9a-f]{16})", s)}
    return L, H
A, HA = load(sys.argv[1])
for f in sys.argv[2:]:
    B, HB = load(f)
    common = sorted(set(A) & set(B)); diff = [p for p in common if A[p] != B[p]]
    print(f"{f.split('/')[-1]}: logits at {len(common)} common positions, {len(diff)} differ" + (f", first at position {diff[0]}" if diff else ": every logit vector bit-identical"))
    keys = sorted(set(HA) & set(HB), key=lambda k: (k[1], k[2], {"prefix": 0, "routed": 1, "cur": 2}[k[0]]))
    if keys:
        bad = [k for k in keys if HA[k] != HB[k]]
        print(f"  layer hashes: {len(keys)} common, {len(bad)} differ" + (f"; first {bad[:6]}" if bad else ": every layer bit-identical"))
        if bad:
            from collections import Counter
            print("  differing by kind:", Counter(k[0] for k in bad), "by position:", sorted(Counter(k[1] for k in bad).items())[:10])
