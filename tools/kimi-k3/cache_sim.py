# What would an expert cache in VRAM (+ pinned RAM) serve? Static top-K by frequency (what an LFU converges to) and an
# online LRU over the actual token stream, per decode token (prefill routings excluded), from DS4_K3_TRACE_ROUTES logs.
import re, sys
from collections import OrderedDict
rows_all = []
for path in sys.argv[1:]:
    n_prompt = None; rows = []
    for line in open(path, errors="replace"):
        m = re.match(r"ds4: K3 prefill (\d+) tokens in", line)
        if m: n_prompt = int(m[1])
        m = re.match(r"route layer (\d+) pos (\d+):(.*)", line)
        if m: rows.append((int(m[2]), int(m[1]), [int(x.split(":")[0]) for x in m[3].split()]))
    rows_all.append((n_prompt or 0, rows))
dec = {}; stream = []   # stream: list of (layer, expert) per decode token-layer in order
for n_prompt, rows in rows_all:
    for pos, layer, ex in sorted(rows):
        if pos < n_prompt: continue
        for e in ex:
            k = (layer, e); dec[k] = dec.get(k, 0) + 1; stream.append(k)
tot = len(stream); n_tok = tot // (92 * 16)
print("decode routings %d (%d tokens), distinct experts %d of 82432" % (tot, n_tok, len(dec)))
ranked = sorted(dec.items(), key=lambda kv: -kv[1])
for K in (500, 1000, 1500, 2000, 3000, 3700, 5000, 8000):
    print("  static top-%5d by frequency: %.1f %% of reads (%.1f GB of cache)" % (K, 100.0 * sum(v for _, v in ranked[:K]) / tot, K * 0.0175))
for K in (1500, 2000, 3700, 5000):
    lru = OrderedDict(); hits = 0
    for k in stream:
        if k in lru: hits += 1; lru.move_to_end(k)
        else:
            lru[k] = 1
            if len(lru) > K: lru.popitem(last=False)
    print("  online LRU %5d: %.1f %% hits over the stream" % (K, 100.0 * hits / tot))
# LFU-with-aging: keep counts, evict the lowest count; admission on every miss
for K in (1500, 3700):
    cnt = {}; cache = {}; hits = 0
    for i, k in enumerate(stream):
        cnt[k] = cnt.get(k, 0) + 1
        if k in cache: hits += 1
        else:
            if len(cache) >= K:
                victim = min(cache, key=lambda x: cnt[x])
                if cnt[victim] < cnt[k]: del cache[victim]; cache[k] = 1
            else: cache[k] = 1
    print("  online LFU %5d: %.1f %% hits" % (K, 100.0 * hits / tot))
