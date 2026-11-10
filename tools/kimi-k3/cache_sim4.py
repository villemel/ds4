# Device-level policies at D slabs over the decode stream (heap-based): LRU, LRU-2, decayed-frequency (score kept
# undecayed: s += base^tok, eviction = min s via a lazy heap; renormalised in the log domain), + heat prior variant.
import re, sys, math, heapq
D = 1584
reqs = []
for path in sys.argv[1:]:
    n_prompt = None; rows = []
    for line in open(path, errors="replace"):
        m = re.match(r"ds4: K3 prefill (\d+) tokens in", line)
        if m: n_prompt = int(m[1])
        m = re.match(r"route layer (\d+) pos (\d+):(.*)", line)
        if m: rows.append((int(m[2]), int(m[1]), [int(x.split(":")[0]) for x in m[3].split()]))
    rows.sort()
    reqs.append([(pos, l, e) for pos, l, ex in rows if pos >= (n_prompt or 0) for e in ex])
stream = []; t = 0
for r in reqs:
    last = None
    for pos, l, e in r:
        if pos != last: t += 1; last = pos
        stream.append((t, (l, e)))
n = len(stream)
freq = {}
for _, k in stream: freq[k] = freq.get(k, 0) + 1
def run(score_fn):
    """score_fn(k, tok, state) -> score (higher = keep); eviction removes the current minimum"""
    cache = {}; heap = []; hits = 0; state = {}
    for i, (tok, k) in enumerate(stream):
        sc = score_fn(k, tok, state)
        if k in cache:
            hits += 1; cache[k] = sc; heapq.heappush(heap, (sc, k)); continue
        if len(cache) >= D:
            while heap:
                sc0, k0 = heapq.heappop(heap)
                if k0 in cache and cache[k0] == sc0: del cache[k0]; break
        cache[k] = sc; heapq.heappush(heap, (sc, k))
    return 100.0 * hits / n
def lru_score(k, tok, st): st[k] = st.get(k, 0) + 1; return (tok, st[k])
def lru2_score(k, tok, st):
    h = st.setdefault(k, []); h.append(tok)
    if len(h) > 2: del h[0]
    return (h[0] if len(h) == 2 else -1, h[-1])
def decayed_score(decay, prior=0.0):
    lb = -math.log(decay)          # score in log domain: log(sum decay^(tok_now - t_use)) + tok_now*lb (monotone shift)
    def f(k, tok, st):
        ls = st.get(k)              # log of undecayed sum: sum_i exp(t_i * lb)
        add = tok * lb
        ls = add if ls is None else max(ls, add) + math.log1p(math.exp(-abs(ls - add)))
        st[k] = ls
        return ls + prior * math.log(1 + freq.get(k, 0))
    return f
print("stream %d routings, %d tokens; D = %d" % (n, t, D))
print("  LRU        %.1f %%" % run(lru_score))
print("  LRU-2      %.1f %%" % run(lru2_score))
for d in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
    print("  decay %.2f %.1f %%" % (d, run(decayed_score(d))))
print("  decay 0.9 + 0.3 log heat  %.1f %%" % run(decayed_score(0.9, 0.3)))
print("  decay 0.9 + 1.0 log heat  %.1f %%" % run(decayed_score(0.9, 1.0)))
