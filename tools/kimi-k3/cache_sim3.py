# Device LRU (all experts) + host LRU holding cold-tier experts only; reports disk reads by tier. Also Belady's OPT
# for the device level as the ceiling. Decode stream of the eight prompts, prompt passes as transient inserts.
import re, sys
from collections import OrderedDict
D, H = 1584, 2202
def names(path):
    out = set()
    for line in open(path):
        line = line.strip()
        if line: l, e = line[:-4].split("-"); out.add((int(l[1:]), int(e[1:])))
    return out
cold = names("<path>") - names("<path>")
reqs = []
for path in sys.argv[1:]:
    n_prompt = None; rows = []
    for line in open(path, errors="replace"):
        m = re.match(r"ds4: K3 prefill (\d+) tokens in", line)
        if m: n_prompt = int(m[1])
        m = re.match(r"route layer (\d+) pos (\d+):(.*)", line)
        if m: rows.append((int(m[2]), int(m[1]), [int(x.split(":")[0]) for x in m[3].split()]))
    rows.sort()
    reqs.append(([(l, e) for pos, l, ex in rows if pos < (n_prompt or 0) for e in ex],
                 [(l, e) for pos, l, ex in rows if pos >= (n_prompt or 0) for e in ex]))
n_dec = sum(len(d) for _, d in reqs)
def run(host_policy):
    dev = OrderedDict(); host = OrderedDict()
    hd = hh = 0; disk_hot = disk_cold = 0
    def ins(c, cap, k, hot):
        if k in c: c.move_to_end(k, last=hot); return
        c[k] = 1
        if not hot: c.move_to_end(k, last=False)
        if len(c) > cap: c.popitem(last=False)
    for pre, dec in reqs:
        for k in pre:
            if k not in dev: ins(dev, D, k, False)
        for k in dec:
            if k in dev:
                hd += 1; dev.move_to_end(k)
                if host_policy == "cold-only" and k in host: host.move_to_end(k)
                continue
            if k in host:
                hh += 1; host.move_to_end(k)
            else:
                if k in cold: disk_cold += 1
                else: disk_hot += 1
                if host_policy == "all" or (host_policy == "cold-only" and k in cold): ins(host, H, k, True)
            ins(dev, D, k, True)
    print("  host=%-9s device %.1f %%, host %.1f %%, disk hot-tier %.1f %%, disk cold-tier %.1f %%  (per layer: %.2f hot, %.2f cold disk reads)" %
          (host_policy, 100.0 * hd / n_dec, 100.0 * hh / n_dec, 100.0 * disk_hot / n_dec, 100.0 * disk_cold / n_dec,
           16.0 * disk_hot / n_dec, 16.0 * disk_cold / n_dec))
run("all"); run("cold-only")
# Belady OPT for the device level alone (decode stream only, no prompt transients): the ceiling for any policy
stream = [k for _, dec in reqs for k in dec]
nxt = {}; next_use = [0] * len(stream)
for i in range(len(stream) - 1, -1, -1):
    k = stream[i]; next_use[i] = nxt.get(k, 1 << 60); nxt[k] = i
import heapq
cache = {}; heap = []; hits = 0
for i, k in enumerate(stream):
    if k in cache:
        hits += 1; cache[k] = next_use[i]; heapq.heappush(heap, (-next_use[i], k)); continue
    if len(cache) >= D:
        while heap:
            nu, kk = heapq.heappop(heap)
            if kk in cache and cache[kk] == -nu: del cache[kk]; break
    cache[k] = next_use[i]; heapq.heappush(heap, (-next_use[i], k))
print("  Belady OPT at %d device slabs: %.1f %% hits (LRU gives 36.4 %%)" % (D, 100.0 * hits / len(stream)))
