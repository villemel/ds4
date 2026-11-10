# Swap evaluation with up to 18 chat logs: rank on the first K logs, evaluate on the rest (held out), for several N.
import re, collections, sys, glob
from collections import OrderedDict
D, H = 1558, 2202
primary = set()
for line in open("primary.lst"):
    m = re.match(r"L(\d+)-E(\d+)\.bin", line.strip())
    if m: primary.add((int(m[1]), int(m[2])))
logs = sorted(glob.glob("r*.log"), key=lambda p: int(re.search(r"r(\d+)\.log", p)[1]))
streams = []
for path in logs:
    n_prompt = 0; rows = []
    for line in open(path):
        m = re.search(r"K3 prefill (\d+) tokens in", line)
        if m: n_prompt = int(m[1])
        m = re.match(r"route layer (\d+) pos (\d+):(.*)", line)
        if m: rows.append((int(m[2]), int(m[1]), [int(x.split(":")[0]) for x in m[3].split()]))
    rows.sort(); toks = {}
    for pos, l, ex in rows:
        if pos >= n_prompt: toks.setdefault(pos, []).append((l, ex))
    streams.append([toks[p] for p in sorted(toks)])
print("logs:", len(streams), "tokens:", [len(s) for s in streams])
class LRU:
    def __init__(self, cap): self.cap = cap; self.od = OrderedDict()
    def hit(self, k):
        if k in self.od: self.od.move_to_end(k); return True
        return False
    def insert(self, k):
        ev = None
        if len(self.od) >= self.cap: ev = self.od.popitem(last=False)[0]
        self.od[k] = 1; return ev
class HostLRU:
    def __init__(self, cap): self.cap = cap; self.od = OrderedDict()
    def take(self, k):
        if k in self.od: del self.od[k]; return True
        return False
    def insert(self, k):
        if k in self.od: return
        while len(self.od) >= self.cap: self.od.popitem(last=False)
        self.od[k] = 1
def freq_of(idx):
    c = collections.Counter()
    for i in idx:
        for layers in streams[i]:
            for l, ex in layers:
                for e in ex: c[(l, e)] += 1
    return c
def run(prim_set, idx):
    tot = dict(disk=0, cold=0, tok=0)
    for i in idx:
        dev = LRU(D); host = HostLRU(H)
        for layers in streams[i]:
            for l, ex in layers:
                for e in ex:
                    k = (l, e)
                    if dev.hit(k): continue
                    if not host.take(k):
                        tot["disk"] += 1
                        if k not in prim_set: tot["cold"] += 1
                    ev = dev.insert(k)
                    if ev is not None: host.insert(ev)
            tot["tok"] += 1
    return tot["disk"] / tot["tok"], tot["cold"] / tot["tok"]
K = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, len(streams) - 4)
train, test = list(range(K)), list(range(K, len(streams)))
rank = freq_of(train)
cold_sorted = sorted([k for k in rank if k not in primary], key=lambda k: (-rank[k], k))
prim_sorted = sorted(primary, key=lambda k: (rank.get(k, 0), k))
print(f"rank on logs 1-{K}, evaluate on {K+1}-{len(streams)}")
for N in (0, 2000, 5000, 7500, 10000, 15000):
    newprim = set(primary)
    for k in prim_sorted[:N]: newprim.discard(k)
    for k in cold_sorted[:N]: newprim.add(k)
    d, c = run(newprim, test)
    print(f"swap {N:5d}: held-out disk {d:5.0f}, Samsung {c:5.1f} per token")
