# Two-level expert cache simulation on the real chat decode streams: device D slabs (policy under test) + host H slabs
# (exclusive victim LRU, cold-tier second chance). Reports per-token device/host/disk and Samsung-tier disk reads.
import re, sys
from collections import OrderedDict
D, H = 1558, 2202
primary = set()
for line in open("primary.lst"):
    m = re.match(r"L(\d+)-E(\d+)\.bin", line.strip())
    if m: primary.add((int(m[1]), int(m[2])))
streams = []
for i in range(1, 9):
    n_prompt = 0; rows = []
    for line in open(f"r{i}.log"):
        m = re.search(r"K3 prefill (\d+) tokens in", line)
        if m: n_prompt = int(m[1])
        m = re.match(r"route layer (\d+) pos (\d+):(.*)", line)
        if m: rows.append((int(m[2]), int(m[1]), [int(x.split(":")[0]) for x in m[3].split()]))
    rows.sort()
    toks = {}
    for pos, l, ex in rows:
        if pos >= n_prompt: toks.setdefault(pos, []).append((l, ex))
    streams.append([toks[p] for p in sorted(toks)])
print("streams:", [len(s) for s in streams], "tokens; primary files", len(primary))

class LRU:
    def __init__(self, cap): self.cap = cap; self.od = OrderedDict()
    def hit(self, k):
        if k in self.od: self.od.move_to_end(k); return True
        return False
    def insert(self, k):   # returns evicted key or None
        ev = None
        if len(self.od) >= self.cap: ev = self.od.popitem(last=False)[0]
        self.od[k] = 1; return ev
    def remove(self, k): self.od.pop(k, None)
    def __contains__(self, k): return k in self.od

class SLRU:
    """segmented LRU: new entries enter probation; a hit in probation promotes to protected (cap prot_cap);
       overflow of protected demotes its LRU to probation's MRU; eviction from probation's LRU."""
    def __init__(self, cap, prot_frac):
        self.cap = cap; self.pc = int(cap * prot_frac); self.prob = OrderedDict(); self.prot = OrderedDict()
    def hit(self, k):
        if k in self.prot: self.prot.move_to_end(k); return True
        if k in self.prob:
            del self.prob[k]; self.prot[k] = 1
            if len(self.prot) > self.pc:
                d = self.prot.popitem(last=False)[0]; self.prob[d] = 1
            return True
        return False
    def insert(self, k):
        ev = None
        if len(self.prob) + len(self.prot) >= self.cap:
            if self.prob: ev = self.prob.popitem(last=False)[0]
            else: ev = self.prot.popitem(last=False)[0]
        self.prob[k] = 1; return ev
    def remove(self, k): self.prob.pop(k, None); self.prot.pop(k, None)
    def __contains__(self, k): return k in self.prob or k in self.prot

class HostLRU:
    """exclusive victim LRU with a second chance for cold-tier entries"""
    def __init__(self, cap, chance): self.cap = cap; self.od = OrderedDict(); self.ch = {}; self.chance = chance
    def take(self, k):   # host hit: promote (remove)
        if k in self.od: del self.od[k]; self.ch.pop(k, None); return True
        return False
    def insert(self, k, cold):
        if k in self.od: return
        while len(self.od) >= self.cap:
            v = next(iter(self.od))
            if self.ch.get(v, 0) > 0: self.ch[v] -= 1; self.od.move_to_end(v); continue
            del self.od[v]; self.ch.pop(v, None)
        self.od[k] = 1
        if cold and self.chance: self.ch[k] = self.chance

def simulate(make_dev, host_chance=1, label=""):
    tot = dict(dev=0, host=0, disk=0, cold=0, tok=0)
    for s in streams:
        dev = make_dev(); host = HostLRU(H, host_chance)
        for t, layers in enumerate(s):
            for l, ex in layers:
                for e in ex:
                    k = (l, e); cold = k not in primary
                    if dev.hit(k): tot["dev"] += 1; continue
                    if host.take(k): tot["host"] += 1
                    else:
                        tot["disk"] += 1
                        if cold: tot["cold"] += 1
                    ev = dev.insert(k)
                    if ev is not None: host.insert(ev, ev not in primary)
            tot["tok"] += 1
    n = tot["tok"]
    print(f"{label:34s} device {tot['dev']/n:6.0f}  host {tot['host']/n:5.0f}  disk {tot['disk']/n:5.0f}  (Samsung {tot['cold']/n:4.0f})  per token")

simulate(lambda: LRU(D), 0, "LRU, host chance 0")
simulate(lambda: LRU(D), 1, "LRU, host chance 1")
simulate(lambda: LRU(D), 2, "LRU, host chance 2")
for f in (0.5, 0.7, 0.8, 0.9):
    simulate(lambda f=f: SLRU(D, f), 1, f"SLRU protected {int(f*100)} %, host chance 1")
