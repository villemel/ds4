import re
exec(open("sim2level.py").read().split("simulate(lambda: LRU(D), 0")[0])   # reuse the classes/streams
class SplitHost:
    """host: a cold-tier LRU quota (qc slots) + a general LRU for the rest (exclusive, promotion removes)"""
    def __init__(self, cap, qc): self.cold = HostLRU(qc, 0); self.gen = HostLRU(cap - qc, 0)
    def take(self, k): return self.cold.take(k) or self.gen.take(k)
    def insert(self, k, cold): (self.cold if cold else self.gen).insert(k, False)
def simulate2(Dn, make_host, label):
    tot = dict(dev=0, host=0, disk=0, cold=0, tok=0)
    for s in streams:
        dev = LRU(Dn); host = make_host()
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
    print(f"{label:40s} device {tot['dev']/n:6.0f}  host {tot['host']/n:5.0f}  disk {tot['disk']/n:5.0f}  (Samsung {tot['cold']/n:4.0f})")
simulate2(1558, lambda: HostLRU(2202, 1), "D 1558, H 2202 (now)")
simulate2(1558, lambda: HostLRU(2450, 1), "D 1558, H 2450 (40 GiB)")
simulate2(1558, lambda: HostLRU(3000, 1), "D 1558, H 3000")
simulate2(1672, lambda: HostLRU(2202, 1), "D 1672 (reserve 2 GB), H 2202")
simulate2(1558, lambda: SplitHost(2202, 300), "D 1558, H 2202 split: cold quota 300")
simulate2(1558, lambda: SplitHost(2202, 600), "D 1558, H 2202 split: cold quota 600")
simulate2(1558, lambda: SplitHost(2202, 1000), "D 1558, H 2202 split: cold quota 1000")
simulate2(1558, lambda: SplitHost(2202, 1500), "D 1558, H 2202 split: cold quota 1500")
# how many distinct cold-tier experts does a conversation touch, and how are cold reads distributed per layer?
import collections
cnt = collections.Counter(); per_layer = collections.Counter(); tokn = 0
for s in streams:
    for layers in s:
        tokn += 1
        for l, ex in layers:
            c = sum(1 for e in ex if (l, e) not in primary); per_layer[c] += 1
            for e in ex:
                if (l, e) not in primary: cnt[(l, e)] += 1
print("distinct cold-tier experts touched:", len(cnt), "of", sum(cnt.values()), "cold requests;", "used once:", sum(1 for v in cnt.values() if v == 1))
print("cold-tier experts per layer-call distribution:", sorted(per_layer.items())[:8], "mean", sum(k*v for k, v in per_layer.items())/sum(per_layer.values()))
