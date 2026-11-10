import re
exec(open("sim2level.py").read().split("simulate(lambda: LRU(D), 0")[0])
class LRUh(LRU):
    def __init__(self, cap): super().__init__(cap); self.hits = {}
    def hit(self, k):
        if k in self.od: self.od.move_to_end(k); self.hits[k] = self.hits.get(k, 0) + 1; return True
        return False
    def insert(self, k):
        ev = None
        if len(self.od) >= self.cap: ev = self.od.popitem(last=False)[0]
        self.od[k] = 1; self.hits[k] = 0; return ev
def run(rule, label):
    tot = dict(dev=0, host=0, disk=0, cold=0, cd=0, tok=0)
    for s in streams:
        dev = LRUh(D); host = HostLRU(H, 1)
        for layers in s:
            for l, ex in layers:
                for e in ex:
                    k = (l, e); cold = k not in primary
                    if dev.hit(k): tot["dev"] += 1; continue
                    if host.take(k): tot["host"] += 1
                    else:
                        tot["disk"] += 1
                        if cold: tot["cold"] += 1
                    ev = dev.insert(k)
                    if ev is not None:
                        h = dev.hits.pop(ev, 0)
                        if rule(ev, h, ev not in primary): host.insert(ev, ev not in primary); tot["cd"] += 1
            tot["tok"] += 1
    n = tot["tok"]
    print(f"{label:44s} device {tot['dev']/n:5.0f} host {tot['host']/n:5.0f} disk {tot['disk']/n:5.0f} Samsung {tot['cold']/n:4.0f} copy-downs {tot['cd']/n:5.0f}")
run(lambda k, h, c: True, "copy down everything (now)")
run(lambda k, h, c: h >= 1 or c, "copy down if hit while in device, or cold-tier")
run(lambda k, h, c: h >= 1, "copy down only if hit while in device")
run(lambda k, h, c: c, "copy down cold-tier only")
