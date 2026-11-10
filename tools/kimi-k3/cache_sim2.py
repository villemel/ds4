# Two-level cache simulation over the decode stream of DS4_K3_TRACE_ROUTES logs (prompt routings as transient inserts
# between requests): device LRU D + host LRU H, with/without refreshing the host copy on a device hit, and S3-FIFO-like
# admission (an expert enters the main device cache only on its second request within a probation window).
import re, sys
from collections import OrderedDict, deque
D, H = 1584, 2202
reqs = []
for path in sys.argv[1:]:
    n_prompt = None; rows = []
    for line in open(path, errors="replace"):
        m = re.match(r"ds4: K3 prefill (\d+) tokens in", line)
        if m: n_prompt = int(m[1])
        m = re.match(r"route layer (\d+) pos (\d+):(.*)", line)
        if m: rows.append((int(m[2]), int(m[1]), [int(x.split(":")[0]) for x in m[3].split()]))
    rows.sort()
    pre = [(l, e) for pos, l, ex in rows if pos < (n_prompt or 0) for e in ex]
    dec = [(l, e) for pos, l, ex in rows if pos >= (n_prompt or 0) for e in ex]
    reqs.append((pre, dec))
def run(touch_host, admission, transient_prefill=True):
    dev = OrderedDict(); host = OrderedDict(); seen = OrderedDict()   # seen: probation (keys recently missed)
    hd = hh = miss = 0; n = 0
    def host_insert(k, hot):
        if k in host: host.move_to_end(k, last=hot); return
        host[k] = 1
        if not hot: host.move_to_end(k, last=False)
        if len(host) > H: host.popitem(last=False)
    def dev_insert(k, hot):
        if k in dev: dev.move_to_end(k, last=hot); return
        dev[k] = 1
        if not hot: dev.move_to_end(k, last=False)
        if len(dev) > D: dev.popitem(last=False)
    for pre, dec in reqs:
        if transient_prefill:
            # prompt pass: a transient insert evicts the coldest entry (which is another transient once the pass is under way)
            for k in pre:
                if k in dev: continue
                if k not in host: host_insert(k, False)
                dev_insert(k, False)
        for k in dec:
            n += 1
            if k in dev:
                hd += 1; dev.move_to_end(k)
                if touch_host and k in host: host.move_to_end(k)
                continue
            if k in host:
                hh += 1; host.move_to_end(k)
            else:
                miss += 1
                if admission == "second-touch" and k not in seen:
                    seen[k] = 1
                    if len(seen) > 8000: seen.popitem(last=False)
                    host_insert(k, True)
                    continue   # not admitted to the device cache yet
                host_insert(k, True)
            dev_insert(k, True)
    print("  %-34s device %.1f %%, host %.1f %%, disk %.1f %%" % ("touch_host=%s admission=%s prefill=%s" % (touch_host, admission, transient_prefill), 100.0 * hd / n, 100.0 * hh / n, 100.0 * miss / n))
print("stream: %d requests, %d decode routings" % (len(reqs), sum(len(d) for _, d in reqs)))
run(False, "lru"); run(True, "lru"); run(True, "lru", False); run(True, "second-touch")
