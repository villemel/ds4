import re, sys
for path in sys.argv[1:]:
    hot = cold = 0; hot_ms = cold_ms = 0.0; n = 0; waits = []; toks = []
    for line in open(path, errors="replace"):
        m = re.search(r"expert pipeline: waited (\d+\.\d+) s for [\d.]+ GB \(([\d.]+) GB/s\); hot (\d+) reads ([\d.]+) ms avg, cold (\d+) reads ([\d.]+) ms avg", line)
        if m:
            w, gbs, h, hm, c, cm = float(m[1]), float(m[2]), int(m[3]), float(m[4]), int(m[5]), float(m[6])
            if n > 0:   # skip the first token (prompt/staging)
                hot += h; cold += c; hot_ms += hm * h; cold_ms += cm * c; waits.append(w)
            n += 1
        m = re.search(r"K3 pos \d+ token \d+ -> argmax \d+ \(.*\) (\d+\.\d+) s$", line)
        if m: toks.append(float(m[1]))
    k = n - 1
    if k > 0:
        print("%s: %d tokens after the first: hot %.1f reads/token (%.1f/layer, %.1f ms avg), cold %.1f reads/token (%.2f/layer, %.1f ms avg), wait %.3f s/token" %
              (path.split("/")[-1], k, hot / k, hot / k / 92, hot_ms / max(hot, 1), cold / k, cold / k / 92, cold_ms / max(cold, 1), sum(waits) / len(waits)))
    if len(toks) > 2:
        d = toks[1:]   # per-token durations as printed; the first token carries the prompt's tail
        print("   token times: mean %.2f s, min %.2f, max %.2f over %d" % (sum(d) / len(d), min(d), max(d), len(d)))
    for line in open(path, errors="replace"):
        if "K3 generated" in line or "rc=" in line or "K3 prefill" in line: print("   " + line.strip())
