# Per-request summary of a K3 serve log with DS4_K3_TRACE_TIMING: prompt tokens, decode tok/s, cache hit shares.
import re, sys
reqs = []; cur = None
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"K3 prefill (\d+) tokens in ([\d.]+) s", line)
    if m: cur = {"prompt": int(m[1]), "prefill_s": float(m[2]), "dev": 0, "host": 0, "disk": 0, "tok": [], "gen": None}; reqs.append(cur); continue
    if cur is None: continue
    m = re.search(r"expert cache: device (\d+), host (\d+), disk (\d+)", line)
    if m: cur["dev"] += int(m[1]); cur["host"] += int(m[2]); cur["disk"] += int(m[3]); continue
    m = re.search(r"K3 pos \d+ token \d+ -> argmax \d+ \(.*\) ([\d.]+) s$", line)
    if m: cur["tok"].append(float(m[1])); continue
    m = re.search(r"K3 generated (\d+) tokens in ([\d.]+) s", line)
    if m: cur["gen"] = (int(m[1]), float(m[2]))
for i, r in enumerate(reqs):
    tot = r["dev"] + r["host"] + r["disk"]
    toks = r["tok"][1:]
    g = r["gen"] or (0, 0.0)
    print("req %d: prompt %d tok, prefill %.1f s; %d generated in %.1f s = %.2f tok/s; token mean %.2f s (min %.2f max %.2f); cache: device %.0f %%, host %.0f %%, disk %.0f %%" %
          (i + 1, r["prompt"], r["prefill_s"], g[0], g[1], g[0] / g[1] if g[1] else 0.0,
           sum(toks) / len(toks) if toks else 0.0, min(toks) if toks else 0.0, max(toks) if toks else 0.0,
           100.0 * r["dev"] / max(tot, 1), 100.0 * r["host"] / max(tot, 1), 100.0 * r["disk"] / max(tot, 1)))
