#!/usr/bin/env python3
"""Build an expert-use histogram for DS4_K3_HEAT from route traces.

Run a few prompts of your own with DS4_K3_TRACE_ROUTES=1 (the engine prints one `route layer L pos P: e:w ...` line per MoE
layer and position on stdout), then:

    heat_from_routes.py <out.bin> <trace> [<trace> ...]

The output has the layout the engine reads: a 48-byte header, then 92 x 896 float32 counts, layer-major over the MoE layers
1..92, each the number of positions that routed to the expert. Only the ratios within a layer matter. The same counts rank
the experts for the weight layout: the most used ones belong on the fastest drive and are the first to get a second home."""
import re, struct, sys

LAYERS, EXPERTS = 92, 896
pat = re.compile(r"route layer (\d+) pos (\d+):((?: -?\d+:[0-9.]+)+)")   # generated text may precede it on the same line
counts = [0.0] * (LAYERS * EXPERTS)
seen = set()
for path in sys.argv[2:]:
    for line in open(path, errors="replace"):
        m = pat.search(line)
        if not m:
            continue
        layer, pos = int(m[1]), int(m[2])
        if not 1 <= layer <= LAYERS or (path, layer, pos) in seen:   # a position traced twice (a re-run step) counts once
            continue
        seen.add((path, layer, pos))
        for item in m[3].split():
            e = int(item.split(":")[0])
            if 0 <= e < EXPERTS:
                counts[(layer - 1) * EXPERTS + e] += 1.0
if len(sys.argv) < 3 or not seen:
    sys.exit("usage: heat_from_routes.py <out.bin> <trace> [<trace> ...]  (no route lines found)")
positions = len(seen) // LAYERS
header = struct.pack("<8sIIIQ", b"K3HEAT01", 1, LAYERS, EXPERTS, positions).ljust(48, b"\0")
with open(sys.argv[1], "wb") as f:
    f.write(header + struct.pack(f"<{LAYERS * EXPERTS}f", *counts))
used = sum(1 for c in counts if c > 0)
print(f"{sys.argv[1]}: {positions} positions from {len(sys.argv) - 2} trace(s), {used} of {LAYERS * EXPERTS} experts used at least once")
