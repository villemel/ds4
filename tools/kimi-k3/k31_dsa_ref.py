#!/usr/bin/env python3
"""Step 95 reference check: recompute one decode position's indexer and sparse attention in float64 from what ds4 dumped
(DS4_K3_DSA_DUMP) and the K3IX weights, and compare. Usage: k31_dsa_ref.py <dump dir> <layer> <k3ix file> [top_k]"""
import struct, sys
import numpy as np
d, L, wfile = sys.argv[1], int(sys.argv[2]), sys.argv[3]; TOPK = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
def ld(name, dt=np.float32): return np.fromfile(f"{d}/L{L}_{name}.bin", dtype=dt)
W = {}
with open(wfile, "rb") as f:
    assert f.read(4) == b"K3IX"; ver, nl = struct.unpack("<II", f.read(8))
    for _ in range(nl):
        blk, nt = struct.unpack("<II", f.read(8))
        for _ in range(nt):
            n = struct.unpack("<I", f.read(4))[0]; name = f.read(n).decode(); dt, nd = struct.unpack("<II", f.read(8))
            shape = struct.unpack(f"<{nd}Q", f.read(8 * nd)); nb = struct.unpack("<Q", f.read(8))[0]; raw = f.read(nb)
            if blk != L: continue
            a = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16 if dt == 1 else np.frombuffer(raw, np.uint32)
            W[name] = a.view(np.float32).astype(np.float64).reshape(shape)
H, D = 32, 128
x = ld("normed").astype(np.float64); qr = ld("q_an").astype(np.float64)
q = (W["wq_b.weight"] @ qr).reshape(H, D); kraw = W["wk.weight"] @ x; w = W["weights_proj.weight"] @ x
mu = kraw.mean(); var = ((kraw - mu) ** 2).mean(); k = (kraw - mu) / np.sqrt(var + 1e-6) * W["k_norm.weight"] + W["k_norm.bias"]
keys = ld("idx_keys").astype(np.float64).reshape(-1, D); npos = keys.shape[0]; pos = npos - 1
rel = lambda a, b: float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))
res = {"index query": rel(ld("idx_q").reshape(H, D), q), "index key (raw)": rel(ld("idx_kraw"), kraw),
       "index key (stored, layer-normed)": rel(keys[pos], k), "per-head weights": rel(ld("idx_w"), w)}
scale = 1.0 / np.sqrt(D * H)
sc = scale * (np.maximum(q @ keys.T, 0.0) * w[:, None]).sum(0)     # q from the reference, earlier keys from the cache
res["scores"] = rel(ld("scores").astype(np.float64), sc)
ours = set(int(i) for i in ld("selected", np.uint32) if i < npos)
ref = set(int(i) for i in np.argsort(-sc, kind="stable")[:TOPK])
kth = np.sort(sc)[::-1][TOPK - 1]
diff = sorted(ours ^ ref)
near = all(abs(sc[i] - kth) <= 1e-5 * max(abs(kth), 1e-30) for i in diff)
mask = ld("mask", np.uint8); mask_ok = set(np.nonzero(mask)[0].tolist()) == ours
mq = ld("mla_q").astype(np.float64).reshape(96, 192); K = ld("key_cache").astype(np.float64).reshape(npos, 96, 192)
V = ld("value_cache").astype(np.float64).reshape(npos, 96, 128); out = ld("heads").astype(np.float64).reshape(96, 128)
def attend(sel):
    idx = np.array(sorted(sel)); s = np.einsum("hd,thd->ht", mq, K[idx]) / np.sqrt(192.0)
    p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); return np.einsum("ht,thd->hd", p, V[idx])
res["attention on our selection"] = rel(out, attend(ours))
full = attend(range(npos))
print(f"layer {L}, position {pos}: {len(ours)} selected of {npos}; overlap with the float64 top-{TOPK}: {len(ours & ref)} "
      f"({len(diff)} differ{', all within 1e-5 of the cut-off score' if diff and near else ''}); mask matches selection: {mask_ok}")
for k_, v in res.items(): print(f"  {k_:34} max relative difference {v:.2e}")
print(f"  full attention differs from ours by {rel(out, full):.2e} (it should, above {TOPK} positions)")
ok = all(v < 1e-4 for v in res.values()) and mask_ok and (not diff or near) and len(ours) == TOPK
print("REF:", "PASS" if ok else "FAIL")
