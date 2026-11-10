#!/usr/bin/env python3
"""Step 95 (K3.1 sparse attention): cut the 120 indexer tensors out of the K3.1 Pro checkpoint into one small file for ds4.
Reads model.safetensors.index.json and only the byte ranges of the indexer tensors, never whole shards. Prints the DSA part of
config.json, every indexer tensor's name, dtype and shape, and refuses anything it does not recognise.
Output <out>.bin: magic 'K3IX', u32 version 1, u32 n_layers, then per layer: u32 block (0-based, as in the spine GGUF), u32 n_tensors,
and per tensor: u32 name_len, name (the checkpoint's suffix after 'indexer.'), u32 dtype (0 f32, 1 bf16), u32 ndim, ndim x u64 shape,
u64 nbytes, raw bytes (little endian, as stored). <out>.json: config's DSA fields, per-tensor shapes, dtypes and sha256.
Usage: k31_indexer_extract.py <checkpoint dir> <out prefix>"""
import hashlib, json, os, re, struct, sys
src, out = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(src, "config.json")))
def walk(d, p=""):
    for k, v in d.items():
        if any(w in k.lower() for w in ("dsa", "index", "topk", "sparse", "layer_types", "full_attn", "kda", "mla_layer", "attn_layer")):
            yield p + k, v
        if isinstance(v, dict): yield from walk(v, p + k + ".")
dsa = dict(walk(cfg))
for k, v in dsa.items(): print("config", k, "=", json.dumps(v)[:400])
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
names = sorted(n for n in idx if ".indexer." in n)
pat = re.compile(r"^(?:language_model\.)?model\.layers\.(\d+)\.self_attn\.indexer\.(.+)$")
layers = {}
for n in names:
    m = pat.match(n)
    if not m: sys.exit(f"unrecognised indexer tensor name: {n}")
    layers.setdefault(int(m.group(1)), []).append((m.group(2), n))
print(f"{len(names)} indexer tensors in {len(layers)} layers: {sorted(layers)}")
hdr_cache = {}
def header(shard):
    if shard not in hdr_cache:
        with open(os.path.join(src, shard), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr_cache[shard] = (8 + n, json.loads(f.read(n)))
    return hdr_cache[shard]
DT = {"F32": 0, "BF16": 1}
man = {"config_dsa": dsa, "layers": {}}
with open(out + ".bin", "wb") as o:
    o.write(b"K3IX" + struct.pack("<II", 1, len(layers)))
    for L in sorted(layers):
        ts = sorted(layers[L])
        o.write(struct.pack("<II", L, len(ts)))
        man["layers"][L] = {}
        for suffix, full in ts:
            base, h = header(idx[full])
            e = h[full]
            if e["dtype"] not in DT: sys.exit(f"{full}: dtype {e['dtype']} not handled (scales or fp8 would need care)")
            a, b = e["data_offsets"]
            with open(os.path.join(src, idx[full]), "rb") as f:
                f.seek(base + a); raw = f.read(b - a)
            if len(raw) != b - a: sys.exit(f"{full}: short read")
            nb = suffix.encode()
            o.write(struct.pack("<I", len(nb)) + nb + struct.pack("<II", DT[e["dtype"]], len(e["shape"])))
            o.write(struct.pack(f"<{len(e['shape'])}Q", *e["shape"]) + struct.pack("<Q", len(raw)) + raw)
            man["layers"][L][suffix] = {"dtype": e["dtype"], "shape": e["shape"], "sha256": hashlib.sha256(raw).hexdigest()}
            print(f"  layer {L:3} {suffix:24} {e['dtype']:5} {e['shape']}")
json.dump(man, open(out + ".json", "w"), indent=1)
print("wrote", out + ".bin", os.path.getsize(out + ".bin"), "bytes")
