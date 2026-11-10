#!/usr/bin/env python3
"""Kimi K3 / K3.1 Pro safetensors -> the ds4 kimi-k3 layout (4 Nov 2026; 5 Nov 2026: --replace-in, --check-homes, --expert-range).

Writes, from a downloaded Hugging Face checkpoint directory:
  --out-experts  L{layer}-E{expert}.bin, one 17,547,264-byte file per routed expert: the six HF tensors
                 w1.weight_packed, w1.weight_scale, w2.weight_packed, w2.weight_scale, w3.weight_packed, w3.weight_scale
                 concatenated verbatim (deltafin raw-v1, what ds4's expert source and deltafin's weight_fetch use);
  --replace-in   DIR[,DIR...]: the in-place cutover. Every routed expert overwrites L{l}-E{e}.bin in EVERY listed
                 directory where that file already exists (.part, fsync, atomic rename), so the existing layout
                 (primary / dir_b / tier-C duplicates) keeps its shape with the new weights. Experts with no home in
                 any listed directory count as "nohome" (and are written to --out-experts when that is given too).
                 Progress is recorded in --done-file (default <first dir>/.k31-replace-done), one expert per line, so a restart skips the
                 completed experts (an overwritten file looks complete either way).
  --check-homes  with --replace-in: only report the home coverage (experts with 0/1/2/3 homes), write nothing.
  --out-resident tensors/<hf name>: every non-expert language_model tensor as its raw little-endian bytes (BF16/F32);
  --out-meta     config.json, generation_config.json, tokenizer files, LICENSE, and tensor_inventory_offsets.json
                 ({name: {dtype, shape, offsets, shard, hlen}}), i.e. everything tools/kimi-k3/k3_spine_to_gguf.py reads.
Pure Python, memory-mapped shards, resumable. Shards that are not yet fully downloaded are skipped and reported.
K3.1 Pro (verified 5 Nov 2026 against the full inventories): the expert tensors are identical to K3's in name, dtype
and shape (494,592 U8 tensors, 17,547,264 B per expert); the only new tensors are 120 self_attn.indexer.* planes on
the 24 full-attention layers, which land in --out-resident like any other plane and which the spine converter ignores.
"""
import argparse, json, os, re, struct, sys, time, mmap, shutil, collections
from concurrent.futures import ThreadPoolExecutor

EXPERT_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.(weight_packed|weight_scale)$")
ORDER = [("w1", "weight_packed"), ("w1", "weight_scale"), ("w2", "weight_packed"), ("w2", "weight_scale"), ("w3", "weight_packed"), ("w3", "weight_scale")]
EXPERT_BYTES = 17547264
META_FILES = ["config.json", "generation_config.json", "tokenizer_config.json", "tiktoken.model", "LICENSE", "tokenization_kimi.py", "encoding_k3.py", "README.md"]

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return n, hdr

class Shards:
    def __init__(self, src):
        self.src = src; self.maps = {}; self.hdr = {}; self.hlen = {}
    def complete(self, shard):
        p = os.path.join(self.src, shard)
        return os.path.exists(p) and not os.path.exists(p + ".incomplete")
    def open(self, shard):
        if shard not in self.maps:
            p = os.path.join(self.src, shard)
            n, hdr = read_header(p); self.hdr[shard] = hdr; self.hlen[shard] = n
            f = open(p, "rb"); self.maps[shard] = (f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        return self.maps[shard][1]
    def tensor_bytes(self, shard, name):
        m = self.open(shard); meta = self.hdr[shard][name]; a, b = meta["data_offsets"]; base = 8 + self.hlen[shard]
        return m[base + a: base + b], meta

def write_atomic(path, parts):
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        for p in parts: f.write(p)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out-experts"); ap.add_argument("--out-resident"); ap.add_argument("--out-meta")
    ap.add_argument("--replace-in", help="comma-separated expert directories to overwrite in place (see module doc)")
    ap.add_argument("--check-homes", action="store_true", help="with --replace-in: report coverage only")
    ap.add_argument("--done-file", help="progress file for --replace-in (default <first dir>/.k31-replace-done; keep it outside the expert directories: the serve guard counts directory entries")
    ap.add_argument("--layers", default="1-92", help="MoE layer range for the expert files, e.g. 1-92 or 1-4")
    ap.add_argument("--expert-range", default="0-895", help="expert index range within each layer (tests)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    index = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    sh = Shards(a.src)
    lo, hi = (int(x) for x in a.layers.split("-")); e_lo, e_hi = (int(x) for x in a.expert_range.split("-"))
    t0 = time.time()
    if a.out_meta:
        os.makedirs(a.out_meta, exist_ok=True)
        for f in META_FILES:
            p = os.path.join(a.src, f)
            if os.path.exists(p): shutil.copy2(p, os.path.join(a.out_meta, f))
        inv = {}
        missing = sorted({s for s in set(index.values()) if not sh.complete(s)})
        for shard in sorted(set(index.values())):
            if not sh.complete(shard): continue
            n, hdr = read_header(os.path.join(a.src, shard))
            for name, meta in hdr.items():
                inv[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "offsets": meta["data_offsets"], "shard": shard, "hlen": n}
        json.dump(inv, open(os.path.join(a.out_meta, "tensor_inventory_offsets.json"), "w"))
        print(f"meta: {len(inv)} inventory entries from {len(set(index.values())) - len(missing)} complete shards; {len(missing)} shards incomplete", flush=True)
    if a.out_resident:
        tdir = os.path.join(a.out_resident, "tensors"); os.makedirs(tdir, exist_ok=True); n_done = n_skip = n_have = 0; nbytes = 0
        want = [(n, s) for n, s in sorted(index.items()) if n.startswith("language_model.") and not EXPERT_RE.match(n)]
        for i, (name, shard) in enumerate(want):
            if not sh.complete(shard): n_skip += 1; continue
            out = os.path.join(tdir, name)
            data, meta = sh.tensor_bytes(shard, name)
            if os.path.exists(out) and os.path.getsize(out) == len(data): n_have += 1; continue
            write_atomic(out, [data]); n_done += 1; nbytes += len(data)
            if n_done % 200 == 0: print(f"resident: {i + 1}/{len(want)} planes, {nbytes / 1e9:.1f} GB written, {time.time() - t0:.0f} s", flush=True)
        print(f"resident: {len(want)} language_model planes expected; {n_done} written ({nbytes / 1e9:.1f} GB), {n_have} already present, {n_skip} skipped (shard incomplete) in {time.time() - t0:.0f} s", flush=True)
    if a.out_experts or a.replace_in:
        homes = [h for h in (a.replace_in.split(",") if a.replace_in else []) if h]
        for h in homes:
            if not os.path.isdir(h): sys.exit(f"not a directory: {h}")
        if a.out_experts: os.makedirs(a.out_experts, exist_ok=True)
        jobs = []
        for l in range(lo, hi + 1):
            for e in range(e_lo, e_hi + 1):
                names = [f"language_model.model.layers.{l}.block_sparse_moe.experts.{e}.{w}.{k}" for w, k in ORDER]
                if any(n not in index for n in names): continue
                jobs.append((l, e, names))
        if a.check_homes:
            cov = collections.Counter()
            for l, e, _ in jobs:
                cov[sum(1 for h in homes if os.path.exists(os.path.join(h, f"L{l}-E{e}.bin")))] += 1
            print(f"check-homes: {len(jobs)} experts; homes per expert: " + ", ".join(f"{k} home(s): {v}" for k, v in sorted(cov.items())), flush=True)
            if cov.get(0, 0): print(f"WARNING: {cov[0]} experts have no home in {homes}", flush=True)
            return
        done = set(); done_f = None
        if homes:
            done_path = a.done_file or os.path.join(homes[0], ".k31-replace-done")
            if os.path.exists(done_path): done = {x.strip() for x in open(done_path) if x.strip()}
            done_f = open(done_path, "a")
        def do(job):
            l, e, names = job
            key = f"L{l}-E{e}"
            if key in done: return "skip"
            targets = [os.path.join(h, key + ".bin") for h in homes if os.path.exists(os.path.join(h, key + ".bin"))]
            out = os.path.join(a.out_experts, key + ".bin") if a.out_experts else None
            if out and os.path.exists(out) and os.path.getsize(out) == EXPERT_BYTES: out = None
            if not targets and not out: return "nohome" if homes else "skip"
            if any(not sh.complete(index[n]) for n in names): return "incomplete"
            parts = [sh.tensor_bytes(index[n], n)[0] for n in names]
            total = sum(len(p) for p in parts)
            if total != EXPERT_BYTES: return f"BAD size {total} for {key}"
            for t in targets: write_atomic(t, parts)
            if out: write_atomic(out, parts)
            return f"ok {key} {len(targets)}"
        counts = collections.Counter(); per_home = collections.Counter(); nb = 0
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, r in enumerate(ex.map(do, jobs)):
                k = r.split()[0]; counts[k] += 1
                if k == "ok":
                    _, key, nt = r.split(); per_home[int(nt)] += 1; nb += EXPERT_BYTES
                    if done_f: done_f.write(key + "\n"); done_f.flush()
                if r.startswith("BAD"): print(r, flush=True)
                if (i + 1) % 2000 == 0:
                    el = time.time() - t0
                    print(f"experts: {i + 1}/{len(jobs)} {dict(counts)} {nb / 1e9:.0f} GB read, {nb / 1e9 / max(el, 1):.2f} GB/s, {el:.0f} s", flush=True)
        if done_f: done_f.close()
        print(f"experts done: {dict(counts)} of {len(jobs)}; homes written per expert: {dict(sorted(per_home.items()))}; {nb / 1e9:.0f} GB in {time.time() - t0:.0f} s", flush=True)
        if counts.get("nohome"): print(f"WARNING: {counts['nohome']} experts had no home in {homes}", flush=True)

if __name__ == "__main__":
    main()
