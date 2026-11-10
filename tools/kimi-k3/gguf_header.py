#!/usr/bin/env python3
"""Print a GGUF file's KV pairs and tensor table (name, type, dims, offset). Stdlib only."""
import struct, sys, json
f = open(sys.argv[1], "rb")
def rd(fmt): return struct.unpack(fmt, f.read(struct.calcsize(fmt)))
def rstr():
    n, = rd("<Q"); return f.read(n).decode("utf-8", "replace")
magic, ver, nt, nkv = rd("<IIQQ")
def rval(t):
    if t in (0, 7): return rd("<B")[0]
    if t == 1: return rd("<b")[0]
    if t == 2: return rd("<H")[0]
    if t == 3: return rd("<h")[0]
    if t == 4: return rd("<I")[0]
    if t == 5: return rd("<i")[0]
    if t == 6: return rd("<f")[0]
    if t == 8: return rstr()
    if t == 9:
        et, = rd("<I"); n, = rd("<Q"); vals = [rval(et) for _ in range(n)]
        return f"array[{n}] " + (json.dumps(vals[:8]) if et != 8 else json.dumps(vals[:5]))
    if t == 10: return rd("<Q")[0]
    if t == 11: return rd("<q")[0]
    if t == 12: return rd("<d")[0]
    raise ValueError(t)
print("magic", hex(magic), "version", ver, "tensors", nt, "kv", nkv)
for _ in range(nkv):
    k = rstr(); t, = rd("<I"); v = rval(t); print("KV", k, "=", v)
for _ in range(nt):
    name = rstr(); nd, = rd("<I"); dims = [rd("<Q")[0] for _ in range(nd)]; typ, = rd("<I"); off, = rd("<Q")
    print("T", name, typ, dims, off)
