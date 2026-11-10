import numpy as np, sys, os
L, E = 1, int(sys.argv[1]) if len(sys.argv) > 1 else 7
p = f"<path>{L}-E{E}.bin"
if not os.path.exists(p): p = f"<path>{L}-E{E}.bin"
raw = np.fromfile(p, dtype=np.uint8); assert raw.size == 17547264
PK, SC = 5505024, 344064
def blocks(packed, scale, rows, cols):
    packed = packed.reshape(rows, cols // 2); scale = scale.reshape(rows, cols // 32)
    nb = cols // 32
    out = np.empty((rows, nb, 17), dtype=np.uint8)
    out[:, :, 0] = scale
    pk = packed.reshape(rows, nb, 16)
    lo = (pk[:, :, 0:8].repeat(2, axis=2) >> (np.arange(16) % 2 * 4).astype(np.uint8)) & 0x0F   # elements 0..15
    hi = (pk[:, :, 8:16].repeat(2, axis=2) >> (np.arange(16) % 2 * 4).astype(np.uint8)) & 0x0F  # elements 16..31
    out[:, :, 1:] = lo | (hi << 4)
    return out.reshape(-1)
w1 = blocks(raw[0:PK], raw[PK:PK+SC], 3072, 3584)
w2 = blocks(raw[PK+SC:2*PK+SC], raw[2*PK+SC:2*(PK+SC)], 3584, 3072)
w3 = blocks(raw[2*(PK+SC):3*PK+2*SC], raw[3*PK+2*SC:3*(PK+SC)], 3072, 3584)
for name, ref in (("gate", w1), ("up", w3), ("down", w2)):
    got = np.fromfile(f"/tmp/k3src-{name}.bin", dtype=np.uint8)
    print(name, "bytes", got.size, "match" if got.size == ref.size and np.array_equal(got, ref) else f"MISMATCH (first diff at {np.argmax(got != ref) if got.size == ref.size else 'size'})")
part = np.fromfile("/tmp/k3src-partial.bin", dtype=np.uint8)
start = 1904 * 5 + 700
print("partial", "match" if np.array_equal(part, w3[start:start + 1000]) else "MISMATCH")
# sanity: dequantize one row of gate with both nibble conventions and compare to deltafin's e2m1 table order
