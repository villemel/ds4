#!/usr/bin/env python3
"""Kimi K3 / K3.1 Pro spine -> GGUF v3 for the ds4 'kimi-k3' architecture (2026-10-31; 5 Nov 2026: --name, --compact).

Reads the deltafin BF16/F32 spine tensors (raw little-endian planes named by HF tensor name) and the K3 config,
writes every non-expert tensor as Q8_0 (matrices) or F32 (vectors, convs) with ds4-style names. Routed experts are
NOT written: ds4's K3 expert source streams them from the deltafin raw-v1 files (metadata records where).
--compact leaves the file at its real length (no sparse tail) and writes <out>.sparse.json with the length the
server needs: on the serving disk run `truncate -s <sparse_bytes> <out>` (exFAT cannot hold the sparse tail).
K3.1 Pro: identical tensor set apart from the 120 self_attn.indexer.* planes, which this converter does not read.
"""
import argparse, base64, json, os, struct, sys, time
import numpy as np

GGUF_MAGIC = 0x46554747
T_UINT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64 = 4, 6, 7, 8, 9, 10
GGML_F32, GGML_F16, GGML_Q8_0, GGML_BF16, GGML_MXFP4 = 0, 1, 8, 30, 39
ALIGN = 32

def w_str(f, s):
    b = s if isinstance(s, (bytes, bytearray)) else s.encode("utf-8")
    f.write(struct.pack("<Q", len(b))); f.write(b)

def w_kv(f, key, vtype, value):
    w_str(f, key); f.write(struct.pack("<I", vtype))
    if vtype == T_UINT32: f.write(struct.pack("<I", value))
    elif vtype == T_UINT64: f.write(struct.pack("<Q", value))
    elif vtype == T_FLOAT32: f.write(struct.pack("<f", value))
    elif vtype == T_BOOL: f.write(struct.pack("<B", 1 if value else 0))
    elif vtype == T_STRING: w_str(f, value)
    else: raise ValueError(vtype)

def w_kv_array(f, key, elem_type, values):
    w_str(f, key); f.write(struct.pack("<I", T_ARRAY)); f.write(struct.pack("<I", elem_type)); f.write(struct.pack("<Q", len(values)))
    if elem_type == T_UINT32: f.write(np.asarray(values, dtype="<u4").tobytes())
    elif elem_type == T_STRING:
        for v in values: w_str(f, v)
    else: raise ValueError(elem_type)

def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)

def q8_0_bytes(x: np.ndarray) -> bytes:
    """x: float32 [rows, cols], cols % 32 == 0 -> Q8_0 blocks (f16 d, 32 x int8) row-major."""
    blocks = x.reshape(-1, 32)
    amax = np.abs(blocks).max(axis=1)
    d = (amax / 127.0).astype(np.float32)
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    q = np.rint(blocks * inv[:, None]).clip(-128, 127).astype(np.int8)
    out = np.empty((blocks.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype("<f2").view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()

class Tensor:
    def __init__(self, name, src, shape, dtype_src, out_type, transform=None):
        self.name, self.src, self.shape, self.dtype_src, self.out_type, self.transform = name, src, tuple(shape), dtype_src, out_type, transform
        n = int(np.prod(self.shape))
        if out_type == GGML_Q8_0:
            assert self.shape[-1] % 32 == 0, (name, shape)
            self.nbytes = n // 32 * 34
        elif out_type == GGML_F32: self.nbytes = n * 4
        elif out_type == GGML_BF16: self.nbytes = n * 2
        elif out_type == GGML_MXFP4:
            assert self.shape[-1] % 32 == 0, (name, shape); self.nbytes = n // 32 * 17
        else: raise ValueError(out_type)
        self.virtual = src is None

def load_plane(path, shape, dtype_src):
    n = int(np.prod(shape)); size = os.path.getsize(path)
    if dtype_src == "BF16":
        assert size == n * 2, (path, size, n * 2)
        return bf16_to_f32(np.fromfile(path, dtype="<u2")).reshape(shape)
    assert size == n * 4, (path, size, n * 4)
    return np.fromfile(path, dtype="<f4").reshape(shape)

def build_tensor_list(cfg, inv, src_dir):
    t = cfg["text_config"]; L = t["num_hidden_layers"]
    full = set(i - 1 for i in t["linear_attn_config"]["full_attn_layers"])  # 1-indexed in config
    hf = "language_model.model."
    tensors = []
    def add(gname, hname, out_type, transform=None):
        meta = inv[hname]
        shape = meta["shape"]
        if hname.endswith("_res_proj.weight") or hname.endswith("res_proj.weight"):
            shape = [shape[-1]]  # [1, 7168] -> [7168]
        tensors.append(Tensor(gname, os.path.join(src_dir, hname), shape, meta["dtype"], out_type, transform))
    add("token_embd.weight", hf + "embed_tokens.weight", GGML_Q8_0)
    add("output_norm.weight", hf + "norm.weight", GGML_F32)
    add("output_res_norm.weight", hf + "output_attn_res_norm.weight", GGML_F32)
    add("output_res_proj.weight", hf + "output_attn_res_proj.weight", GGML_F32)
    add("output.weight", "language_model.lm_head.weight", GGML_Q8_0)
    layer_types = []
    for l in range(L):
        p = f"{hf}layers.{l}."; b = f"blk.{l}."
        add(b + "attn_norm.weight", p + "input_layernorm.weight", GGML_F32)
        add(b + "ffn_norm.weight", p + "post_attention_layernorm.weight", GGML_F32)
        add(b + "res_attn_norm.weight", p + "self_attention_res_norm.weight", GGML_F32)
        add(b + "res_attn_proj.weight", p + "self_attention_res_proj.weight", GGML_F32)
        add(b + "res_ffn_norm.weight", p + "mlp_res_norm.weight", GGML_F32)
        add(b + "res_ffn_proj.weight", p + "mlp_res_proj.weight", GGML_F32)
        if l in full:
            layer_types.append(1)
            add(b + "attn_q_a.weight", p + "self_attn.q_a_proj.weight", GGML_Q8_0)
            add(b + "attn_q_a_norm.weight", p + "self_attn.q_a_layernorm.weight", GGML_F32)
            add(b + "attn_q_b.weight", p + "self_attn.q_b_proj.weight", GGML_Q8_0)
            add(b + "attn_kv_a_mqa.weight", p + "self_attn.kv_a_proj_with_mqa.weight", GGML_Q8_0)
            add(b + "attn_kv_a_norm.weight", p + "self_attn.kv_a_layernorm.weight", GGML_F32)
            add(b + "attn_kv_b.weight", p + "self_attn.kv_b_proj.weight", GGML_Q8_0)   # [96*(128+128), 512], head-major
            add(b + "attn_gate.weight", p + "self_attn.g_proj.weight", GGML_Q8_0)
            add(b + "attn_output.weight", p + "self_attn.o_proj.weight", GGML_Q8_0)
        else:
            layer_types.append(0)
            add(b + "kda_q.weight", p + "self_attn.q_proj.weight", GGML_Q8_0)
            add(b + "kda_k.weight", p + "self_attn.k_proj.weight", GGML_Q8_0)
            add(b + "kda_v.weight", p + "self_attn.v_proj.weight", GGML_Q8_0)
            add(b + "kda_q_conv.weight", p + "self_attn.q_conv1d.weight", GGML_F32)
            add(b + "kda_k_conv.weight", p + "self_attn.k_conv1d.weight", GGML_F32)
            add(b + "kda_v_conv.weight", p + "self_attn.v_conv1d.weight", GGML_F32)
            add(b + "kda_f_a.weight", p + "self_attn.f_a_proj.weight", GGML_Q8_0)
            add(b + "kda_f_b.weight", p + "self_attn.f_b_proj.weight", GGML_Q8_0)
            add(b + "kda_beta.weight", p + "self_attn.b_proj.weight", GGML_Q8_0)
            add(b + "kda_gate.weight", p + "self_attn.g_proj.weight", GGML_Q8_0)
            add(b + "kda_a_log.weight", p + "self_attn.A_log", GGML_F32)
            add(b + "kda_dt_bias.weight", p + "self_attn.dt_bias", GGML_F32)
            add(b + "kda_o_norm.weight", p + "self_attn.o_norm.weight", GGML_F32)
            add(b + "kda_output.weight", p + "self_attn.o_proj.weight", GGML_Q8_0)
        if l < t["first_k_dense_replace"]:
            add(b + "ffn_gate.weight", p + "mlp.gate_proj.weight", GGML_Q8_0)
            add(b + "ffn_up.weight", p + "mlp.up_proj.weight", GGML_Q8_0)
            add(b + "ffn_down.weight", p + "mlp.down_proj.weight", GGML_Q8_0)
        else:
            m = p + "block_sparse_moe."
            add(b + "ffn_gate_inp.weight", m + "gate.weight", GGML_F32)   # ds4 router kernels read F32
            add(b + "exp_probs_b.bias", m + "gate.e_score_correction_bias", GGML_F32)
            add(b + "moe_latent_down.weight", m + "routed_expert_down_proj.weight", GGML_Q8_0)
            add(b + "moe_latent_norm.weight", m + "routed_expert_norm.weight", GGML_F32)
            add(b + "moe_latent_up.weight", m + "routed_expert_up_proj.weight", GGML_Q8_0)
            add(b + "ffn_gate_shexp.weight", m + "shared_experts.gate_proj.weight", GGML_Q8_0)
            add(b + "ffn_up_shexp.weight", m + "shared_experts.up_proj.weight", GGML_Q8_0)
            add(b + "ffn_down_shexp.weight", m + "shared_experts.down_proj.weight", GGML_Q8_0)
    # Routed experts: declared in GGUF block_mxfp4 layout ([experts, rows, cols], 17 bytes per 32) but with no data in
    # the file; ds4's K3 expert source serves these offsets from the deltafin raw-v1 files (planes -> blocks on the fly).
    E, FF, LAT = t["num_experts"], t["moe_intermediate_size"], t["routed_expert_hidden_size"]
    for l in range(t["first_k_dense_replace"], L):
        b = f"blk.{l}."
        tensors.append(Tensor(b + "ffn_gate_exps.weight", None, (E, FF, LAT), None, GGML_MXFP4))
        tensors.append(Tensor(b + "ffn_up_exps.weight", None, (E, FF, LAT), None, GGML_MXFP4))
        tensors.append(Tensor(b + "ffn_down_exps.weight", None, (E, LAT, FF), None, GGML_MXFP4))
    return tensors, layer_types

def load_vocab(meta_dir):
    toks = {}
    with open(os.path.join(meta_dir, "tiktoken.model"), "rb") as f:
        for line in f:
            if not line.strip(): continue
            b64, rank = line.split(); toks[int(rank)] = base64.b64decode(b64)
    base = len(toks); assert base == 163584 and set(toks) == set(range(base))
    cfg = json.load(open(os.path.join(meta_dir, "tokenizer_config.json")))
    added = {int(k): v["content"] for k, v in cfg.get("added_tokens_decoder", {}).items()}
    vocab = [toks[i] for i in range(base)]
    for i in range(base, base + 256):
        vocab.append(added.get(i, f"<|reserved_token_{i}|>").encode("utf-8"))
    return vocab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="directory with config.json and tensor_inventory_offsets.json")
    ap.add_argument("--src", required=True, help="directory of the resident (non-expert) tensor planes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None, help="only the first N layers (format smoke test)")
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--primary-experts", required=True, help="expert directory recorded as kimi-k3.experts.primary_dir")
    ap.add_argument("--dir-b-experts", default="", help="optional second expert directory (kimi-k3.experts.dir_b)")
    ap.add_argument("--name", default="Kimi K3", help="general.name")
    ap.add_argument("--compact", action="store_true", help="no sparse tail; write <out>.sparse.json instead")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.meta, "config.json"))); t = cfg["text_config"]
    inv = json.load(open(os.path.join(a.meta, "tensor_inventory_offsets.json")))
    tensors, layer_types = build_tensor_list(cfg, inv, a.src)
    if a.layers is not None:
        keep = {f"blk.{l}." for l in range(a.layers)}
        tensors = [x for x in tensors if not x.name.startswith("blk.") or any(x.name.startswith(k) for k in keep)]
    # real tensors first, virtual expert tensors last (they only extend the file sparsely)
    tensors = [x for x in tensors if not x.virtual] + [x for x in tensors if x.virtual]
    if a.skip_embed:
        tensors = [x for x in tensors if x.name not in ("token_embd.weight", "output.weight")]
    vocab = load_vocab(a.meta)
    kv = [
        ("general.architecture", T_STRING, "kimi-k3"), ("general.name", T_STRING, a.name),
        ("general.file_type", T_UINT32, 7),
        ("kimi-k3.block_count", T_UINT32, t["num_hidden_layers"]),
        ("kimi-k3.context_length", T_UINT64, t["max_position_embeddings"]),
        ("kimi-k3.embedding_length", T_UINT32, t["hidden_size"]),
        ("kimi-k3.vocab_size", T_UINT32, t["vocab_size"] if "vocab_size" in t else 163840),
        ("kimi-k3.feed_forward_length", T_UINT32, t["intermediate_size"]),
        ("kimi-k3.expert_feed_forward_length", T_UINT32, t["moe_intermediate_size"]),
        ("kimi-k3.expert_count", T_UINT32, t["num_experts"]),
        ("kimi-k3.expert_used_count", T_UINT32, t["num_experts_per_token"]),
        ("kimi-k3.expert_shared_count", T_UINT32, t["num_shared_experts"]),
        ("kimi-k3.expert_shared_feed_forward_length", T_UINT32, 6144),
        ("kimi-k3.routed_latent_length", T_UINT32, t["routed_expert_hidden_size"]),
        ("kimi-k3.leading_dense_block_count", T_UINT32, t["first_k_dense_replace"]),
        ("kimi-k3.expert_weights_scale", T_FLOAT32, float(t["routed_scaling_factor"])),
        ("kimi-k3.expert_weights_norm", T_BOOL, bool(t["moe_renormalize"])),
        ("kimi-k3.attention.head_count", T_UINT32, t["num_attention_heads"]),
        ("kimi-k3.attention.key_length", T_UINT32, t["qk_nope_head_dim"] + t["qk_rope_head_dim"]),
        ("kimi-k3.attention.value_length", T_UINT32, 128),
        ("kimi-k3.attention.nope_dimension_count", T_UINT32, t["qk_nope_head_dim"]),
        ("kimi-k3.attention.rope_dimension_count", T_UINT32, t["qk_rope_head_dim"]),
        ("kimi-k3.attention.rope_applied", T_BOOL, False),
        ("kimi-k3.attention.q_lora_rank", T_UINT32, t["q_lora_rank"]),
        ("kimi-k3.attention.kv_lora_rank", T_UINT32, t["kv_lora_rank"]),
        ("kimi-k3.attention.layer_norm_rms_epsilon", T_FLOAT32, float(t["rms_norm_eps"])),
        ("kimi-k3.linear_attention.head_count", T_UINT32, t["num_attention_heads"]),
        ("kimi-k3.linear_attention.head_dimension", T_UINT32, t["linear_attn_config"]["head_dim"]),
        ("kimi-k3.linear_attention.conv_kernel", T_UINT32, 4),
        ("kimi-k3.linear_attention.gate_lower_bound", T_FLOAT32, float(t["linear_attn_config"]["gate_lower_bound"])),
        ("kimi-k3.residual.block_size", T_UINT32, t["attn_res_block_size"]),
        ("kimi-k3.activation.situ_beta", T_FLOAT32, float(t["activation_situ_beta"])),
        ("kimi-k3.activation.situ_linear_beta", T_FLOAT32, float(t["activation_situ_linear_beta"])),
        ("kimi-k3.experts.source", T_STRING, "deltafin-raw-v1"),
        ("kimi-k3.experts.primary_dir", T_STRING, a.primary_experts),
        ("kimi-k3.experts.dir_b", T_STRING, a.dir_b_experts),
        ("kimi-k3.experts.file_bytes", T_UINT64, 17547264),
        ("tokenizer.ggml.model", T_STRING, "kimi-k3-tiktoken"),
        ("tokenizer.ggml.bos_token_id", T_UINT32, cfg["bos_token_id"]),
        ("tokenizer.ggml.eos_token_id", T_UINT32, cfg["eos_token_id"]),
        ("tokenizer.ggml.padding_token_id", T_UINT32, cfg["pad_token_id"]),
    ]
    # header
    with open(a.out, "wb") as f:
        f.write(struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), len(kv) + 2))
        for key, vt, val in kv: w_kv(f, key, vt, val)
        w_kv_array(f, "kimi-k3.layer_types", T_UINT32, layer_types)
        w_kv_array(f, "tokenizer.ggml.tokens", T_STRING, vocab)
        offset = 0
        for x in tensors:
            w_str(f, x.name); dims = list(reversed(x.shape))
            f.write(struct.pack("<I", len(dims)))
            for d in dims: f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", x.out_type)); f.write(struct.pack("<Q", offset))
            offset += (x.nbytes + ALIGN - 1) // ALIGN * ALIGN
        pos = f.tell(); pad = (ALIGN - pos % ALIGN) % ALIGN; f.write(b"\0" * pad)
        data_start = f.tell()
        total = sum(x.nbytes for x in tensors if not x.virtual); done = 0; t0 = time.time()
        virtual_end = data_start + offset
        for i, x in enumerate(tensors):
            if x.virtual:
                continue
            start = f.tell()
            rows = x.shape[0] if len(x.shape) > 1 else 1
            cols = int(np.prod(x.shape[1:])) if len(x.shape) > 1 else int(np.prod(x.shape))
            if x.out_type == GGML_Q8_0:
                step = max(1, (64 << 20) // (cols * 4))
                n = int(np.prod(x.shape)); size = os.path.getsize(x.src)
                assert size == n * (2 if x.dtype_src == "BF16" else 4), (x.src, size)
                with open(x.src, "rb") as src:
                    for r0 in range(0, rows, step):
                        r1 = min(rows, r0 + step)
                        if x.dtype_src == "BF16":
                            raw = np.frombuffer(src.read((r1 - r0) * cols * 2), dtype="<u2"); chunk = bf16_to_f32(raw).reshape(r1 - r0, cols)
                        else:
                            chunk = np.frombuffer(src.read((r1 - r0) * cols * 4), dtype="<f4").reshape(r1 - r0, cols)
                        f.write(q8_0_bytes(chunk))
            elif x.out_type == GGML_F32:
                f.write(load_plane(x.src, x.shape, x.dtype_src).astype("<f4").tobytes())
            elif x.out_type == GGML_BF16:
                if x.dtype_src == "BF16": f.write(open(x.src, "rb").read())
                else:
                    v = load_plane(x.src, x.shape, x.dtype_src); f.write((v.view(np.uint32) >> 16).astype("<u2").tobytes())
            written = f.tell() - start; assert written == x.nbytes, (x.name, written, x.nbytes)
            pad = (ALIGN - f.tell() % ALIGN) % ALIGN; f.write(b"\0" * pad)
            done += x.nbytes
            if i % 25 == 0 or i == len(tensors) - 1:
                el = time.time() - t0
                print(f"[{i+1}/{len(tensors)}] {x.name} {done/1e9:.1f}/{total/1e9:.1f} GB {el:.0f}s", flush=True)
    real_size = os.path.getsize(a.out)
    if a.compact:
        json.dump({"real_bytes": real_size, "sparse_bytes": virtual_end, "note": "run: truncate -s <sparse_bytes> <file> on the serving disk"}, open(a.out + ".sparse.json", "w"))
    else:
        os.truncate(a.out, virtual_end)   # sparse: the expert region is never read from this file
    print("done:", a.out, "real bytes", real_size / 1e9, "GB; sparse size", virtual_end / 1e9, "GB; tensors", len(tensors), "(virtual", sum(1 for x in tensors if x.virtual), "); data at", data_start)

if __name__ == "__main__":
    main()
