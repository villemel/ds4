import json, sys, os, struct, re, collections
k3cfg, k3inv, d31 = sys.argv[1], sys.argv[2], sys.argv[3]
a = json.load(open(k3cfg)); b = json.load(open(os.path.join(d31, "config.json")))
def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, p + k + "."))
        else: out[p + k] = v
    return out
fa, fb = flat(a), flat(b)
skip = re.compile(r"^(text_config\.)?(_name_or_path|add_cross_attention|bad_words_ids|begin_suppress_tokens|chunk_size_feed_forward|cross_attention_hidden_size|decoder_start_token_id|diversity_penalty|do_sample|early_stopping|encoder_no_repeat_ngram_size|exponential_decay_length_penalty|finetuning_task|forced_.*|id2label.*|is_decoder|is_encoder_decoder|label2id.*|length_penalty|max_length|min_length|no_repeat_ngram_size|num_beam.*|num_return_sequences|output_.*|prefix|problem_type|pruned_heads|remove_invalid_values|repetition_penalty|return_dict.*|sep_token_id|suppress_tokens|task_specific_params|temperature|tf_legacy_loss|tie_encoder_decoder|top_k|top_p|torchscript|transformers_version|typical_p|use_cache)$")
print("=== CONFIG DIFF (K3 -> K3.1), ignoring HF generation boilerplate")
for k in sorted(set(fa) | set(fb)):
    if skip.match(k): continue
    va, vb = fa.get(k, "<absent>"), fb.get(k, "<absent>")
    if va != vb: print(f"  {k}: {json.dumps(va)[:160]}  ->  {json.dumps(vb)[:160]}")
print("=== TENSOR DIFF")
inv3 = json.load(open(k3inv))
idx = json.load(open(os.path.join(d31, "model.safetensors.index.json")))["weight_map"]
hdrs = {}
inv31 = {}
for name, shard in idx.items():
    if shard not in hdrs:
        with open(os.path.join(d31, shard), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; hdrs[shard] = json.loads(f.read(n))
    h = hdrs[shard][name]; inv31[name] = {"dtype": h["dtype"], "shape": h["shape"]}
json.dump(inv31, open(sys.argv[4], "w"))
def pat(n): return re.sub(r"\.\d+\.", ".N.", n)
def sig(inv):
    c = collections.Counter(); ex = {}
    for n, m in inv.items():
        p = pat(n); key = (p, m["dtype"], tuple(m["shape"]) if ".experts." not in n else tuple(m["shape"]))
        c[key] += 1
    return c
s3, s31 = sig(inv3), sig(inv31)
only3 = sorted(k for k in s3 if k not in s31); only31 = sorted(k for k in s31 if k not in s3)
print("K3 tensors", len(inv3), "K3.1 tensors", len(inv31))
print("--- patterns only in K3:")
for k in only3: print(f"  {s3[k]:7d} {k[1]:5s} {str(k[2]):22s} {k[0]}")
print("--- patterns only in K3.1:")
for k in only31: print(f"  {s31[k]:7d} {k[1]:5s} {str(k[2]):22s} {k[0]}")
print("--- count differences on shared patterns:")
for k in sorted(set(s3) & set(s31)):
    if s3[k] != s31[k]: print(f"  {k[0]} {k[1]} {k[2]}: {s3[k]} -> {s31[k]}")
print("--- experts: ", end="")
e3 = [n for n in inv3 if ".experts." in n]; e31 = [n for n in inv31 if ".experts." in n]
print("K3", len(e3), "K3.1", len(e31), "same names:", set(e3) == set(e31), "same dtype/shape:", all(inv3[n]["dtype"] == inv31[n]["dtype"] and inv3[n]["shape"] == inv31[n]["shape"] for n in e3) if set(e3) == set(e31) else "n/a")
res3 = {n for n in inv3 if ".experts." not in n}; res31 = {n for n in inv31 if ".experts." not in n}
print("--- non-expert tensors: K3", len(res3), "K3.1", len(res31), "| in K3 only:", len(res3 - res31), "| in K3.1 only:", len(res31 - res3))
print("    names in K3 only (first 10):", sorted(res3 - res31)[:10])
print("    names in K3.1 only (patterns):", sorted({pat(n) for n in res31 - res3}))
same = [n for n in res3 & res31 if inv3[n]["dtype"] != inv31[n]["dtype"] or inv3[n]["shape"] != inv31[n]["shape"]]
print("    shared non-expert tensors with different dtype/shape:", len(same), same[:10])
