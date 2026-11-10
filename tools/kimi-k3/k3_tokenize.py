#!/usr/bin/env python3
"""Kimi K3 tokenizer (tiktoken ranks + K3 pre-tokenisation pattern), for reference prompts. Prints token ids."""
import base64, json, os, sys
import regex
META = "<path>"
PATTERN = regex.compile(
    r"[\p{Han}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)|\s+", flags=regex.V1)
def load_ranks():
    ranks = {}
    with open(os.path.join(META, "tiktoken.model"), "rb") as f:
        for line in f:
            if line.strip():
                b64, r = line.split(); ranks[base64.b64decode(b64)] = int(r)
    return ranks
def bpe(piece: bytes, ranks):
    parts = [bytes([b]) for b in piece]
    while len(parts) > 1:
        best, best_rank = None, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best_rank is None or r < best_rank):
                best, best_rank = i, r
        if best is None: break
        parts[best:best + 2] = [parts[best] + parts[best + 1]]
    return [ranks[p] for p in parts]
def encode(text, ranks):
    ids = []
    for m in PATTERN.finditer(text):
        piece = m.group(0).encode("utf-8")
        ids.extend(ranks.get(piece, None) is not None and [ranks[piece]] or bpe(piece, ranks))
    return ids
if __name__ == "__main__":
    ranks = load_ranks()
    text = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    ids = encode(text, ranks)
    cfg = json.load(open(os.path.join(META, "config.json")))
    print("text:", repr(text)); print("ids:", ids); print("with BOS:", [cfg["bos_token_id"]] + ids)
    inv = {v: k for k, v in ranks.items()}
    print("pieces:", [inv[i] for i in ids])
