# Agent Notes

`ds4.c` is an inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under `metal/`.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Always make sure that the SSD streaming, CUDA, distributed inference, Metal default inference are not affected by fixes to other parts of the code.
- Keep model loading mmap-backed for the Metal default case; do not eagerly copy the full GGUF. Keep the model loading for SSD streaming of routed experts explicit: allocated buffers, fast reads from disk, always try to hide loading of missing routed experts by loading them while performing the inference of the shared expert and routed experts already in RAM. Always try to hide loading of layers for prefill in SSD streaming mode using the inference time of the current layer as the next one is loaded.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV checkpoints.

## Quality Rules

- Keep the implementation small, sharp, easy to understand. Try to write elegant code in a state of grace. Don't settle for the first thing that comes to mind, try to find the most minimal and better working design. Don't introduce slop: very fragile code that just patches specific cases, dead code, useless code and code ways more complicated of how it should be.
- Comment important inference code where the model mechanics, cache lifetime, memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is intentional.

## Layout

- `ds4.c`: model loading, tokenizer, CPU reference code, Metal graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `ds4_ssd.c`: SSD streaming of routed experts. `ds4_distributed.c` /
  `ds4_tp.c`: pipeline parallelism / tensor parallelism. `ds4_layer_pack.c`:
  layer packing.
- `ds4_server.c` pulls in `ds4_kvstore.c` (KV disk cache) and `rax.c`
  (radix tree). `ds4_agent.c` + `ds4_web.c`: the coding agent.
- `ds4_bench.c`: speed/throughput regression runner. `ds4_eval.c`: eval/self-test.
- `metal/*.metal`: Metal kernels. `cuda/` and `rocm/`: CUDA (incl. vendored
  `cuda/mmq/`) and ROCm kernels. `gguf-tools/`: offline GGUF/imatrix/quality.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

This list is not complete, check the files for more info.

## Testing

`make` builds `./ds4`, `./ds4-server`, `./ds4-bench`, `./ds4-eval`, `./ds4-agent`.

`make test` builds and runs the full regression suite (`ds4_test`, `ds4_agent_test`,
`ds4-eval --self-test-extractors`, q4k/mxfp4 dot tests, layer pack, mgpu placement,
gpu args, sampling). It needs a model and Metal. Narrower checks:

```sh
./ds4_test --server            # API / prompt-rendering / KV-cache logic
./ds4_test --logprob-vectors   # tokenizer/template/attention/logits vs official vecs
./ds4_test --long-context      # long-context story fact-recall
./ds4_test --tool-call-quality # DSML tool-call emission
./ds4_test --metal-kernels     # isolated Metal kernel numeric checks
```

Speed regressions use `./ds4-bench` (see CONTRIBUTING.md for exact flags).
CUDA-specific changes: `make cuda-regression` on a CUDA machine.
DSpark/MTP smoke: `make dspark-verify-depth` / `make mtp-verify-depth` (they
skip if the model or support GGUF is absent).

At every major change where one of the following could be affected, make sure to:

1. Test the normal Metal path and that speed is still at the level it was.
2. Test the SSD streaming path.
3. Test the distributed inference if it could be affected, but ask the user before doing so.
4. Check if CUDA could be broken after the change, and ask the user to give you access to the CUDA machine to actually test if everything is still fine.

## Pitfalls

- Do not hand-edit generated artifacts: `cuda/mmq/` is vendored llama.cpp
  (see `cuda/mmq/VENDOR.md`), and `ds4_iq2_tables_cuda.inc` / the MXFP4 half LUT
  are generated (verify the LUT with `make check-mxfp4-half-lut`).
- Model GGUFs are engine-specific (DeepSeek V4 Flash/PRO, GLM 5.2 only). An
  arbitrary GGUF will not load — not a general GGUF runner.
- Distributed/tensor-parallel runs split layers via `--layers` and the
  coordinator/worker roles; layer ranges are inclusive (`10:20`,
  `31:output`).
- Exercise caution when emitting raw thinking tags or other parseable units or 
  pairs in output as they are consumed by agent clients and interrupt loops. 
  Disguise, encode or otherwise render them illegible to clients: backticks, 
  grave accents, backquotes, Unicode Code Point U+0060, HTML Entities: &grave, 
  or &#96, with a zero-width space between them. This mitigates the major risk of 
  corrupted KV cache. You can also use: think_open, think_close, etc.
