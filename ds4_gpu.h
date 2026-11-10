#ifndef DS4_GPU_H
#define DS4_GPU_H

#include <stdbool.h>
#include <stdint.h>
#include "ds4_qwen4_vision.h"
#include "ds4_gpu_copy.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the DS4-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif

#ifndef DS4_GPU_ATTENTION_DECODE_ROW_DEFINED
#define DS4_GPU_ATTENTION_DECODE_ROW_DEFINED
#define DS4_GPU_ATTENTION_DECODE_BATCH_MAX 32u
typedef struct {
    uint64_t raw_kv;
    uint64_t comp_kv;
    uint64_t topk;
    uint32_t pos;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t window;
    uint32_t ratio;
    uint32_t indexed;
} ds4_gpu_attention_decode_row;
#endif

int ds4_gpu_init(void);
void ds4_gpu_cleanup(void);
#if defined(__APPLE__)
/* Teardown only: stop Metal producers and drain queued commands without
 * initializing a backend. Keep caller-owned tensors and model maps alive
 * until this returns; cleanup then releases the backend's no-copy views. */
void ds4_gpu_prepare_cleanup(void);
/* Metal: the V4.1 decode fast paths, each with one opt-out, DS4_METAL_DISABLE_V41_<NAME>, read once into this mask;
 * "0" means not set. */
enum {
    DS4_GPU_V41_DEFERRED_HC = 1u << 0,
    DS4_GPU_V41_LITERAL_SHAPES = 1u << 1,
    DS4_GPU_V41_INDEXER_E4M3 = 1u << 2,
    DS4_GPU_V41_MODEL_TRACKED = 1u << 3,   /* opt-out: DS4_METAL_MODEL_TRACKED */
    DS4_GPU_V41_ROPE_TABLE = 1u << 4,
    DS4_GPU_V41_SPLIT_REDUCE = 1u << 5,
    DS4_GPU_V41_INDEX_BF16 = 1u << 6,
    DS4_GPU_V41_DECODE_TOPK = 1u << 7,
    DS4_GPU_V41_PLAIN_READERS = 1u << 29,   /* test only: the narrow readers' own kernels */
    DS4_GPU_V41_UNFUSED_TOPK = 1u << 30,    /* test only: the decode top-k without its fused merge */
};
uint32_t ds4_gpu_v41_off(void);
/* Weights built at load: allocated with the model views' storage options, and kept in a residency set of their
 * own; ds4_gpu_weight_residency replaces that set, n == 0 releases it. */
ds4_gpu_tensor *ds4_gpu_tensor_alloc_weights(uint64_t bytes);
/* Model views and weights built at load skip hazard tracking (call before the model is mapped). */
void ds4_gpu_set_model_untracked_default(int on);
int ds4_gpu_weight_residency(ds4_gpu_tensor *const *tensors, uint32_t n);
#endif

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);
int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count);
int ds4_gpu_moe_handoff_pack_tensor(
        ds4_gpu_tensor       *packed,
        const ds4_gpu_tensor *ffn_norm,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_embd,
        uint32_t              n_expert);
int ds4_gpu_pack_slot_rows_f32_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *slots,
        uint32_t                n_rows,
        uint32_t                width,
        uint32_t                n_slots,
        uint32_t                slot_cap);

int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_encoder(void);
int ds4_gpu_flush_commands(void);
int ds4_gpu_commands_active(void);
#include "ds4_deepseek41_gpu.h"
#ifdef __APPLE__
int ds4_gpu_parallel_ffn_finish(void);
void ds4_gpu_parallel_ffn_abort(void);
int ds4_gpu_parallel_ffn_start(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        const ds4_gpu_tensor *x,
        float                 clamp);
int ds4_gpu_parallel_ffn_start_bf16(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        const ds4_gpu_tensor *x,
        float                 clamp);
/* A TP rank's lanes of the V4.1 shared expert beside its routed experts:
 * BF16 mid, then the K slice of the down projection with F32 output. */
int ds4_gpu_parallel_ffn_start_bf16_kslice(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        uint32_t              shared_lane_offset,
        uint32_t              shared_lane_count,
        const ds4_gpu_tensor *x,
        float                 clamp);
int ds4_gpu_parallel_ffn_start_sliced(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        uint32_t              shared_lane_offset,
        uint32_t              shared_lane_count,
        const ds4_gpu_tensor *x,
        float                 clamp);

/* GPU-decided shared-expert lane split for two-rank TP decode: the split
 * kernels read the selected expert ids and take complementary lane ranges
 * sized to balance the bytes each rank streams (shift_q16 = routed expert
 * bytes / (2 * shared expert bytes) in Q16; 0 reproduces static halves). */
int ds4_gpu_parallel_ffn_start_split(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        const ds4_gpu_tensor *x,
        float                 clamp,
        const ds4_gpu_tensor *selected,
        uint32_t              tp_rank,
        uint32_t              tp_world,
        uint32_t              n_expert,
        uint32_t              n_expert_used,
        uint32_t              shift_q16);

/* out = a + b into this rank's TP slab slot for (layer, gate), publishing the
 * gate's checked flag from the same kernel; falls back to ds4_gpu_add_tensor
 * when the fold does not apply.  Call right before ds4_gpu_tp_gate_encode. */
int ds4_gpu_add_tensor_tp_flag(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t              n,
        uint32_t              layer,
        uint32_t              gate);

/* Register that the next TP partial producer for (layer, gate) may publish
 * the gate's checked flag itself (taken by the attention output K-slice
 * matvec when its output is that slot; otherwise ignored). */
void ds4_gpu_tp_flag_fold_request(uint32_t layer, uint32_t gate);
/* Bind every routed expert and mask none (the caller splits the list). */
void ds4_gpu_tp_full_expert_bind(int on);

/* Deferred kv norm task: call before ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor
 * to run only its q task now and fold the kv task into the KV staging
 * kernel of the same layer; flush runs it standalone if nothing consumed it. */
void ds4_gpu_dsv4_qkv_norm_defer_kv_next(void);
int ds4_gpu_kv_norm_task_pending(void);
int ds4_gpu_kv_norm_task_flush(void);
int ds4_gpu_kv_norm_task_begin_concurrent(void);
void ds4_gpu_kv_norm_task_end_concurrent(void);
#endif
int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value);
int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label);
int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label);
#ifdef DS4_ROCM_BUILD
int ds4_gpu_tensor_read_after_selected_event(const ds4_gpu_tensor *tensor,
                                             uint64_t offset,
                                             void *data,
                                             uint64_t bytes,
                                             uint64_t event_value,
                                             const char *label);
#endif
int ds4_gpu_end_commands(void);
int ds4_gpu_synchronize(void);

#ifdef __APPLE__
/* Diagnostic-only census window.  Requires DS4_KERNEL_LEDGER=1; begin drops
 * all earlier counts and dump writes the current counters to `path`. */
int ds4_gpu_kernel_ledger_window_begin(void);
int ds4_gpu_kernel_ledger_window_dump(const char *path);
/* Verifier experiments: process-global selectors are changed only around one
 * serialized target forward and restored immediately afterwards. */
void ds4_gpu_glm53_indexer_small_tiled_set(int enabled);
int ds4_gpu_glm53_indexer_small_tiled_get(void);
void ds4_gpu_glm53_bf16_mv_max_set(uint32_t v);
uint32_t ds4_gpu_glm53_bf16_mv_max_get(void);
#endif

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size);
/* Resolve mmap-backed Metal model hazard tracking before the first view. */
void ds4_gpu_set_model_untracked(int enabled);
int ds4_gpu_set_model_fd(int fd);
int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map);
int ds4_gpu_build_derived_artifacts(const void *model_map, uint64_t model_size,
                                  const char *model_path);
/* Two-rank expert-only artifacts; never build or cache the unowned half.
 * file_size includes any disk-only tail outside the model mapping. */
int ds4_gpu_build_derived_artifacts_shard(const void *model_map, uint64_t model_size,
                                        uint64_t file_size,
                                        const char *model_path, uint32_t rank);
/* CUDA network TP: each rank keeps half the intermediate dimension of every
 * MXFP4 expert, read directly from the GGUF. Both ranks must use this layout.
 * The arrays describe count expert tensors; is_down selects K halves rather
 * than gate/up row halves. */
/* Returns the allocated expert bytes, or zero on failure. */
uint64_t ds4_gpu_build_mxfp4_tp_split(
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        file_size,
        const char     *model_path,
        uint32_t        rank,
        const uint64_t *offsets,
        const uint32_t *is_down,
        const uint32_t *in_dims,
        const uint32_t *out_dims,
        const uint32_t *n_experts,
        uint32_t        count);
void ds4_gpu_release_mxfp4_tp_split(void);
/* The same split for the lockstep DSpark drafter: IQ2_XXS gate/up and Q2_K
 * down, or native MXFP4 experts. types holds each tensor's type. */
uint64_t ds4_gpu_build_dspark_tp_split(
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        file_size,
        const char     *model_path,
        uint32_t        rank,
        const uint64_t *offsets,
        const uint32_t *is_down,
        const uint32_t *in_dims,
        const uint32_t *out_dims,
        const uint32_t *n_experts,
        const uint32_t *types,
        uint32_t        count);
void ds4_gpu_release_dspark_tp_split(void);

/* CUDA side stream (single GPU).  ds4_gpu_side_begin() returns 1 when the
 * following work is encoded on the side stream (ordered after all work
 * already queued), 0 when unavailable (encode it on the main stream as
 * usual) and -1 on error.  Every successful begin must be closed with
 * ds4_gpu_side_end(); ds4_gpu_side_join() then orders the main stream after
 * the side work and is a no-op once joined.  Not implemented on Metal/ROCm. */
int ds4_gpu_side_begin(void);
int ds4_gpu_side_end(void);
int ds4_gpu_side_join(void);
/* 1 when dense Q8_0 matmuls may be encoded in a side region beside main-stream
 * Q8 work (no persistent activation scratch shared between the streams). */
int ds4_gpu_side_dense_q8_ok(void);

int ds4_gpu_model_range_replaced(const void *model_map, uint64_t offset,
                                 uint64_t bytes);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
/* Add a secondary GGUF mapping without replacing the primary model mapping. */
int ds4_gpu_set_aux_model_map_range(const void *model_map,
                                    uint64_t model_size,
                                    uint64_t map_offset,
                                    uint64_t map_size);
int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes);
int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label);
int ds4_gpu_q8_cache_suppressed(void);
void ds4_gpu_set_q8_cache_suppressed(int suppressed);
#ifdef DS4_ROCM_BUILD
void ds4_gpu_release_q8_f16_cache(void);
#endif

/* Model-file ranges assigned to CUDA devices by the multi-GPU placement
 * planner. Metal keeps these declarations for the shared engine interface. */
#ifndef DS4_MAX_GPUS
#define DS4_MAX_GPUS 16
#endif
typedef struct {
    uint64_t source_offset;
    uint64_t bytes;
    int target_device;
} ds4_tensor_range;

int ds4_gpu_device_cache_tensors(int device_id,
                                 const ds4_tensor_range *ranges,
                                 int n_ranges);
int ds4_gpu_register_support_map(const void *map, uint64_t size, uint64_t bias);
int ds4_gpu_device_cache_support_tensors(int device_id,
                                         int entry_device_id,
                                         const ds4_tensor_range *ranges,
                                         int n_ranges,
                                         int from_main_map);
uint64_t ds4_gpu_tier_free_vram(int logical_tier);
int ds4_gpu_lookup_cache(uint64_t source_offset, uint64_t bytes,
                         int *out_device_id, void **out_device_ptr);
int ds4_gpu_lookup_cache_device(uint64_t source_offset, uint64_t bytes);

int ds4_gpu_pro_q4_expert_table_auto_available(void);
int ds4_gpu_preload_q4_expert_tables(const void *model_map, uint64_t model_size,
                                     uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                                     uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
                                     uint32_t n_total_expert);
int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
void ds4_gpu_set_quality(bool quality);
void ds4_gpu_set_glm_model(bool enabled);
void ds4_gpu_set_ssd_streaming(bool enabled);
void ds4_gpu_set_glm_streaming_prefill_full_layer(bool enabled);
#ifdef __APPLE__
int ds4_gpu_device_is_pre_m5_apple_silicon(void);
/* M6 shares the M5 dispatch defaults and existing M5 rollback switches. */
int ds4_gpu_device_is_m5_apple_silicon(void);
/* Read initialized host metadata; does not initialize or run Metal. */
int ds4_gpu_dflash_budget_profile_device(void);
int ds4_gpu_device_is_m5_or_m6_apple_silicon(void);
int ds4_gpu_set_decode_pipeline_fast_lookup(int enabled);
/* Strict test oracle for the fixed decode mul_mv pipeline lookup cache. */
int ds4_gpu_test_decode_pipeline_fast_lookup(void);
/* Strict test oracle for the extended decode mul_mv_ext (nsg + nxpsg) cache. */
int ds4_gpu_test_decode_pipeline_fast_lookup_ext(void);
/* Strict test oracle for the generated resident-prefill MXFP4 half LUT. */
int ds4_gpu_test_mxfp4_down_half_lut(uint16_t *legacy_bits,
                                     uint16_t *lut_bits);
enum {
    DS4_GPU_TEST_MXFP4_PAIR_TAIL_CULL = 1u << 0,
    DS4_GPU_TEST_MXFP4_PAIR_COMPACT_TILE = 1u << 1,
    DS4_GPU_TEST_MXFP4_MAP_SCATTER = 1u << 2,
    DS4_GPU_TEST_MXFP4_DOWN_TAIL_CULL = 1u << 3,
    DS4_GPU_TEST_MXFP4_DOWN_HALF_LUT = 1u << 4,
    DS4_GPU_TEST_OUTPUT_HC_WEIGHTS4 = 1u << 5,
    DS4_GPU_TEST_HC_RMS_SCALE_PROJ = 1u << 6,
    DS4_GPU_TEST_INDEX_SCORE_DIRECT = 1u << 7,
    DS4_GPU_TEST_INDEX_SCORE_WIDE = 1u << 8,
    DS4_GPU_TEST_COUNT_KERNELS = 1u << 9,
};
void ds4_gpu_test_set_flags(uint32_t flags);
/* With DS4_GPU_TEST_COUNT_KERNELS, each V4.1 fast path counts its dispatches by kernel name. */
uint64_t ds4_gpu_test_kernel_runs(const char *name);
/* Replaces the DS4_GPU_V41_* opt-out mask read from the environment. */
void ds4_gpu_test_set_v41_off(uint32_t off);
void ds4_gpu_release_zero_prefix_prefill_mask_cache(void);
#else
static inline int ds4_gpu_device_is_pre_m5_apple_silicon(void) { return 0; }
static inline int ds4_gpu_device_is_m5_apple_silicon(void) { return 0; }
static inline int ds4_gpu_device_is_m5_or_m6_apple_silicon(void) { return 0; }
#endif
void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts);
void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes);
int ds4_gpu_qwen4_conv_stream_rows_tensor(
        ds4_gpu_tensor *x, ds4_gpu_tensor *hist_pool,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        const uint32_t *slots, uint32_t n_rows, uint32_t n_channels,
        uint32_t conv_kernel, uint32_t state_stride, uint32_t x_stride, int apply_silu);
int ds4_gpu_qwen4_gdn_scan_rows_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *state_pool,
        const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *ga, const ds4_gpu_tensor *gb,
        const uint32_t *slots, uint32_t n_rows,
        uint32_t n_k_head, uint32_t n_v_head, uint32_t head_dim,
        uint32_t state_stride, uint32_t qkv_stride, uint32_t out_stride);
/* One row of a Qwen3.8 decode batch for the attention rows kernels: the
 * session's caches for the layer, its rope positions, its token position and
 * whether it attends through the block selection or densely. */
typedef struct {
    ds4_gpu_tensor *k_cache, *v_cache, *ik_cache, *block_key;
    const ds4_gpu_tensor *pos3;
    uint32_t pos;
    int use_sel;
} ds4_gpu_qwen4_attn_row;
#define DS4_GPU_QWEN4_ATTN_ROW_BYTES 64u   /* one staged table entry */
/* One session of a decode batch for the GDN rows kernels: its state and
 * history, optional snapshots of both after its first token, its first row
 * and its row count (one or two). */
typedef struct {
    ds4_gpu_tensor *state, *hist, *snap_state, *snap_hist;
    uint32_t row0, n_tok;
} ds4_gpu_qwen4_gdn_row;
#define DS4_GPU_QWEN4_GDN_ROW_BYTES 48u
int ds4_gpu_qwen4_gdn_rows_stage(ds4_gpu_tensor *table, uint64_t entry0,
                                 const ds4_gpu_qwen4_gdn_row *rows, uint32_t n_rows);
int ds4_gpu_qwen4_gdn_scan_rows2_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *ga, const ds4_gpu_tensor *gb,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_gdn_row *rows, uint32_t n_rows,
        uint32_t n_batch_rows, uint32_t n_k_head, uint32_t n_v_head, uint32_t head_dim,
        uint32_t qkv_stride, uint32_t out_stride);
int ds4_gpu_qwen4_conv_stream_rows2_tensor(
        ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_gdn_row *rows, uint32_t n_rows,
        uint32_t n_batch_rows, uint32_t n_channels, uint32_t conv_kernel, uint32_t x_stride, int apply_silu);
int ds4_gpu_qwen4_attn_rows_stage(ds4_gpu_tensor *table, uint64_t entry0,
                                  const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows, uint32_t ratio);
int ds4_gpu_qwen4_attn_prep_rows_tensor(
        ds4_gpu_tensor *q_out, ds4_gpu_tensor *gate_out, ds4_gpu_tensor *iq_out,
        const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kproj, const ds4_gpu_tensor *vproj,
        const ds4_gpu_tensor *iq, const ds4_gpu_tensor *ik,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        const void *model_map, uint64_t model_size,
        uint64_t g_q_offset, uint64_t g_k_offset, uint64_t g_iq_offset,
        uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t n_rot,
        uint32_t n_idx_head, uint32_t idx_dim, float rope_base, float eps);
int ds4_gpu_qwen4_idx_block_key_rows_tensor(
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        const void *model_map, uint64_t model_size, uint64_t g_ik_offset,
        uint32_t ratio, uint32_t idx_dim, uint32_t n_rot, float rope_base, float eps);
int ds4_gpu_qwen4_idx_score_rows_tensor(
        ds4_gpu_tensor *score, ds4_gpu_tensor *tile_max, const ds4_gpu_tensor *iq,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        uint32_t n_block_stride, uint32_t n_idx_head, uint32_t idx_dim, uint32_t ratio);
int ds4_gpu_qwen4_idx_select_rows_tensor(
        ds4_gpu_tensor *sel, const ds4_gpu_tensor *score, const ds4_gpu_tensor *tile_max,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        uint32_t n_block_stride, uint32_t top_k);
int ds4_gpu_qwen4_idx_expand_rows_tensor(
        ds4_gpu_tensor *sel_tokens, ds4_gpu_tensor *n_sel, const ds4_gpu_tensor *sel_blocks,
        const ds4_gpu_tensor *table, uint64_t entry0, uint32_t n_rows,
        uint32_t n_sel_blocks, uint32_t ratio, uint32_t sel_stride);
int ds4_gpu_qwen4_attn_decode_rows_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *sel_tokens, const ds4_gpu_tensor *n_sel, ds4_gpu_tensor *part,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t sel_stride, float scale);
int ds4_gpu_qwen4_batch_mm_q8_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_tokens, uint32_t in_dim, uint32_t out_rows);
uint64_t ds4_gpu_recommended_working_set_size(void);
uint32_t ds4_gpu_stream_expert_cache_configured_count(void);
uint32_t ds4_gpu_stream_expert_cache_current_count(void);
typedef struct ds4_gpu_stream_expert_table {
    const void *model_map;
    uint64_t    model_size;
    uint32_t    layer;
    uint32_t    n_total_expert;
    uint64_t    gate_offset;
    uint64_t    up_offset;
    uint64_t    down_offset;
    uint64_t    gate_expert_bytes;
    uint64_t    down_expert_bytes;
} ds4_gpu_stream_expert_table;
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
/* Optional CUDA look-ahead between completed layers, inside the existing
 * expert cache. The foreground owns slots; the reader cannot publish them
 * or evict the current layer's inputs. */
int ds4_gpu_stream_expert_cache_prefetch(
        const ds4_gpu_stream_expert_table *current,
        const ds4_gpu_stream_expert_table *next);
void ds4_gpu_stream_expert_cache_prefetch_finish(bool cancel);
#endif
/* Reset only the prompt-local eviction heuristic.  The resident SSD expert
 * cache itself is intentionally kept warm across sessions. */
void ds4_gpu_stream_expert_cache_reset_route_hotness(void);
void ds4_gpu_stream_expert_cache_release_resident(void);
uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes);
int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected);
int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected);
int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected,
        uint32_t                           n_selected);
#ifdef __APPLE__
/* The async selected-load worker registers itself so Metal cache paths never
 * wait on command buffers from that thread (they fail the load instead and
 * the caller retries synchronously). */
void ds4_gpu_stream_expert_cache_note_service_thread(void);
#endif
#if defined(DS4_ROCM_BUILD) || (!defined(DS4_NO_GPU) && !defined(__APPLE__))
int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected);
#endif
#ifdef DS4_ROCM_BUILD
int ds4_gpu_stream_expert_cache_load_layer(
        const ds4_gpu_stream_expert_table *table);
int ds4_gpu_stream_expert_cache_seed_from_layer_selected(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor             *selected,
        uint32_t                          n_tokens,
        uint32_t                          n_seed_tokens,
        uint32_t                          n_selected);
int ds4_gpu_stream_expert_cache_finish_pending_batch(void);
int ds4_gpu_stream_expert_cache_release_layer_cache(void);
#endif
int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts);
#ifdef __APPLE__
/* Seed from mapped weights with blits appended to the active command buffer. */
int ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts);
#endif
void ds4_gpu_print_memory_report(const char *label);

#include "ds4_gpu_tp.h"
/* Skip the whole-file model residency set (TP sharding: only the
 * owned ranges are warmed; the rest must never be paged in). Call before
 * the model is mapped. */
void ds4_gpu_model_residency_skip(int skip);
/* Submit one trivial command buffer (first-submission costs paid at load). */
int ds4_gpu_warm_command_queue(void);

/* Tensor-parallel sliced projections (Metal decode path only).
 *
 * ds4_gpu_matmul_q8_0_kslice_tensor computes a k-range partial matvec:
 * out[out_dim] = W[:, k_off : k_off + k_cnt] @ x[x_elem_off : +k_cnt] where
 * W rows span full_in_dim quantized Q8_0 elements.  k offsets/counts must be
 * multiples of 32 (Q8_0 block).  Partial results from both ranks sum to the
 * full projection.
 *
 * ds4_gpu_attention_output_q8_tp_tensor is the group-sliced attention output
 * pair: low projection for groups [group0, group0+group_cnt) plus the
 * matching k-slice of the expand projection, producing this rank's partial
 * attention block output (n_tokens == 1 only). */
int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off);
/* One row's kslice matvec over an input not yet rounded to BF16: rounded as
 * it loads. Returns 0 (nothing encoded) where only the rounded input works. */
int ds4_gpu_matmul_q8_0_kslice_bf16in_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t k_off,
        uint64_t k_cnt, uint64_t out_dim, const ds4_gpu_tensor *x);
/* CUDA input rows contain only the owned contiguous K slice. Metal input
 * rows span full_in_dim and the kernel reads the slice at k_off within them.
 * Both backends produce full-width output rows. */
int ds4_gpu_matmul_q8_0_kslice_rows_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              full_in_dim,
        uint64_t              out_dim,
        uint64_t              k_off,
        uint64_t              k_cnt,
        const ds4_gpu_tensor *x,
        uint64_t              n_rows);
int ds4_gpu_matmul_quant_kslice_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off);
int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads);

/* =========================================================================
 * Embeddings and Indexer Helpers.
 * =========================================================================
 *
 * These kernels seed HC state from token embeddings and implement the ratio-4
 * compressed-attention indexer that chooses visible compressed rows.
 */

int ds4_gpu_embed_token_hc_tensor(
        ds4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc);

int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_embed_token_q8_0_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd);

int ds4_gpu_embed_tokens_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd);

int ds4_gpu_embed_token_quant_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          weight_type,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd);

int ds4_gpu_embed_tokens_quant_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd);

int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale);

int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

int ds4_gpu_dspark_markov_argmax_tensor(ds4_gpu_tensor *out_idx,
                                        const ds4_gpu_tensor *logits_row,
                                        const void *model_map,
                                        uint64_t model_size,
                                        uint64_t w1_offset,
                                        uint64_t w2_offset,
                                        uint32_t prev_token,
                                        uint32_t vocab,
                                        uint32_t rank);
/* CUDA TP drafter: score only rows [row0, row0 + rows) of the vocab; the
 * logits tensor holds those rows and the key carries global token ids. */
int ds4_gpu_dspark_markov_argmax_range_tensor(ds4_gpu_tensor *out_idx,
                                              const ds4_gpu_tensor *logits_row,
                                              const void *model_map,
                                              uint64_t model_size,
                                              uint64_t w1_offset,
                                              uint64_t w2_offset,
                                              uint32_t prev_token,
                                              uint32_t vocab,
                                              uint32_t row0,
                                              uint32_t rows,
                                              uint32_t rank);
int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* Bench/identity hook for the GLM DSA indexer fast paths (Metal only).
 * Each argument: -1 keep current, 0 force the legacy kernel chain, 1 force the
 * fused kernel.  Production code never calls this; the shipping default comes
 * from DS4_GLM_DISABLE_INDEXER_SCORE_STREAM / DS4_GLM_DISABLE_TOPK_ONESHOT. */
void ds4_gpu_glm_indexer_select_override(int score_stream, int topk_oneshot);

int ds4_gpu_indexer_top1_value_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *values,
        const ds4_gpu_tensor *scores,
        uint32_t              n_comp,
        uint32_t              n_tokens,
        uint32_t              index_offset);

int ds4_gpu_matmul_q8_0_top1_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *values,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              index_offset);

int ds4_gpu_set_decode_fast_attention(int enabled);
int ds4_gpu_set_decode_score_vec4(int enabled);

/* GPU argmax over n_vocab F32 logits. Writes the winning index as int32 at
 * out_idx[0]. Tie-break: lower index wins (matches host sample_argmax). */
int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab);

int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* =========================================================================
 * Dense Projections, Norms, RoPE, and KV Rounding.
 * =========================================================================
 *
 * The graph uses these primitives for Q/KV projections, HC/output projections,
 * attention output projections, and DS4's tail-only RoPE.
 */

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);
#if !defined(DS4_ROCM_BUILD)
/* Qwen projections preserve FP32 activations on both GPU backends. */
int ds4_gpu_qwen4_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);
#else
#define ds4_gpu_qwen4_matmul_q8_0_tensor ds4_gpu_matmul_q8_0_tensor
#endif

int ds4_gpu_matmul_q8_0_decode_mpp_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* Optional fused GPU operations.
 *
 * These are acceleration hooks, not required backend primitives.  A backend
 * that does not provide the fused kernel must still define the symbol and
 * return 0.  Callers then use the portable sequence of required primitives.
 * Backends that return nonzero from a fused half-output operation must also
 * implement the matching half-input HC expansion helpers below.
 */
int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* Same shared-input Q8_0 pair, but over a flat concatenated row space instead
 * of a max-extent grid, so no threadgroup retires without work.  Requires both
 * output extents to be multiples of the matvec's rows per simdgroup; returns 0
 * without encoding anything when that does not hold, so callers can fall back
 * to the two standalone dispatches. */
int ds4_gpu_matmul_q8_0_pair_flat_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_offset,
        uint64_t              weight1_offset,
        uint64_t              in_dim,
        uint64_t              out0_dim,
        uint64_t              out1_dim,
        const ds4_gpu_tensor *x);

/* GLM-5.3 DSA only: the same flat Q8_0 pair (q_a + kv_a) with the indexer's
 * three BF16 projections over the same input row folded into the head of the
 * grid, one dispatch instead of three.  Returns 0 without encoding anything if
 * any shape or range check fails, so the caller can emit the production
 * ladder unchanged. */
int ds4_gpu_glm53_dsa_qakv_indexer_fold_tensor(
        ds4_gpu_tensor       *out_q_a,
        ds4_gpu_tensor       *out_kv_a,
        ds4_gpu_tensor       *out_indexer_k,
        ds4_gpu_tensor       *out_indexer_gate,
        ds4_gpu_tensor       *out_indexer_proj,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_a_offset,
        uint64_t              kv_a_offset,
        uint64_t              indexer_k_offset,
        uint64_t              indexer_gate_offset,
        uint64_t              indexer_proj_offset,
        uint64_t              in_dim,
        uint64_t              q_a_dim,
        uint64_t              kv_a_dim,
        uint64_t              indexer_dim,
        uint64_t              indexer_proj_dim,
        const ds4_gpu_tensor *x);

int ds4_gpu_matmul_q4_K_pair_decode_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_offset,
        uint64_t              weight1_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x);

/* Multi-row decode projections that preserve the one-row reduction order. */
int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);
/* Experimental DFlash vocabulary head: four tokens share Q8 loads while
 * retaining the scalar head's NSG8 K walk and per-token reduction tree.
 * Returns 0 without dispatch for shapes other than 4096 -> 154880, rows=8. */
int ds4_gpu_dflash_head_nt4_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_rows);
int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_offset,
        uint64_t              weight1_offset,
        uint64_t              in_dim,
        uint64_t              out0_dim,
        uint64_t              out1_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);

int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor       *out_h,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_router_shared_gate_up_q8_0_tensor(
        ds4_gpu_tensor       *router_logits,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              router_weight_offset,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              in_dim,
        uint64_t              router_out_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        float                 clamp,
        bool                  router_only);
#ifdef __APPLE__
int ds4_gpu_router_project_select_fused_tensor(
        ds4_gpu_tensor       *router_logits,
        ds4_gpu_tensor       *probs,
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              router_weight_offset,
        uint64_t              bias_offset,
        bool                  has_bias,
        const ds4_gpu_tensor *x);
#endif
int ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *prequant,
        uint32_t                expert_split,
        bool                    home_rank);

int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp);

int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* CUDA batch path: fold an input RMS normalization into the FP16 activation
 * conversion used by the following projection. Returns 0 without touching
 * out when the optimized path is unavailable. */
int ds4_gpu_matmul_f16_rms_fold_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   norm_eps);

/* Exact multi-row form of the DeepSeek 4096x256 F16 router projection. */
int ds4_gpu_matmul_f16_router_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        const ds4_gpu_tensor *x,
        uint32_t                n_rows);

int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* Optional Metal decode fusion. Returns 1 when the paired projection and
 * recurrent compressor-state store were encoded, 0 when the optimized path
 * is unavailable, and -1 on an attempted-path error. */
int ds4_gpu_matmul_f16_pair_compressor_store_tensor(
        ds4_gpu_tensor       *out_kv,
        ds4_gpu_tensor       *out_score,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_kv_offset,
        uint64_t                weight_score_offset,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                in_dim,
        uint32_t                width,
        const ds4_gpu_tensor *x,
        uint32_t                ratio,
        uint32_t                pos);

int ds4_gpu_matmul_f16_quad_compressor_store_tensor(
        ds4_gpu_tensor       *out0_kv,
        ds4_gpu_tensor       *out0_score,
        ds4_gpu_tensor       *out1_kv,
        ds4_gpu_tensor       *out1_score,
        ds4_gpu_tensor       *state0_kv,
        ds4_gpu_tensor       *state0_score,
        ds4_gpu_tensor       *state1_kv,
        ds4_gpu_tensor       *state1_score,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_kv_offset,
        uint64_t              weight0_score_offset,
        uint64_t              weight1_kv_offset,
        uint64_t              weight1_score_offset,
        uint64_t              ape0_offset,
        uint32_t              ape0_type,
        uint64_t              ape1_offset,
        uint32_t              ape1_type,
        uint64_t              in_dim,
        uint32_t              width0,
        uint32_t              width1,
        const ds4_gpu_tensor *x,
        uint32_t              ratio,
        uint32_t              pos);

/* Decode-only M5 fusion: emit-path compressor row finalize (norm + rope +
 * fp8/commit + indexer qat) in one dispatch.  Bit-exact vs the separate
 * dispatches.  Returns 1 when fused, 0 to fall back. */
int ds4_gpu_dsv4_comp_row_finalize_tensor(
        ds4_gpu_tensor       *attn_stage,
        ds4_gpu_tensor       *attn_cache,
        uint32_t              attn_comp_row,
        uint64_t              attn_norm_offset,
        ds4_gpu_tensor       *index_cache,
        uint32_t              index_comp_row,
        uint64_t              index_norm_offset,
        ds4_gpu_tensor       *attn_state_kv,
        ds4_gpu_tensor       *attn_state_score,
        ds4_gpu_tensor       *index_state_kv,
        ds4_gpu_tensor       *index_state_score,
        const void           *model_map,
        uint64_t              model_size,
        uint32_t              pos,
        uint32_t              n_rot,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 rms_eps);

/* Decode-only M5 fusion: q_a/kv Q8 pair projection + F16 quad compressor
 * projection/store in one dispatch.  Bit-exact vs the separate dispatches.
 * Returns 1 when fused, 0 to fall back, -1 on error. */
int ds4_gpu_qkv_pair_quad_compressor_store_tensor(
        ds4_gpu_tensor       *qr,
        ds4_gpu_tensor       *kv_raw,
        ds4_gpu_tensor       *out0_kv,
        ds4_gpu_tensor       *out0_score,
        ds4_gpu_tensor       *out1_kv,
        ds4_gpu_tensor       *out1_score,
        ds4_gpu_tensor       *state0_kv,
        ds4_gpu_tensor       *state0_score,
        ds4_gpu_tensor       *state1_kv,
        ds4_gpu_tensor       *state1_score,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_a_offset,
        uint64_t              kv_offset,
        uint64_t              weight0_kv_offset,
        uint64_t              weight0_score_offset,
        uint64_t              weight1_kv_offset,
        uint64_t              weight1_score_offset,
        uint64_t              ape0_offset,
        uint32_t              ape0_type,
        uint64_t              ape1_offset,
        uint32_t              ape1_type,
        uint32_t              in_dim,
        uint32_t              q_rank,
        uint32_t              kv_dim,
        uint32_t              width0,
        uint32_t              width1,
        const ds4_gpu_tensor *x,
        uint32_t              ratio,
        uint32_t              pos);

int ds4_gpu_matmul_e4m3_bf16_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *codes,
                                    const ds4_gpu_tensor *scales, uint64_t in_dim, uint64_t out_dim,
                                    const ds4_gpu_tensor *x);
int ds4_gpu_matmul_bf16_weights_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *weights,
                                       uint64_t in_dim, uint64_t out_dim,
                                       const ds4_gpu_tensor *x);
/* step 91: K3 spine matmuls over several rows, each row bit-identical to the single-row call (1 done, 0 failed,
   -1 not applicable: loop the single-row call instead) */
int ds4_gpu_k3_matmul_q8_0_rows_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset,
                                       uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok);
int ds4_gpu_k3_matmul_f32_rows_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset,
                                      uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok);

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);
/* The same over n_tok rows, with the kernel chosen as for plan_tok rows
 * (plan_tok >= n_tok), so each row's result matches the larger dispatch. */
int ds4_gpu_matmul_f32_plan_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        uint64_t                plan_tok);

int ds4_gpu_repeat_hc_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_repeat_hc_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *rows,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_rms_norm_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_plain_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_rms_norm_weight_bf16_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);
int ds4_gpu_matmul_q8_0_bf16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);
int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t              kv_weight_offset,
        uint32_t              kv_n,
        ds4_gpu_tensor       *raw_cache,
        uint64_t              raw_cap,
        uint32_t              raw_row,
        uint32_t              n_rot,
        uint32_t              pos0,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 eps);

int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        uint32_t                kv_n_head,
        uint32_t                kv_head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        bool                    inverse,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   eps);

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps);

int ds4_gpu_head_rms_norm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow,
        float             eps);

int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *q_half,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tok,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              n_rot,
        uint32_t              pos0,
        uint32_t              n_ctx_orig,
        bool                  inverse,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 eps);

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot);

int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim);



int ds4_gpu_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow);

int ds4_gpu_glm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow);

int ds4_gpu_glm_kv_lora_rms_norm_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_raw,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        float                 eps);

int ds4_gpu_glm_k_b_project_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_norm,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              n_head);

int ds4_gpu_glm_k_b_project_typed_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_norm,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              n_head);

int ds4_gpu_glm_store_compact_kv_tensor(
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_norm,
        const ds4_gpu_tensor *kv_raw,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16);

int ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_raw,
        uint64_t              kv_weight_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16,
        float                 eps);

int ds4_gpu_glm_store_indexer_k_tensor(
        ds4_gpu_tensor       *indexer_key_cache,
        const ds4_gpu_tensor *raw_k,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              bias_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              n_ctx_orig,
        float                 eps,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16);

/* GLM-5.3 pools four normalized indexer keys with a learned, per-channel
 * softmax. Partial pools are retained in tail_k/tail_gate across calls. */
int ds4_gpu_glm53_indexer_pool_update_tensor(
        ds4_gpu_tensor       *pool_cache,
        ds4_gpu_tensor       *tail_k,
        ds4_gpu_tensor       *tail_gate,
        const ds4_gpu_tensor *raw_k,
        const ds4_gpu_tensor *gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              norm_weight_offset,
        uint64_t              norm_bias_offset,
        uint64_t              ape_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              pool_size,
        float                 eps,
        bool                  cache_f16);

int ds4_gpu_glm53_expand_pool_selection_tensor(
        ds4_gpu_tensor       *raw_selected,
        const ds4_gpu_tensor *pool_selected,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              selected_pools,
        uint32_t              index_topk,
        uint32_t              pool_size,
        uint32_t              output_width);

/* GLM-5.3 DSA decode selection with the pool expansion folded into the final
 * merge dispatch.  Returns 0 when the fused path is unavailable, in which case
 * the caller runs ds4_gpu_indexer_topk_tensor + the expansion separately. */
int ds4_gpu_glm53_indexer_topk_expand_tensor(
        ds4_gpu_tensor       *raw_selected,
        ds4_gpu_tensor       *pool_selected,
        const ds4_gpu_tensor *scores,
        uint32_t              n_comp,
        uint32_t              n_tokens,
        uint32_t              selected_pools,
        uint32_t              pos0,
        uint32_t              index_topk,
        uint32_t              pool_size,
        uint32_t              output_width);

int ds4_gpu_glm_build_kv_cache_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16);

int ds4_gpu_glm_build_kv_cache_flash_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16);

int ds4_gpu_glm_attention_full_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16);

int ds4_gpu_glm_fill_selected_range_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_selected);

int ds4_gpu_glm_fill_selected_range_batch_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_tokens,
        uint32_t        pos0,
        uint32_t        n_selected,
        uint32_t        pad_row);

int ds4_gpu_glm_indexer_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow);

/* Scores GPU-selected eight-key blocks into the original cache positions.
 * Other score slots remain untouched until the candidate mask is applied. */
int ds4_gpu_dsv41_indexer_score_blocks(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys,
        const ds4_gpu_tensor *blocks, uint32_t visible_rows, uint32_t block_count);

int ds4_gpu_glm_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16);

int ds4_gpu_glm_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16);

int ds4_gpu_glm53_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              pool_size,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16);

int ds4_gpu_glm_qk_lowrank_q8_0_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_qk_lowrank_typed_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim);

int ds4_gpu_glm_value_project_typed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim);

int ds4_gpu_glm_attention_indexed_decode_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_rope_tail_decode_rows_tensor(
        ds4_gpu_tensor                     *x,
        const ds4_gpu_attention_decode_row *rows,
        uint32_t                            n_rows,
        uint32_t                            n_head,
        uint32_t                            head_dim,
        uint32_t                            n_rot,
        uint32_t                            n_ctx_orig,
        bool                                inverse,
        float                               freq_base,
        float                               freq_scale,
        float                               ext_factor,
        float                               attn_factor,
        float                               beta_fast,
        float                               beta_slow);

int ds4_gpu_glm_attention_indexed_decode_exact_typed_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *scores,
        ds4_gpu_tensor       *lora,
        ds4_gpu_tensor       *denom,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim);

int ds4_gpu_glm_attention_indexed_decode_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

/* guaranteed_prefix: when selected_rows_valid is false, the number of leading
 * selected slots that are known to index live cache rows (0 = no such claim,
 * every row bounds-tested).  Ignored when selected_rows_valid is true. */
int ds4_gpu_glm_attention_indexed_decode_split_group8_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *partial_lora,
        ds4_gpu_tensor       *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        bool                  selected_rows_valid,
        uint32_t              guaranteed_prefix,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        uint32_t              block_rows,
        uint32_t              n_blocks,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *partial_lora,
        ds4_gpu_tensor       *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        bool                  selected_rows_valid,
        uint32_t              guaranteed_prefix,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        uint32_t              block_rows,
        uint32_t              n_blocks,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_batch_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_batch_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_sort_i32_rows_asc_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        uint32_t              row_width,
        uint32_t              n_rows);

int ds4_gpu_glm_attention_indexed_batch_lora_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

/* Dense causal MLA over the shared compact latent cache. qk_low and lora_out
 * are [token, head, kv_lora_dim]; the F16 cache is shared by all heads. */
int ds4_gpu_glm_attention_dense_compact_lora_causal_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        uint32_t              q_row0,
        uint32_t              n_q,
        uint32_t              n_kv,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_dim);

/* Selected rows [0, guaranteed_prefix) are known to index live cache rows;
 * slots at or above it may hold the pooled producer's 0xffffffff pad.  A
 * strictly weaker claim than ..._valid_tensor's; backends that have no kernel
 * able to exploit it must behave exactly like ..._tensor. */
int ds4_gpu_glm_attention_indexed_batch_lora_pooled_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              guaranteed_prefix,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_flash_staged_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16);

int ds4_gpu_glm_attention_flash_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16);

/* Release decode fused KV finalizer: after the standalone RoPE kernel, this
 * performs DS4's FP8 non-RoPE KV round trip and writes the F16-rounded raw
 * attention cache row in one dispatch. */
int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot);

/* Exact multi-session form of the decode KV finalizer. KV rows are
 * contiguous, while each output row is written to its session-private cache. */
int ds4_gpu_kv_fp8_store_raw_decode_rows_tensor(
        ds4_gpu_tensor        *kv,
        ds4_gpu_tensor *const *raw_caches,
        const uint32_t        *raw_caps,
        const uint32_t        *raw_rows,
        uint32_t               n_rows,
        uint32_t               head_dim,
        uint32_t               n_rot);

/* Reference/raw-cache primitive kept for prefill and diagnostics.  Decode uses
 * ds4_gpu_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
int ds4_gpu_store_raw_kv_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim);

int ds4_gpu_store_raw_kv_batch_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim);

/* =========================================================================
 * KV Compression and Attention.
 * =========================================================================
 *
 * Compressed layers maintain rolling score/KV state and append pooled rows at
 * ratio boundaries.  Attention kernels consume raw SWA rows, compressed rows,
 * and optional indexer masks.
 */

int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        bool                    state_already_stored,
        bool                    decode_one_token,
        bool                    defer_finalize);

int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens);

/* Batched replay of the per-row compressor update for a short block of rows
 * (n_rows <= 8; ratio 4, or a single-window ratio >= n_rows): stores every
 * row, pools emitted rows into comp_cache rows comp_row0.., applies the
 * ratio-4 shift, and writes snapshot t (state after row t) for t < n_snap.
 * Bit-identical to calling ds4_gpu_compressor_update_tensor row by row and
 * copying the state after each row; norm/rope/quantization of the pooled
 * rows is left to the caller.  CUDA only. */
int ds4_gpu_compressor_verify_rows_tensor(
        const ds4_gpu_tensor *kv_rows,
        const ds4_gpu_tensor *sc_rows,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        uint32_t                comp_row0,
        ds4_gpu_tensor       *snap_kv,
        ds4_gpu_tensor       *snap_score,
        uint32_t                n_snap,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_rows);

int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0);

int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_decode_heads_rope_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        int                    *fused_inv_rope);

/* Multi-session decode over contiguous Q/head rows and private KV caches.
 * The row table is copied into CUDA launch parameters, so no device-side
 * descriptor upload or synchronization is required. */
int ds4_gpu_attention_decode_rows_rope_tensor(
        ds4_gpu_tensor                       *heads,
        const void                           *model_map,
        uint64_t                              model_size,
        uint64_t                              sinks_offset,
        const ds4_gpu_tensor                 *q,
        const ds4_gpu_attention_decode_row   *rows,
        uint32_t                              n_rows,
        uint32_t                              n_head,
        uint32_t                              head_dim,
        uint32_t                              n_rot,
        uint32_t                              n_ctx_orig,
        float                                 freq_base,
        float                                 freq_scale,
        float                                 ext_factor,
        float                                 attn_factor,
        float                                 beta_fast,
        float                                 beta_slow);
/* Diagnostic/public form of the dk=512 gathered decode-attention KV staging
 * step. The compressed source must be F16; dst writes chronological raw-ring
 * rows followed by compressed rows and must not overlap either source. */
int ds4_gpu_flash_kv_stage_f16_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_raw,
        const ds4_gpu_tensor *comp,
        uint32_t                comp_is_f16,
        uint32_t                n_comp,
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_raw_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

/* Rectangular raw prefill attention: q is a view of the n_q query rows at
 * token positions [q_row0, q_row0 + n_q) of the chunk, raw_kv keeps all
 * n_kv rows, heads receives n_q output rows.  Used by the TP prefill row
 * split; the square entry above is the q_row0 = 0, n_q = n_kv case. */
int ds4_gpu_attention_prefill_raw_heads_range_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                q_row0,
        uint32_t                n_q,
        uint32_t                n_kv,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

/* Rectangular static-mixed prefill attention: q is a view of the n_q query
 * rows at token positions [q_row0, q_row0 + n_q) of the chunk, while raw_kv
 * keeps all n_tokens rows and comp_kv all n_comp compressed keys.  Used by
 * the TP prefill row split; the square entry above is q_row0 = 0,
 * n_q = n_tokens. */
int ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                q_row0,
        uint32_t                n_q,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

/* DeepSeek Vision-Exp attention over the current prefill chunk. The raw cache
 * is chronological from raw_start and may include the preceding SWA rows.
 * Synthetic image spans in tokens are made bidirectional as specified by the
 * checkpoint; text and compressed keys retain the normal causal masks. */
int ds4_gpu_attention_visual_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        const int32_t          *tokens,
        uint32_t                vocab_size,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);
int ds4_gpu_attention_output_q4_K_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint32_t                out_b_type,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor       *out_h,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads);
int ds4_gpu_attention_output_low_q4_K_slice_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads);

int ds4_gpu_attention_output_low_q8_rows_exact_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads,
        uint32_t                n_rows);

int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads);

/* =========================================================================
 * Router, Shared Expert, and Routed MoE.
 * =========================================================================
 *
 * These kernels implement the FFN body: router probabilities/top-k or hash
 * routing, shared SwiGLU, and the IQ2_XXS/Q2_K/Q4_K routed experts.
 */

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight);

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t                n);

int ds4_gpu_add3_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *c,
        uint32_t                n);

int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale);

int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits);

int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

/* DeepSeek Vision-Exp prefill may mix ordinary vocabulary IDs and synthetic
 * image IDs in one batch. Text rows keep the normal/hash route; image rows use
 * the checkpoint's visual selection bias. Routing weights always come from
 * the original, unbiased scores. */
int ds4_gpu_router_select_batch_visual_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        bool                    has_bias,
        bool                    hash_mode,
        const void             *vision_map,
        uint64_t                vision_size,
        uint64_t                visual_bias_offset,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                vocab_size,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

int ds4_gpu_glm_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale);

/* Decode router: the logits matvec with the top-k selection folded onto its
 * tail, in one dispatch. `counter` is a 4-byte device tensor the caller zeroes
 * once; the electing threadgroup resets it, so it stays zero between
 * dispatches. Returns 0 without encoding anything when the shape does not fit
 * the fused launch, so callers fall back to the two separate primitives. */
int ds4_gpu_glm_router_logits_select_tail_tensor(
        ds4_gpu_tensor       *logits,
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        ds4_gpu_tensor       *counter,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              bias_offset,
        uint64_t              in_dim,
        uint32_t              n_expert,
        uint32_t              n_expert_used,
        float                 expert_weight_scale,
        const ds4_gpu_tensor *x);

/* The GLM-5.3 decode router folded into the HEAD of the shared expert's
 * gate+up grid: ONE dispatch of (n_expert/2 + n_ff_exp/4) threadgroups of 256
 * threads replaces the router logits+select tail fold and the shared expert's
 * fused gate/up SwiGLU.  The ticket is counted over the router threadgroups
 * only, so the elected threadgroup's selection tail runs under the shared
 * expert's weight stream.  Bit-exact with the two dispatches it replaces.
 * `counter` is the same 4-byte tensor the router tail fold uses (zeroed once
 * by the caller; the electing threadgroup re-arms it) and `shared_mid` is a
 * separate n_ff_exp-float tensor, NOT `ffn_mid`, because the fold puts the
 * shared gate/up in front of the routed gate+up.  Returns 0 without encoding
 * anything on any refusal, so the caller emits the two-dispatch ladder
 * unchanged.  Kill switch DS4_GLM_DISABLE_ROUTER_SHARED_FOLD=1. */
int ds4_gpu_glm_router_shared_gateup_fold_enabled(void);
int ds4_gpu_glm_router_shared_gateup_fold_tensor(
        ds4_gpu_tensor       *logits,
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        ds4_gpu_tensor       *counter,
        ds4_gpu_tensor       *shared_mid,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              router_weight_offset,
        uint64_t              router_bias_offset,
        uint64_t              sh_gate_offset,
        uint64_t              sh_up_offset,
        uint64_t              in_dim,
        uint32_t              n_ff_exp,
        uint32_t              n_expert,
        uint32_t              n_expert_used,
        float                 expert_weight_scale,
        float                 swiglu_clamp,
        const ds4_gpu_tensor *x);

/* The whole GLM-5.3 sparse FFN block -- router, shared expert gate/up and
 * down, the eight routed experts' gate/up and down, the per-slot sum, the
 * residual add and hc_expand4 -- in ONE persistent dispatch of
 * DS4_GLM_MOE_BLOCK_GRID (default 240) threadgroups of 256 threads, instead of
 * the five dependent dispatches.  Bit-exact with them.  `counters` is a
 * 672-word device tensor the caller zeroes once; the kernel re-arms it.
 * Returns 0 without encoding anything when anything does not fit, so the
 * caller emits the production ladder unchanged.  Opt-in:
 * DS4_GLM_ENABLE_MOE_BLOCK_DATAFLOW=1; kill switch
 * DS4_GLM_DISABLE_MOE_BLOCK_DATAFLOW=1 wins over it. */
int ds4_gpu_glm53_moe_block_dataflow_enabled(void);
int ds4_gpu_glm53_moe_block_dataflow_poisoned(const ds4_gpu_tensor *counters);
int ds4_gpu_glm53_moe_block_dataflow_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        ds4_gpu_tensor       *logits,
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        ds4_gpu_tensor       *counters,
        ds4_gpu_tensor       *shared_mid,
        ds4_gpu_tensor       *routed_mid,
        ds4_gpu_tensor       *routed_partials,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              router_weight_offset,
        uint64_t              router_bias_offset,
        uint64_t              sh_gate_offset,
        uint64_t              sh_up_offset,
        uint64_t              sh_down_offset,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              up_expert_bytes,
        uint64_t              up_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              n_embd,
        uint32_t              n_ff_exp,
        uint32_t              expert_mid_dim,
        uint32_t              n_expert,
        uint32_t              n_expert_used,
        uint32_t              n_hc,
        float                 expert_weight_scale,
        float                 swiglu_clamp,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split);

int ds4_gpu_glm_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

int ds4_gpu_glm_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   swiglu_clamp,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        bool                    force_resident,
        /* Expert-parallel routed down (item B).  When routed_partials is
         * non-NULL and the shape qualifies, the Q4_K down matvec runs one
         * expert slot per threadgroup and writes per-slot partials there
         * instead of the summed row into `out`; *used_split is set to 1 and the
         * caller must consume the partials with the slot-summing shared-down
         * epilogue.  Pass NULL / NULL for the unsplit behaviour. */
        ds4_gpu_tensor       *routed_partials,
        int                    *used_split);

/* --- GLM-5.3 expanded-expert one-layer bank (EXPERT-BANK-DESIGN.md rev 3) ---
 * One routed layer's three expert tensors dequantized once into a half image
 * in the routed GEMM's own A-stage order. Bit-exact with the packed path by
 * construction. The C driver resolves the automatic device/model/length
 * profile; DS4_GLM_DISABLE_EXPERT_BANK=1 kills it outright. */
int ds4_gpu_glm_expert_bank_profile_device(void);
int ds4_gpu_glm_expert_bank_enabled(void);
void ds4_gpu_glm_expert_bank_set_enabled(int enabled);
uint64_t ds4_gpu_glm_expert_bank_bytes(uint32_t expert_in_dim,
                                       uint32_t expert_mid_dim,
                                       uint32_t out_dim,
                                       uint32_t n_total_expert);
int ds4_gpu_glm_expert_bank_ensure(uint32_t expert_in_dim,
                                   uint32_t expert_mid_dim,
                                   uint32_t out_dim,
                                   uint32_t n_total_expert);
/* End-to-end expansion accounting.  The command time includes any VM page-in
 * or first destination touch paid while Metal executes the expansion. */
typedef struct {
    uint64_t allocation_count;
    uint64_t expansion_count;
    uint64_t capacity_bytes;
    uint64_t current_allocated_bytes;
    double   last_ensure_ms;
    double   last_model_view_ms;
    double   last_command_ms;
} ds4_gpu_glm_expert_bank_stats;
void ds4_gpu_glm_expert_bank_get_stats(ds4_gpu_glm_expert_bank_stats *stats);
/* Expands one layer into the bank and arms it.  Opens and finishes its own
 * command buffer, so it must not be called with one open. */
int ds4_gpu_glm_expert_bank_expand_layer(const void *model_map,
                                         uint64_t    model_size,
                                         uint64_t    gate_offset,
                                         uint64_t    up_offset,
                                         uint64_t    down_offset,
                                         uint32_t    gate_type,
                                         uint32_t    up_type,
                                         uint32_t    down_type,
                                         uint64_t    gate_expert_bytes,
                                         uint64_t    gate_row_bytes,
                                         uint64_t    up_expert_bytes,
                                         uint64_t    up_row_bytes,
                                         uint64_t    down_expert_bytes,
                                         uint64_t    down_row_bytes,
                                         uint32_t    expert_in_dim,
                                         uint32_t    expert_mid_dim,
                                         uint32_t    out_dim,
                                         uint32_t    n_total_expert,
                                         uint32_t    layer_index);
void ds4_gpu_glm_expert_bank_disarm(void);
void ds4_gpu_glm_expert_bank_free(void);
int ds4_gpu_glm_expert_bank_armed_layer(void);
/* Test-only: independent inverse-mapping check of one bank section
 * (0 gate, 1 up, 2 down) against a fresh dequantization of the packed rows. */
int ds4_gpu_glm_expert_bank_verify_section(const void *model_map,
                                           uint64_t    model_size,
                                           uint64_t    tensor_offset,
                                           uint64_t    expert_bytes,
                                           uint64_t    row_bytes,
                                           uint32_t    rows,
                                           uint32_t    ne00,
                                           uint32_t    n_total_expert,
                                           int         section,
                                           uint64_t    counters_out[4]);

int ds4_gpu_glm_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   swiglu_clamp,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride,
        bool                    force_resident);

int ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   swiglu_clamp,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride);

int ds4_gpu_routed_moe_set_selected_override(const int32_t *selected, uint32_t n_selected);
void ds4_gpu_set_glm_mtp_verify_mode(bool enabled);
#ifdef DS4_ROCM_BUILD
int ds4_gpu_dspark_gfx1151_fast_path(void);
void ds4_gpu_set_dspark_verify_mode(bool enabled);
#elif !defined(__APPLE__)
int ds4_gpu_device_is_spark(void);
/* Spark: the spans as batched launches on the decode stream, ordered like
 * the asynchronous tensor copies.  Returns 1 when queued; 0 when declined
 * without copying anything (another device, more than
 * DS4_GPU_COPY_SPANS_MAX spans, a span outside its tensors or over 4 GiB,
 * spans on different devices, or any destination overlapping another span):
 * the caller then copies them one by one; -1 on a launch error. */
int ds4_gpu_tensor_copy_spans(const ds4_gpu_copy_span *spans, uint32_t n);
#endif

int ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        uint64_t              in_start,
        uint64_t              in_count,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t              n_embd,
        uint32_t              n_hc);

int ds4_gpu_routed_moe_one_owned_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              gate_type,
        uint32_t              down_type,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_total_expert,
        uint32_t              n_expert,
        uint32_t              resident_expert_base,
        uint32_t              resident_expert_count,
        float                 clamp,
        const ds4_gpu_tensor *x,
        ds4_gpu_tensor       *down_output,
        bool                  pack_fixed3,
        ds4_gpu_tensor       *shared_prequant);

int ds4_gpu_routed_moe_batch_owned_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              gate_type,
        uint32_t              down_type,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        uint32_t              n_total_expert,
        uint32_t              n_expert,
        uint32_t              resident_expert_base,
        uint32_t              resident_expert_count,
        float                 clamp,
        const ds4_gpu_tensor *x,
        uint32_t              layer_index,
        uint32_t              n_tokens,
        bool                 *mid_is_f16);

int ds4_gpu_routed_moe_owned_slots_combine_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_slots,
        const ds4_gpu_tensor *selected,
        uint32_t              out_dim,
        uint32_t              expert_split);

int ds4_gpu_routed_moe_owned_slots_combine_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_slots,
        const ds4_gpu_tensor *selected,
        uint32_t              out_dim,
        uint32_t              expert_split,
        uint32_t              rows);

int ds4_gpu_routed_moe_owned_packed_combine_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_packed,
        const ds4_gpu_tensor *selected,
        uint32_t              out_dim,
        uint32_t              expert_split);

int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *add_in,
        uint32_t                layer_index,
        bool                    force_resident);

int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16,
        bool                    force_resident);

/* =========================================================================
 * Hyper-Connection Kernels.
 * =========================================================================
 *
 * HC kernels reduce four residual streams before a sublayer and expand the
 * sublayer output back into four streams afterward.
 */

int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_weighted_sum_bf16_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc);
int ds4_gpu_matmul_q8_0_bf16io_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        ds4_gpu_tensor       *x,
        uint64_t                n_tok);
int ds4_gpu_hc_expand_split_add_bf16_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);
int ds4_gpu_hc_expand_split_bf16_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);
int ds4_gpu_hc_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_weighted_sum_split_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Release decode fused HC pre-sublayer operation: split the HC mixer and
 * immediately reduce four HC streams into the active 4096-wide sublayer row. */
int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps);

#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
/* Spark: complete HC pre boundary from the raw F32 HC state for 1..8 rows:
 * F16 mixer with the RMS factor applied after it, split, weighted sum and
 * output norm.  Deterministic, but not bit-identical to RMSNorm + F16 mixer +
 * split.  scratch (>= 26 * n_hc * n_embd / 256 floats per row, e.g. the
 * graph's HC RMS buffer) is consumed within the call; its contents are
 * undefined after.  Outputs must not overlap the state, the scratch or each
 * other, except norm_out == out.  Returns 1 when launched, 0 when unsupported
 * or disabled (DS4_CUDA_DISABLE_HC_PRE_FUSED, DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED,
 * DS4_CUDA_SERIAL_F16_MATMUL, DS4_CUDA_NO_F16_CUBLAS_BATCH; outputs
 * untouched), -1 on a launch failure. */
int ds4_gpu_hc_pre_f32_mix_split_norm_tensor(
        ds4_gpu_tensor       *mix,
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        ds4_gpu_tensor       *scratch,
        const ds4_gpu_tensor *residual_hc,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              mix_weight_offset,
        uint64_t              scale_offset,
        uint64_t              base_offset,
        uint64_t              norm_weight_offset,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              n_rows,
        uint32_t              sinkhorn_iters,
        float                 eps,
        float                 hc_eps,
        float                 norm_eps);
#endif

int ds4_gpu_hc_rms_norm_mix_f16_available(void);
int ds4_gpu_hc_rms_norm_mix_f16_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n,
        uint32_t              out_dim,
        float                 eps);

/* Batched HC RMSNorm followed by its narrow F16 mixer projection. On the
 * tuned Metal path, scale_scratch stores one float per row instead of the
 * full normalized HC tensor; other shapes retain the established fallback. */
int ds4_gpu_hc_rms_scale_project_f16_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *scale_scratch,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                in_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *x,
        uint32_t                n_rows,
        float                   eps);

#ifdef __APPLE__
int ds4_gpu_hc_rms_norm_mix_split_norm_f16_tensor(
        ds4_gpu_tensor       *mix,
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *residual_hc,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              mix_weight_offset,
        uint64_t              scale_offset,
        uint64_t              base_offset,
        uint64_t              norm_weight_offset,
        uint32_t              n,
        uint32_t              mix_dim,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              sinkhorn_iters,
        float                 eps,
        float                 hc_eps,
        float                 norm_eps);
int ds4_gpu_hc_expand_add_rms_norm_mix_split_norm_f16_tensor(
        ds4_gpu_tensor       *mix,
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_prev,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              mix_weight_offset,
        uint64_t              scale_offset,
        uint64_t              base_offset,
        uint64_t              norm_weight_offset,
        uint32_t              n,
        uint32_t              mix_dim,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              sinkhorn_iters,
        float                 eps,
        float                 hc_eps,
        float                 norm_eps);

/* Whole decode HC-pre (RMSNorm + HC-mix matvec + gated four-stream collapse +
 * output RMSNorm) in a single 1024-thread threadgroup. weight_bf16 picks the
 * BF16 mixer flavour instead of F16. Bit-identical to the dispatch ladder it
 * replaces in both flavours; single-row decode shapes only. */
int ds4_gpu_hc_pre_decode_fused_available(int weight_bf16);
int ds4_gpu_hc_pre_decode_fused_tensor(
        ds4_gpu_tensor       *mix,
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *residual_hc,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              mix_weight_offset,
        uint64_t              scale_offset,
        uint64_t              base_offset,
        uint64_t              norm_weight_offset,
        uint32_t              n,
        uint32_t              mix_dim,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              sinkhorn_iters,
        float                 eps,
        float                 hc_eps,
        float                 norm_eps,
        int                   weight_bf16);

#endif
int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps);

int ds4_gpu_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);
int ds4_gpu_hc_expand_add_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);


int ds4_gpu_hc_expand_add_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Prefill lever 28, pass SCALE: the split-form HC expands that also publish
 * the following RMSNorm's per-row scale.  Return 0 without encoding anything
 * when the fold is switched off or the shape does not qualify, so the caller
 * can fall back to the plain form. */
int ds4_gpu_hc_expand_split_scale_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        ds4_gpu_tensor       *scale_out,
        float                   norm_eps,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_add_split_scale_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        ds4_gpu_tensor       *scale_out,
        float                   norm_eps,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_chain_scale_enabled(void);
int ds4_gpu_hc_chain_collapse_enabled(void);

int ds4_gpu_rms_norm_scale_rows_tensor(
        ds4_gpu_tensor       *scale_out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_glm53_matmul_bf16_scaled(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                in_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *scales,
        uint32_t                n_rows);

int ds4_gpu_hc_expand_split_half_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out_h,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_add_split_half_add_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add_h,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        /* Non-NULL when the routed down ran expert-parallel: the epilogue then
         * sums n_slots per-expert partials at each row, in ascending slot
         * order, instead of reading routed_out. */
        const ds4_gpu_tensor *routed_partials,
        uint32_t                n_slots);

/* Opt-in DFlash capture sibling of the routed-slot path above.  It writes the
 * ordinary HC result unchanged and also collapses its four just-produced F32
 * streams with mean_weights into capture_out.  The caller keeps the ordinary
 * path as the fallback and reference. */
int ds4_gpu_shared_down_hc_expand_capture_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const ds4_gpu_tensor *routed_partials,
        uint32_t                n_slots,
        ds4_gpu_tensor       *capture_out,
        const ds4_gpu_tensor *mean_weights);

int ds4_gpu_shared_down_hc_expand_add_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *routed_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_packed,
        const ds4_gpu_tensor *selected,
        uint32_t                expert_split,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_glm53_embedding_bf16(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        const ds4_gpu_tensor *token_ids,
        uint32_t              n_tokens,
        uint32_t              n_embd,
        uint32_t              n_vocab);

int ds4_gpu_glm53_matmul_bf16(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);

/* Inner-dimension slices used by ds4_gpu_glm53_matmul_bf16_splitk. The caller
 * sizes the partials scratch as out_dim * n_rows * this many floats. */
#define DS4_GLM53_BF16_SPLITK_SLICES 8u

/* The decode HC-pre pair (ds4_gpu_glm53_hc_pre_splitk_fused_tensor) can run its
 * split-K matvec at a slice count other than the one above, selected at run
 * time by DS4_GLM_HC_PRE_SLICES; see ds4_gpu_glm53_hc_pre_slices() in
 * ds4_metal.m. The partials scratch is therefore sized for the widest legal
 * count, which costs 24 * 32 floats = 3 KB. */
#define DS4_GLM53_HC_PRE_SLICES_MAX 32u

/* Extra floats the caller must leave past the end of the partials array for
 * the hc_pre algebra lever's per-slice sums of squares (half A publishes one
 * per split-K slice at partials[out_dim*n_slices*n_rows + slice]; the tail
 * reads all 32 slots with one load per lane of one simdgroup). */
#define DS4_GLM53_HC_PRE_SUMSQ_SLOTS 32u

int ds4_gpu_glm53_matmul_bf16_splitk(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *partials,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);

/* Whole decode HC-pre with the split-K HC mixer in two dispatches: RMSNorm +
 * split-K partials, then split-K reduce + gated four-stream collapse + output
 * RMSNorm. Bit-identical to the four-dispatch ladder it replaces
 * (ds4_gpu_rms_norm_plain_tensor + ds4_gpu_glm53_matmul_bf16_splitk +
 * ds4_gpu_hc_split_weighted_sum_norm_tensor), including the split-K partials
 * themselves. Single-row decode shapes only. */
int ds4_gpu_glm53_hc_pre_splitk_fused_available(void);
int ds4_gpu_glm53_hc_pre_splitk_single_available(void);
int ds4_gpu_glm53_hc_pre_splitk_single_tensor(
        ds4_gpu_tensor       *mix,
        ds4_gpu_tensor       *partials,
        ds4_gpu_tensor       *ticket,
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *residual_hc,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              mix_weight_offset,
        uint64_t              scale_offset,
        uint64_t              base_offset,
        uint64_t              norm_weight_offset,
        uint32_t              n,
        uint32_t              mix_dim,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              sinkhorn_iters,
        float                 eps,
        float                 hc_eps,
        float                 norm_eps);
int ds4_gpu_glm53_hc_pre_splitk_fused_tensor(
        ds4_gpu_tensor       *mix,
        ds4_gpu_tensor       *partials,
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        ds4_gpu_tensor       *tail_counters,
        const ds4_gpu_tensor *residual_hc,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              mix_weight_offset,
        uint64_t              scale_offset,
        uint64_t              base_offset,
        uint64_t              norm_weight_offset,
        uint32_t              n,
        uint32_t              mix_dim,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              sinkhorn_iters,
        float                 eps,
        float                 hc_eps,
        float                 norm_eps);

/* Two same-shape BF16 matvecs over one shared input in a single dispatch;
 * bit-identical to two ds4_gpu_glm53_matmul_bf16 calls with n_rows = 1. */
int ds4_gpu_glm53_matmul_bf16_pair(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_a_offset,
        uint64_t              weight_b_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x);

/* KDA BF16 low-rank projection pack. flat3 runs f_a + g_a + beta (shared
 * input, different output extents) over one flat row space; pair2in runs
 * f_b + g_b (same shape, different input rows). Both are bit-identical to the
 * separate ds4_gpu_glm53_matmul_bf16 dispatches they replace. */
int ds4_gpu_glm53_matmul_bf16_flat3(
        ds4_gpu_tensor       *out_0,
        ds4_gpu_tensor       *out_1,
        ds4_gpu_tensor       *out_2,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_0_offset,
        uint64_t              weight_1_offset,
        uint64_t              weight_2_offset,
        uint32_t              in_dim,
        uint32_t              out_dim_0,
        uint32_t              out_dim_1,
        uint32_t              out_dim_2,
        const ds4_gpu_tensor *x);

int ds4_gpu_glm53_matmul_bf16_pair2in(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_a_offset,
        uint64_t              weight_b_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x_a,
        const ds4_gpu_tensor *x_b);

int ds4_gpu_glm53_matmul_bf16_qkv(
        ds4_gpu_tensor       *out_q,
        ds4_gpu_tensor       *out_k,
        ds4_gpu_tensor       *out_v,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_q_offset,
        uint64_t              weight_k_offset,
        uint64_t              weight_v_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x);

#ifndef DS4_GLM53_VISION_TYPES_DEFINED
#define DS4_GLM53_VISION_TYPES_DEFINED
#define DS4_GLM53_VISION_LAYERS 24u

typedef struct {
    uint64_t norm1;
    uint64_t qkv_weight;
    uint64_t qkv_bias;
    uint64_t q_norm;
    uint64_t k_norm;
    uint64_t attn_proj_weight;
    uint64_t attn_proj_bias;
    uint64_t norm2;
    uint64_t gate_weight;
    uint64_t gate_bias;
    uint64_t up_weight;
    uint64_t up_bias;
    uint64_t down_weight;
    uint64_t down_bias;
} ds4_glm53_vision_layer_weights;

typedef struct {
    uint64_t patch_weight;
    uint64_t patch_bias;
    uint64_t post_norm;
    uint64_t downsample_weight;
    uint64_t downsample_bias;
    uint64_t merger_proj;
    uint64_t merger_norm;
    uint64_t merger_norm_bias;
    uint64_t merger_gate;
    uint64_t merger_up;
    uint64_t merger_down;
    ds4_glm53_vision_layer_weights layer[DS4_GLM53_VISION_LAYERS];
} ds4_glm53_vision_weights;
#endif

/* Encode normalized, block-major image patches into 4096-wide language-model
 * embeddings. GPU implementations keep every intermediate on device. */
int ds4_gpu_glm53_vision_encode(
        float                          *out,
        const float                    *patches,
        uint32_t                        grid_h,
        uint32_t                        grid_w,
        const void                     *model_map,
        uint64_t                        model_size,
        const ds4_glm53_vision_weights *weights);

#ifndef DS4_DEEPSEEK4_VISION_TYPES_DEFINED
#define DS4_DEEPSEEK4_VISION_TYPES_DEFINED
#define DS4_DEEPSEEK4_VISION_LAYERS 32u
#define DS4_DEEPSEEK4_LANGUAGE_LAYERS 43u
#define DS4_DEEPSEEK4_MTP_LAYERS 3u

typedef struct {
    uint64_t norm1;
    uint64_t qkv_weight;
    uint64_t qkv_bias;
    uint64_t attn_proj_weight;
    uint64_t attn_proj_bias;
    uint64_t norm2;
    uint64_t mlp_w1;
    uint64_t mlp_w2;
} ds4_deepseek4_vision_layer_weights;

typedef struct {
    uint64_t patch_weight;
    uint64_t patch_bias;
    uint64_t post_norm;
    uint64_t aligner_w1;
    uint64_t aligner_w1_bias;
    uint64_t aligner_w2;
    uint64_t aligner_w2_bias;
    uint64_t image_start;
    uint64_t image_pad;
    uint64_t image_newline;
    uint64_t image_end;
    uint64_t visual_router_bias[DS4_DEEPSEEK4_LANGUAGE_LAYERS];
    uint64_t mtp_visual_router_bias[DS4_DEEPSEEK4_MTP_LAYERS];
    uint64_t hash_router_bias[3];
    ds4_deepseek4_vision_layer_weights layer[DS4_DEEPSEEK4_VISION_LAYERS];
    uint32_t projection_dim;
} ds4_deepseek4_vision_weights;
#endif

/* Encode row-major normalized 14x14 RGB patches. The output is the natural
 * row-major 3x3-aligned grid; N-layout permutation and sentinels are applied
 * by the prompt layer once the image's token position is known. */
int ds4_gpu_deepseek4_vision_encode(
        float                              *out,
        const float                        *patches,
        uint32_t                            grid_h,
        uint32_t                            grid_w,
        const void                         *model_map,
        uint64_t                            model_size,
        const ds4_deepseek4_vision_weights *weights);

/* Replace token rows with projected image embeddings and repeat each row into
 * every GLM hyperconnection stream. Must be called in an active command batch. */
int ds4_gpu_glm53_scatter_image_hc(
        ds4_gpu_tensor       *hc,
        const ds4_gpu_tensor *image,
        uint32_t              dst_row,
        uint32_t              image_row,
        uint32_t              rows,
        uint32_t              total_rows,
        uint32_t              n_embd,
        uint32_t              n_hc);

/* GLM-5.3 Kimi Delta Attention. Recurrent and convolution state stay FP32. */
int ds4_gpu_glm53_kda_decode(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint32_t              n_heads,
        uint32_t              n_rows,
        float                 gate_lower_bound,
        float                 norm_eps);

#ifdef __APPLE__
/* Three-dispatch form of the decode recurrence: same math, same bits, but
 * the 128-row state pass runs on 8x the threadgroups. `split_scratch` must
 * hold n_rows * n_heads * (516 + 128) floats. Returns 0 having encoded
 * nothing when the arguments or the scratch are unusable, or when a KDA
 * snapshot is armed, so the caller can fall back to the fused kernel. */
int ds4_gpu_glm53_kda_decode_split(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *split_scratch,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint32_t              n_heads,
        uint32_t              n_rows,
        float                 gate_lower_bound,
        float                 norm_eps);

/* Two-dispatch form: phases 1-3 in one threadgroup per head, as wide as the
 * device allows, then the same out kernel. Same math, same bits, one fewer
 * dispatch than the split above. Same scratch requirement (only the trailing
 * so[] half is touched) and the same fall-back-to-caller contract. */
int ds4_gpu_glm53_kda_decode_split2(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *split_scratch,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint32_t              n_heads,
        uint32_t              n_rows,
        float                 gate_lower_bound,
        float                 norm_eps);

/* Glued form: the f_b/g_b BF16 expansions folded into the per-head
 * threadgroup's prologue (do_prologue -- the caller must then skip the
 * pair2in dispatch) and the out kernel folded into its tail (do_out).  Both
 * bit-identical to the dispatches they replace; each is independently
 * selectable, and when do_out is 0 the standalone out kernel is emitted after
 * it.  lr_q8 selects the prologue's row body: 0 for BF16 f_b/g_b, 1 for Q8_0
 * (refused unless lr_in_dim is a multiple of 32 and lr_in_dim/32 <= 8, which
 * is what keeps the Q8_0 body bit-identical to the standalone matvec).  With
 * do_prologue 0 none of the f_b/g_b or low-rank arguments are touched, so the
 * out-tail fold runs on any lower-projection encoding.  Returns 0 without
 * encoding anything if it refuses. */
int ds4_gpu_glm53_kda_decode_glue(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *split_scratch,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        ds4_gpu_tensor       *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        ds4_gpu_tensor       *output_gate,
        const ds4_gpu_tensor *lowrank_f,
        const ds4_gpu_tensor *lowrank_g,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint64_t              f_b_offset,
        uint64_t              g_b_offset,
        uint32_t              lr_in_dim,
        int                   lr_q8,
        uint32_t              n_heads,
        uint32_t              n_rows,
        int                   do_prologue,
        int                   do_out,
        float                 gate_lower_bound,
        float                 norm_eps);
#endif

int ds4_gpu_glm53_kda_prefill(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *k,
        ds4_gpu_tensor       *v,
        ds4_gpu_tensor       *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint32_t              n_heads,
        uint32_t              n_tokens,
        float                 gate_lower_bound,
        float                 norm_eps);

/* Decode-island CUDA graph capture (CUDA backend; Metal/ROCm/CPU stub it
 * out and stay eager).  Design ported from the Entrpi/ds4 batched-serving
 * fork's per-layer decode graph capture.  The key identifies a captured
 * island: layer, island index, and the activation buffers whose addresses
 * the captured kernels bake in.  ds4_cuda.cu mirrors this struct
 * byte-for-byte (it does not include this header); keep both in sync. */
typedef struct ds4_decode_graph_key {
    uint32_t il;
    uint32_t island;    /* 0: layer top; 1: FFN tail; 2: V4.1 TP attn-out */
    uint32_t variant;
    uint32_t _pad;
    void    *cur_hc;
    void    *after_attn_hc;
    void    *after_ffn_hc;
    void    *attn_norm;
} ds4_decode_graph_key;

int  ds4_gpu_decode_graphs_supported(void);
/* 1: replayed (island already executed; skip encoding it)
 * 0: capturing (encode the island, then call _end)
 * -1: run eagerly */
int  ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *key);
/* 0: capture committed and launched; -1: capture failed (entry retired;
 * the caller must re-encode the island eagerly -- no work was executed). */
int  ds4_gpu_decode_graph_end(const ds4_decode_graph_key *key);
void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *key);
void ds4_gpu_decode_graphs_invalidate(void);

/* Scoped concurrent dispatch group for the batch pass (Metal only; other
 * backends may stub to 0). See ds4_metal.m for the ordering contract. */
int ds4_gpu_matmul_q8_0_qkv_tensor(
        ds4_gpu_tensor       *out_q,
        ds4_gpu_tensor       *out_k,
        ds4_gpu_tensor       *out_v,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_q_offset,
        uint64_t              weight_k_offset,
        uint64_t              weight_v_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x);

/* Q8_0 q/k/v with the three BF16 KDA low-rank rows (f_a, g_a, beta) folded in
 * as extra threadgroups of the same dispatch. Bit-identical to
 * ds4_gpu_matmul_q8_0_qkv_tensor + ds4_gpu_glm53_matmul_bf16_flat3. */
int ds4_gpu_matmul_q8_0_qkv_bf16_lowrank_tensor(
        ds4_gpu_tensor       *out_q,
        ds4_gpu_tensor       *out_k,
        ds4_gpu_tensor       *out_v,
        ds4_gpu_tensor       *out_lr_0,
        ds4_gpu_tensor       *out_lr_1,
        ds4_gpu_tensor       *out_lr_2,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_q_offset,
        uint64_t              weight_k_offset,
        uint64_t              weight_v_offset,
        uint64_t              weight_lr_0_offset,
        uint64_t              weight_lr_1_offset,
        uint64_t              weight_lr_2_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        uint32_t              lr_dim_0,
        uint32_t              lr_dim_1,
        uint32_t              lr_dim_2,
        const ds4_gpu_tensor *x);
/* Six-way fusion of the GLM-5.3 KDA projections that share the attention-norm
 * row: q, k, v, f_a, beta, g_a. Output extents differ per slice, so each is
 * passed alongside its weight offset (index order must match). */
int ds4_gpu_matmul_q8_0_kda6_tensor(
        ds4_gpu_tensor       *outs[6],
        const void           *model_map,
        uint64_t              model_size,
        const uint64_t        weight_offsets[6],
        uint64_t              in_dim,
        const uint64_t        out_dims[6],
        const ds4_gpu_tensor *x);
/* Same fusion over a flat concatenated row space, so no threadgroup retires
 * without work. Every out_dim must be a multiple of the matvec row count. */
int ds4_gpu_matmul_q8_0_kda6_flat_tensor(
        ds4_gpu_tensor       *outs[6],
        const void           *model_map,
        uint64_t              model_size,
        const uint64_t        weight_offsets[6],
        uint64_t              in_dim,
        const uint64_t        out_dims[6],
        const ds4_gpu_tensor *x);
/* Two same-shaped Q8_0 projections over two different input rows in one
 * dispatch (GLM-5.3 KDA f_b and g_b over their low-rank vectors). */
int ds4_gpu_matmul_q8_0_pair2in_tensor(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_a_offset,
        uint64_t              weight_b_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x_a,
        const ds4_gpu_tensor *x_b);
/* prefill lever 24: fill the three-slot dense half-copy ring from `count`
 * same-shaped Q8_0 weights on the current serial encoder, right before a
 * concurrent dispatch group is opened over their GEMMs. Returns 1 when armed;
 * release() disarms it after the group closes. */
int ds4_gpu_dense_half_ring_prepare(const void *model_map,
                                    uint64_t    model_size,
                                    const uint64_t *weight_offsets,
                                    int         count,
                                    uint32_t    weight_type,
                                    uint64_t    in_dim,
                                    uint64_t    out_dim,
                                    uint64_t    rows);
void ds4_gpu_dense_half_ring_release(void);

/* GLM-5.3 prefill router B4: four F32 matrix-unit K slices followed by a
 * fixed balanced F32 reduction. The graph supplies a scratch tensor holding
 * 4*n_tokens*288 floats. This Tier-2 path is default off and exact-mode
 * clamped; callers should query active() before dispatching it. */
int ds4_gpu_router_splitk_b4_active(void);
int ds4_gpu_router_splitk_b4_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *partials,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens);
/* --------------------------------------------------------------------------
 * Tier-2 screening instrumentation.
 *
 * Every screened candidate announces, on its FIRST dispatch, the exact shape
 * that was selected, and prints a per-tag dispatch count at exit.  A recorded
 * trial that does not carry both lines is not evidence about which variant ran
 * (candidate selection is cached in statics at first use, so an env change
 * inside one process does not switch it: one process per arm, always).
 * Not thread safe by design -- the decode encoder is single threaded and the
 * counter must cost nothing on the dispatch path.
 * -------------------------------------------------------------------------- */
typedef struct {
    const char        *tag;
    int                announced;
    unsigned long long count;
} ds4_t2s_slot;

void ds4_t2s_hit(ds4_t2s_slot *slot, const char *fmt, ...);

int ds4_gpu_concurrent_group_begin(void);
int ds4_gpu_concurrent_group_barrier(void);
int ds4_gpu_concurrent_group_end(void);
/* Qwen3.8-Flash-Next kernels (metal/qwen4.metal).  Weights are model-map
 * offsets; f32 transients unless noted.  Layouts: residual [T][hc*E],
 * GDN qkv [T][2*Hk*D + Hv*D] with tiled value heads, GDN state [Hv][D][D]
 * (dv-major), KV cache [cap][Hkv*D] f16, indexer k cache [cap][Di] f32,
 * block keys [n_blocks][Di] f16. */
/* xn = grouped RMSNorm(R) * gamma.  inj_part [T][hc*DS4_QWEN4_HC_CHUNKS][n_inject]
 * holds per-chunk partial inject dots (consumers apply 2*sigmoid(sum/hc)); the
 * low-rank projection is a plain GEMV of xn and gate_mix applies silu(lo/hc). */
#define DS4_QWEN4_HC_CHUNKS 8
/* CPU-compatible argmax: lowest index wins ties; scores <= -1e30 and
 * NaNs cannot replace initial index zero. Scratch needs ceil(V/4096)*8 bytes. */
/* Single-token Q8_0 matvec whose weight rows live in a GPU tensor (the
 * gathered MTP draft head); same kernel and geometry as the model-range path. */
int ds4_gpu_qwen4_matmul_q8_0_weights_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *w,
                                             uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x);
int ds4_gpu_qwen4_argmax_tensor(ds4_gpu_tensor *out_idx, ds4_gpu_tensor *scratch,
                               const ds4_gpu_tensor *logits, uint32_t n_vocab);
/* M3 Ultra decode defaults; DS4_QWEN4_DECODE_FUSIONS=0 restores old paths. */
int ds4_gpu_qwen4_decode_fusions_enabled(void);
/* Single-token F16 injection: old_R/old_inj and next_R/inj_part must be distinct. */
int ds4_gpu_qwen4_hc_combine_norm_tensor(
        ds4_gpu_tensor *next_R, const ds4_gpu_tensor *blk, const ds4_gpu_tensor *old_inj,
        ds4_gpu_tensor *xn, ds4_gpu_tensor *inj_part, const ds4_gpu_tensor *R,
        const void *model_map, uint64_t model_size, uint64_t gamma_offset, uint64_t inject_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc, uint32_t n_inject, float eps);
int ds4_gpu_qwen4_q8_pair_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);
int ds4_gpu_qwen4_hc_norm_tensor(
        ds4_gpu_tensor *xn, ds4_gpu_tensor *inj_part, const ds4_gpu_tensor *R,
        const void *model_map, uint64_t model_size, uint64_t gamma_offset, uint64_t inject_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc, uint32_t n_inject, float eps);
int ds4_gpu_qwen4_hc_gate_mix_tensor(
        ds4_gpu_tensor *mixed, const ds4_gpu_tensor *xn, const ds4_gpu_tensor *lo,
        const void *model_map, uint64_t model_size, uint64_t up_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc, uint32_t n_rank);
int ds4_gpu_qwen4_hc_combine_tensor(
        ds4_gpu_tensor *R, const ds4_gpu_tensor *out, const ds4_gpu_tensor *inj,
        uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc);
int ds4_gpu_qwen4_conv_stream_tensor(
        ds4_gpu_tensor *x, ds4_gpu_tensor *state,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_tokens, uint32_t n_channels, uint32_t conv_kernel, bool apply_silu);
int ds4_gpu_qwen4_gdn_prep_tensor(
        ds4_gpu_tensor *qkv, ds4_gpu_tensor *a, ds4_gpu_tensor *b,
        const void *model_map, uint64_t model_size, uint64_t ssm_a_offset, uint64_t dt_bias_offset,
        uint32_t n_tokens, uint32_t n_k_head, uint32_t n_v_head, uint32_t head_dim);
/* snap_state/snap2_state (optional) receive the state right after tokens
 * snap_tok/snap2_tok; two points serve the 3-row MTP verifier */
void ds4_gpu_qwen4_set_verify_rows_exact(bool on);
int ds4_gpu_qwen4_gdn_scan_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *state, const ds4_gpu_tensor *qkv,
        const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
        uint32_t n_tokens, uint32_t n_k_head, uint32_t n_v_head, uint32_t head_dim,
        ds4_gpu_tensor *snap_state, uint32_t snap_tok,
        ds4_gpu_tensor *snap2_state, uint32_t snap2_tok);
int ds4_gpu_qwen4_gdn_out_tensor(
        ds4_gpu_tensor *o, const ds4_gpu_tensor *z,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim, float eps);
int ds4_gpu_qwen4_ple_gate_tensor(
        ds4_gpu_tensor *gated, ds4_gpu_tensor *normed, const ds4_gpu_tensor *R,
        const ds4_gpu_tensor *key, const ds4_gpu_tensor *value,
        const void *model_map, uint64_t model_size,
        uint64_t g_key_offset, uint64_t g_query_offset, uint64_t g_conv_offset,
        uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc, float eps);
int ds4_gpu_qwen4_ple_conv_tensor(
        ds4_gpu_tensor *R, const ds4_gpu_tensor *gated, const ds4_gpu_tensor *normed,
        ds4_gpu_tensor *history, const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_channels, uint32_t conv_kernel, uint32_t dilation,
        ds4_gpu_tensor *snap_history, uint32_t snap_tok,
        ds4_gpu_tensor *snap2_history, uint32_t snap2_tok);
/* up to four projections of x in one dispatch; weight types 0 f32, 1 f16,
 * 2 q4_0, 8 q8_0, 30 bf16, 39 mxfp4 */
int ds4_gpu_qwen4_multi_gemv_tensor(
        const ds4_gpu_tensor *x, uint32_t n_tokens, uint32_t in_dim, uint32_t n_out,
        ds4_gpu_tensor *const *outs, const void *model_map, uint64_t model_size,
        const uint64_t *offsets, const uint32_t *types, const uint32_t *out_rows);
/* softmax top-k router; with in_dim != 0 also the shared expert gate logit
 * (one row of gate_type at gate_offset dotted with x) into shared_gate [T] */
int ds4_gpu_qwen4_router_topk_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t gate_offset,
        uint32_t gate_type, uint32_t in_dim, ds4_gpu_tensor *shared_gate,
        uint32_t n_tokens, uint32_t n_expert, uint32_t n_used);
/* Rotary table for the Qwen3.8 kernels: n_pairs inverse frequencies and the
 * cos/sin magnitude scale (YaRN); NULL restores plain rope from the base. */
void ds4_gpu_qwen4_set_rope(const float *freq, uint32_t n_pairs, float mscale);
int ds4_gpu_qwen4_attn_prep_tensor(
        ds4_gpu_tensor *q_out, ds4_gpu_tensor *gate_out, ds4_gpu_tensor *k_cache, ds4_gpu_tensor *v_cache,
        ds4_gpu_tensor *iq_out, ds4_gpu_tensor *ik_cache,
        const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kproj, const ds4_gpu_tensor *vproj,
        const ds4_gpu_tensor *iq, const ds4_gpu_tensor *ik, const ds4_gpu_tensor *pos3,
        const void *model_map, uint64_t model_size,
        uint64_t g_q_offset, uint64_t g_k_offset, uint64_t g_iq_offset,
        uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t n_rot,
        uint32_t n_idx_head, uint32_t idx_dim, uint32_t pos0, uint32_t cache_cap,
        float rope_base, float eps);
/* Predictor prefix preparation: identical K/V/IK arithmetic to attn_prep,
 * without query, gate or indexer-query inputs/outputs. Cache spans and all
 * inputs must be disjoint from the three output spans. */
int ds4_gpu_qwen4_attn_cache_prep_tensor(
        ds4_gpu_tensor *k_cache, ds4_gpu_tensor *v_cache, ds4_gpu_tensor *ik_cache,
        const ds4_gpu_tensor *kproj, const ds4_gpu_tensor *vproj, const ds4_gpu_tensor *ik,
        const ds4_gpu_tensor *pos3, const void *model_map, uint64_t model_size, uint64_t g_k_offset,
        uint32_t n_tokens, uint32_t n_head_kv, uint32_t head_dim, uint32_t n_rot,
        uint32_t idx_dim, uint32_t pos0, uint32_t cache_cap, float rope_base, float eps);
int ds4_gpu_qwen4_idx_block_key_tensor(
        ds4_gpu_tensor *block_key, const ds4_gpu_tensor *ik_cache, const ds4_gpu_tensor *pos3,
        const void *model_map, uint64_t model_size, uint64_t g_ik_offset,
        uint32_t block0, uint32_t n_blocks, uint32_t ratio, uint32_t idx_dim, uint32_t n_rot,
        float rope_base, float eps);
int ds4_gpu_qwen4_idx_score_tensor(
        ds4_gpu_tensor *score, ds4_gpu_tensor *tile_max, const ds4_gpu_tensor *iq, const ds4_gpu_tensor *block_key,
        uint32_t n_tokens, uint32_t n_blocks, uint32_t n_idx_head, uint32_t idx_dim,
        uint32_t pos0, uint32_t ratio);
int ds4_gpu_qwen4_idx_select_tensor(
        ds4_gpu_tensor *sel, const ds4_gpu_tensor *score, const ds4_gpu_tensor *tile_max,
        uint32_t n_blocks, uint32_t n_tokens, uint32_t top_k);
int ds4_gpu_qwen4_idx_expand_tensor(
        ds4_gpu_tensor *sel_tokens, ds4_gpu_tensor *n_sel, const ds4_gpu_tensor *sel_blocks,
        uint32_t n_tokens, uint32_t n_sel_blocks, uint32_t ratio, uint32_t pos0, uint32_t sel_stride);
/* part (optional, decode-sized batches): partial-softmax scratch of
 * ds4_gpu_qwen4_attn_part_floats() floats enabling key-split parallelism */
uint64_t ds4_gpu_qwen4_attn_part_floats(uint32_t n_tokens, uint32_t n_head, uint32_t head_dim);
int ds4_gpu_qwen4_attn_decode_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *sel_tokens, const ds4_gpu_tensor *n_sel, ds4_gpu_tensor *part,
        uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
        uint32_t pos0, bool use_sel, uint32_t sel_stride, float scale);
#ifdef __APPLE__
/* Metal: a subrange of a matrix-attention prefill retains its parent kernel
 * arithmetic even when the dense/sparse partition contains only 1..8 rows. */
int ds4_gpu_qwen4_attn_prefill_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *sel_tokens, const ds4_gpu_tensor *n_sel,
        uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
        uint32_t pos0, bool use_sel, uint32_t sel_stride, float scale);
#endif
/* Routed experts; shared_type == UINT32_MAX disables the shared-expert slot,
 * otherwise mid/part carry n_slots+1 entries and the reduce weights the last
 * one by sigmoid(shared_gate). */
int ds4_gpu_qwen4_moe_mid_tensor(
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *selected,
        const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint32_t weight_type, uint32_t n_total_expert, uint32_t n_tokens, uint32_t n_slots,
        uint32_t in_dim, uint32_t ff_dim,
        uint64_t shared_gate_offset, uint64_t shared_up_offset, uint32_t shared_type);
int ds4_gpu_qwen4_moe_down_tensor(
        ds4_gpu_tensor *part, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected,
        const void *model_map, uint64_t model_size, uint64_t down_offset,
        uint32_t weight_type, uint32_t n_total_expert, uint32_t n_tokens, uint32_t n_slots,
        uint32_t ff_dim, uint32_t out_dim,
        uint64_t shared_down_offset, uint32_t shared_type);
int ds4_gpu_qwen4_moe_mid_grouped_tensor(
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, uint32_t list_cap,
        const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint32_t weight_type, uint32_t n_total_expert, uint32_t n_tokens, uint32_t n_slots,
        uint32_t in_dim, uint32_t ff_dim);
int ds4_gpu_qwen4_moe_down_grouped_tensor(
        ds4_gpu_tensor *part, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, uint32_t list_cap,
        const void *model_map, uint64_t model_size, uint64_t down_offset,
        uint32_t weight_type, uint32_t n_total_expert, uint32_t n_tokens, uint32_t n_slots,
        uint32_t ff_dim, uint32_t out_dim);
/* shared_gate NULL: no shared expert; shared NULL: the shared output is part
 * slot n_slots, otherwise `shared` [T][dim] holds it.  part_stride = slots per
 * token in part. */
int ds4_gpu_qwen4_moe_down_reduce_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *shared_gate, ds4_gpu_tensor *R,
        const ds4_gpu_tensor *inj, const void *model_map, uint64_t model_size, uint64_t down_offset,
        uint32_t weight_type, uint32_t n_total_expert, uint32_t n_slots, uint32_t ff_dim, uint32_t out_dim,
        uint64_t shared_down_offset, uint32_t shared_type, uint32_t n_hc);
int ds4_gpu_qwen4_moe_reduce_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *part, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *shared_gate, const ds4_gpu_tensor *shared, ds4_gpu_tensor *R, const ds4_gpu_tensor *inj,
        uint32_t n_tokens, uint32_t n_slots, uint32_t part_stride, uint32_t dim, uint32_t n_hc);
/* prefill experts: per-expert token lists, then expert-grouped tiled GEMMs
 * (q8_0/mxfp4/q4_K/q2_K/iq2_xxs) */
int ds4_gpu_qwen4_moe_build_lists_tensor(
        ds4_gpu_tensor *lists, ds4_gpu_tensor *counts, const ds4_gpu_tensor *selected,
        uint32_t n_tokens, uint32_t n_slots, uint32_t n_expert, uint32_t list_cap);
int ds4_gpu_qwen4_moe_mm_mid_tensor(
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts,
        const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint32_t weight_type, uint32_t n_expert, uint32_t n_tokens, uint32_t n_slots, uint32_t n_out,
        uint32_t in_dim, uint32_t ff_dim, uint32_t list_cap);
int ds4_gpu_qwen4_moe_mm_down_tensor(
        ds4_gpu_tensor *part, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts,
        const void *model_map, uint64_t model_size, uint64_t down_offset,
        uint32_t weight_type, uint32_t n_expert, uint32_t n_tokens, uint32_t n_slots, uint32_t n_out,
        uint32_t ff_dim, uint32_t out_dim, uint32_t list_cap);
/* Metal SSD routed experts. Table sizes are the stored bytes per expert,
 * including Q2_K down-row padding; ff_dim remains the logical activation width.
 * NULL lists/counts use the ordinary row kernels, including the shared slot.
 * Non-NULL lists/counts use the ordinary expert-grouped MM kernels with routed
 * slots only; the caller computes the shared expert and performs the reduction.
 * The call completes the router before reading IDs and preserves command-batch
 * ownership. Cache overflow and off-size layers use explicit owned SSD reads,
 * never mapped routed weights. Returns zero on invalid input or I/O failure. */
#ifdef __APPLE__
int ds4_gpu_qwen4_moe_stream_tensor(
        ds4_gpu_tensor *mid, ds4_gpu_tensor *part, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts,
        const ds4_gpu_stream_expert_table *table,
        uint32_t gate_type, uint32_t down_type, uint32_t n_tokens, uint32_t n_slots,
        uint32_t in_dim, uint32_t ff_dim, uint32_t list_cap,
        uint64_t shared_gate_offset, uint64_t shared_up_offset, uint64_t shared_down_offset,
        uint32_t shared_type, uint32_t shared_down_type);
#endif
/* weight_type covers both the alpha and the beta projection */
int ds4_gpu_qwen4_gdn_front_tensor(
        ds4_gpu_tensor *qkv, ds4_gpu_tensor *state, const ds4_gpu_tensor *mixed,
        ds4_gpu_tensor *ga, ds4_gpu_tensor *gb,
        const void *model_map, uint64_t model_size, uint64_t conv_offset,
        uint64_t alpha_offset, uint64_t beta_offset, uint64_t ssm_a_offset, uint64_t dt_bias_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_k_head, uint32_t n_v_head, uint32_t head_dim,
        uint32_t conv_kernel, uint32_t in_dim, ds4_gpu_tensor *snap_state, uint32_t snap_tok,
        ds4_gpu_tensor *snap2_state, uint32_t snap2_tok);
/* Encode one image: patches [n_patches][3*P*P] in 2x2 window order plus the
 * resampled position embedding [n_patches][n_embd]; out receives
 * [n_patches/4][n_out].  Weights are read from the mapped mmproj GGUF. */
int ds4_gpu_qwen4_vision_encode(float *out, const float *patches, const float *pos_embed, uint32_t n_patches,
                                uint32_t grid_w, const void *model_map, uint64_t model_size,
                                const ds4_qwen4_vision_weights *w);
/* Dense GEMM (f32/f16/q8_0 rows, 32x32 tiles) and the batched hc mix pieces. */
int ds4_gpu_qwen4_dense_mm_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t weight_type,
        uint32_t n_tokens, uint32_t in_dim, uint32_t out_rows);
int ds4_gpu_qwen4_hc_lo_act_tensor(ds4_gpu_tensor *lo_act, const ds4_gpu_tensor *lo,
                                   uint32_t n_tokens, uint32_t n_hc, uint32_t n_rank);
int ds4_gpu_qwen4_hc_mix_rows_tensor(ds4_gpu_tensor *mixed, const ds4_gpu_tensor *u, const ds4_gpu_tensor *xn,
                                     uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc);
int ds4_gpu_qwen4_mtp_stage_tensor(
        ds4_gpu_tensor *cat, const ds4_gpu_tensor *e, const ds4_gpu_tensor *R,
        const void *model_map, uint64_t model_size, uint64_t g_e_offset, uint64_t g_h_offset,
        uint32_t n_embd, uint32_t n_hc, float eps);
/* Apply Q8 weights [E][2E] to the exact [T][hc+1][2E] layout written by
 * mtp_stage: [embedding|0], then hc rows [0|hidden]. Preserves the existing
 * Q8 batch arithmetic and full row strides; out must not overlap cat/weights. */
int ds4_gpu_qwen4_mtp_project_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *cat,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens);
int ds4_gpu_qwen4_mtp_combine_tensor(
        ds4_gpu_tensor *R_out, const ds4_gpu_tensor *proj, uint32_t n_embd, uint32_t n_hc);

#ifdef __cplusplus
}
#endif


/* ---- Kimi K3 (2026-10-31): decode primitives implemented at the end of ds4_cuda.cu ---- */
int ds4_gpu_k3_residual_mix_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *anchors,
                                   uint32_t n_anchor, const void *model_map, uint64_t model_size,
                                   uint64_t norm_offset, uint64_t proj_offset, uint32_t n, float eps);
int ds4_gpu_k3_situ_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
                           uint32_t n, float beta, float beta_linear);
int ds4_gpu_k3_sigmoid_mul_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *gate, uint32_t n);
int ds4_gpu_k3_split_kvb_tensor(ds4_gpu_tensor *k_nope, ds4_gpu_tensor *value, const ds4_gpu_tensor *kvb,
                                uint32_t heads, uint32_t nope, uint32_t vdim);
int ds4_gpu_k3_mla_store_tensor(ds4_gpu_tensor *key_cache, ds4_gpu_tensor *value_cache, const ds4_gpu_tensor *k_nope,
                                const ds4_gpu_tensor *kv_raw, const ds4_gpu_tensor *value, uint32_t pos,
                                uint32_t cache_cap, uint32_t heads, uint32_t nope, uint32_t rot, uint32_t kv_lora,
                                uint32_t vdim);
int ds4_gpu_k3_mla_attend_tensor(ds4_gpu_tensor *heads_out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *key_cache,
                                 const ds4_gpu_tensor *value_cache, uint32_t n_pos, uint32_t heads, uint32_t qk,
                                 uint32_t vdim);
int ds4_gpu_k3_kda_decode(ds4_gpu_tensor *out, ds4_gpu_tensor *conv_state, ds4_gpu_tensor *recurrent_state,
                          const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
                          const ds4_gpu_tensor *raw_gate, const ds4_gpu_tensor *raw_beta,
                          const ds4_gpu_tensor *output_gate, const void *model_map, uint64_t model_size,
                          uint64_t q_conv_offset, uint64_t k_conv_offset, uint64_t v_conv_offset,
                          uint64_t a_log_offset, uint64_t dt_bias_offset, uint64_t output_norm_offset,
                          uint32_t n_heads, float gate_lower_bound, float norm_eps, int a_log_per_channel);
int ds4_gpu_k3_moe_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const int32_t *selected, const float *weights,
                          uint32_t n_used, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                          uint32_t ff, uint32_t latent, float beta, float beta_linear);
/* Raw-v1 expert pipeline: whole expert files read by a persistent reader pool into pinned slots, uploaded as they
 * land, GEMVs straight on the raw planes. `layer` is the GGUF block index (= raw-v1 file layer). */
int ds4_gpu_k3_moe_raw_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const int32_t *selected,
                              const float *weights, uint32_t n_used, uint32_t layer, uint32_t ff, uint32_t latent,
                              float beta, float beta_linear);
void ds4_gpu_k3_moe_raw_stats(double *wait_s, uint64_t *bytes);
int ds4_gpu_k3_cache_grow(void);
uint64_t ds4_gpu_k3_cache_lend(void);
uint32_t ds4_gpu_k3_attn_max_pos(void);
uint64_t ds4_gpu_k3_attn_scratch_need(uint32_t n_tok, uint32_t n_cols, uint32_t heads);   /* step 103: GEMM attention scratch growth */   /* step 99: positions the K3 attention kernels take (shared memory opt-in limit) */   /* step 97: frees the grown region for a prompt pass; returns the bytes freed */
uint64_t ds4_gpu_k3_pass_scratch_need(uint32_t n_tok, uint32_t n_used, uint32_t ff, uint32_t latent);   /* step 97 */   /* sizes the device expert cache from free VRAM; call after the first full forward */
int ds4_gpu_k3_skip_mode(void);   /* step 57: DS4_K3_EXPERT_SKIP (0 off, 1 on, 2 observe only) */
void ds4_gpu_k3_skip_stats(uint64_t *skipped, double *mass, uint64_t *obs_n, uint64_t obs[6]);
void ds4_gpu_k3_cache_stats(uint64_t *hits_dev, uint64_t *hits_host, uint64_t *misses);   /* expert cache counters */
uint64_t ds4_gpu_k3_cache_copydowns(void);   /* victim copy-downs so far */
void ds4_gpu_k3_set_f32_act(int on);   /* step 106: K3 spine and prompt-pass expert matmuls with fp32 activations */
int ds4_gpu_k3_kvb_chain_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset,
                                uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x, uint32_t n_tok);   /* step 107 */
int ds4_gpu_k3_lat_replay_sel_tensor(ds4_gpu_tensor *key_cache, ds4_gpu_tensor *value_cache, ds4_gpu_tensor *kvb,
        const ds4_gpu_tensor *norm_rows, const ds4_gpu_tensor *raw_rows, const ds4_gpu_tensor *cls, const ds4_gpu_tensor *mask,
        uint32_t q0, uint32_t q1, uint32_t max_rows, uint32_t cache_cap, const void *model_map, uint64_t model_size,
        uint64_t kvb_offset, uint32_t heads, uint32_t nope, uint32_t rot, uint32_t kv_lora, uint32_t vdim);   /* step 107 */
int ds4_gpu_k3_memset_tensor(ds4_gpu_tensor *t, uint64_t offset, int value, uint64_t bytes);   /* step 107 */
void ds4_gpu_k3_prefetch_hint(uint32_t layer, const int32_t *experts, uint32_t n);   /* ranked candidates for a layer */
void ds4_gpu_k3_prefetch_stats(uint64_t *issued, uint64_t *adopted);
void ds4_gpu_k3_set_verify(int on);
int ds4_gpu_k3_mla_cache_elem_bytes(void);   /* step 48: 4 (fp32) or 2 (DS4_K3_MLA_CACHE=f16|bf16) */
/* step 46: the native DSpark drafter (ds4_k3_draft.cuh) */
int ds4_gpu_k3_draft_load(const char *dir, uint32_t cache_cap);
int ds4_gpu_k3_draft_capture(int ci, const ds4_gpu_tensor *cur, uint32_t T);
int ds4_gpu_k3_draft_commit(uint32_t pos0, uint32_t T);
int ds4_gpu_k3_draft_truncate(uint32_t len);
uint32_t ds4_gpu_k3_draft_len(void);
int ds4_gpu_k3_draft_backbone(const ds4_gpu_tensor *embed7, uint32_t cache_len, ds4_gpu_tensor *hidden7_out);
int ds4_gpu_k3_draft_finish(ds4_gpu_tensor *logits7, int anchor, int ids[7], float conf[7]);
double ds4_gpu_k3_draft_gpu_bytes(void);                                                   /* step 44: verification batches keep decode policies */
void ds4_gpu_k3_queue_stats(double *wait_s, uint64_t *n_waits, uint64_t *pf_hit);   /* step 44: demand reads' queue delay; prompt-pass prefetch hits */
int  ds4_gpu_k3_route_hist(uint32_t *counts, const ds4_gpu_tensor *logits, const float *bias, uint32_t n_tok,
                           uint32_t n_expert, uint32_t n_used);   /* step 35: per-token top-k of sigmoid(logit)+bias, counted per expert */
void ds4_gpu_k3_set_spine_gemm_max(uint32_t n);   /* step 37: Q8 spine matmuls of 2..n rows run as dequant-to-f16 GEMMs (0: never) */
void ds4_gpu_k3_argodrive_stats(uint64_t *chunks3, double *stream_gbps3);   /* step 38: chunk reads per tier, measured per-stream GB/s */
void ds4_gpu_k3_source_stats(uint64_t *bytes3, uint64_t *fallback_reads, uint64_t *fallback_bytes, int *tier_c_configured);   /* step 42 */
/* Batched prompt pass: n_tok rows at once (n_tok = 1 is the decode step). */
int ds4_gpu_k3_pool_prepare(uint32_t ff, uint32_t latent, uint32_t n_tok, uint32_t n_used);   /* step 77 */
int ds4_gpu_k3_pool_prepare_staging(uint32_t ff, uint32_t latent, uint32_t n_tok, uint32_t n_used, const void *model_map, const uint64_t *offsets, const uint64_t *sizes, uint32_t n_ranges);   /* step 87 */
int ds4_gpu_k3_moe_raw_batch_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const int32_t *selected,
                                    const float *weights, uint32_t n_tok, uint32_t n_used, uint32_t layer, uint32_t ff,
                                    uint32_t latent, float beta, float beta_linear);
int ds4_gpu_k3_residual_mix_batch_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *anchors,
                                         uint32_t n_anchor, uint32_t n_tok, const void *model_map, uint64_t model_size,
                                         uint64_t norm_offset, uint64_t proj_offset, uint32_t n, float eps);
int ds4_gpu_k3_anchor_push_batch_tensor(ds4_gpu_tensor *anchors, const ds4_gpu_tensor *cur, uint32_t slot, uint32_t n_tok,
                                        uint32_t n);
int ds4_gpu_k3_copy_rows_tensor(ds4_gpu_tensor *dst, uint32_t dst_stride, const ds4_gpu_tensor *src, uint32_t src_stride,
                                uint32_t width, uint32_t n_rows);
int ds4_gpu_k3_split_kvb_batch_tensor(ds4_gpu_tensor *k_nope, ds4_gpu_tensor *value, const ds4_gpu_tensor *kvb,
                                      uint32_t n_tok, uint32_t heads, uint32_t nope, uint32_t vdim);
int ds4_gpu_k3_mla_store_batch_tensor(ds4_gpu_tensor *key_cache, ds4_gpu_tensor *value_cache, const ds4_gpu_tensor *k_nope,
                                      const ds4_gpu_tensor *kv_raw, const ds4_gpu_tensor *value, uint32_t pos0,
                                      uint32_t n_tok, uint32_t cache_cap, uint32_t heads, uint32_t nope, uint32_t rot,
                                      uint32_t kv_lora, uint32_t vdim);
int ds4_gpu_k3_mla_attend_batch_tensor(ds4_gpu_tensor *heads_out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *key_cache,
                                       const ds4_gpu_tensor *value_cache, uint32_t pos0, uint32_t n_tok, uint32_t heads,
                                       uint32_t qk, uint32_t vdim);
/* step 95: K3.1 sparse attention (DSA). The indexer's weights are their own device range (f32, host copy as the key). */
int ds4_gpu_k3_lat_store_kvb_tensor(ds4_gpu_tensor *key_cache, ds4_gpu_tensor *value_cache, const ds4_gpu_tensor *kvb,
                                    const ds4_gpu_tensor *kv_raw, const ds4_gpu_tensor *mask, uint32_t pos0, uint32_t n_tok,
                                    uint32_t cache_cap, uint32_t heads, uint32_t nope, uint32_t rot, uint32_t kv_lora, uint32_t vdim);   /* step 96c */
int ds4_gpu_k3_register_aux_weights(const void *host, uint64_t bytes);
int ds4_gpu_k3_dsa_matmul_tensor(ds4_gpu_tensor *out, const void *aux, uint64_t aux_size, uint64_t offset, uint32_t in_dim,
                                 uint32_t out_dim, const ds4_gpu_tensor *x, uint32_t n_tok);
int ds4_gpu_k3_dsa_ln_store_tensor(ds4_gpu_tensor *key_cache, const ds4_gpu_tensor *raw, const void *aux, uint64_t aux_size,
                                   uint64_t w_off, uint64_t b_off, uint32_t pos0, uint32_t n_tok, uint32_t cap, float eps);
int ds4_gpu_k3_dsa_scores_tensor(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights,
                                 const ds4_gpu_tensor *key_cache, uint32_t n_rows, uint32_t n_tokens, uint32_t pos0,
                                 uint32_t n_head, float scale);
int ds4_gpu_k3_dsa_mask_tensor(ds4_gpu_tensor *mask, const ds4_gpu_tensor *selected, uint32_t stride, uint32_t n_tokens,
                               uint32_t pos0, uint32_t top_k);
int ds4_gpu_k3_mla_attend_masked_tensor(ds4_gpu_tensor *heads_out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *key_cache,
                                        const ds4_gpu_tensor *value_cache, const ds4_gpu_tensor *mask, uint32_t n_pos,
                                        uint32_t heads, uint32_t qk, uint32_t vdim);
int ds4_gpu_k3_mla_attend_batch_masked_tensor(ds4_gpu_tensor *heads_out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *key_cache,
                                              const ds4_gpu_tensor *value_cache, const ds4_gpu_tensor *mask, uint32_t pos0,
                                              uint32_t n_tok, uint32_t mask_stride, uint32_t top_k, uint32_t heads, uint32_t qk,
                                              uint32_t vdim);
int ds4_gpu_k3_kda_decode_batch(ds4_gpu_tensor *out, ds4_gpu_tensor *conv_state, ds4_gpu_tensor *recurrent_state,
                                const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
                                const ds4_gpu_tensor *raw_gate, const ds4_gpu_tensor *raw_beta,
                                const ds4_gpu_tensor *output_gate, const void *model_map, uint64_t model_size,
                                uint64_t q_conv_offset, uint64_t k_conv_offset, uint64_t v_conv_offset,
                                uint64_t a_log_offset, uint64_t dt_bias_offset, uint64_t output_norm_offset,
                                uint32_t n_heads, float gate_lower_bound, float norm_eps, int a_log_per_channel,
                                uint32_t n_tok);

#endif
