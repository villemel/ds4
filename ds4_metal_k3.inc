/* Kimi K3 / K3.1 Pro on Metal (8 Nov 2026), included at the end of ds4_metal.m. The decode path of the CUDA port: residual mix,
   SiTU, sigmoid gate, kv_b split, MLA store and attention (dense or with a DSA mask), KDA step, and the routed experts from a
   unified-memory slab cache (expert files read straight into GPU-visible memory: no host to device copies on Apple Silicon).
   Batched prompt passes, the DSA indexer, the latent cache and the drafter are not ported yet: run with
   DS4_K3_PREFILL_SEQUENTIAL=1 and -c 2048. Every op commits and waits (the K3 driver does not batch command buffers). */
#include "ds4_kimi_k3.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <dispatch/dispatch.h>
#include <stdatomic.h>
#include <os/lock.h>

static int k3m_unsupported(const char *what) {
    static int said[64]; static const char *names[64]; static int n = 0;
    for (int i = 0; i < n; i++) if (names[i] == what) return 0;
    if (n < 64) { names[n] = what; said[n] = 1; n++; }
    fprintf(stderr, "ds4: K3 on Metal: %s is not ported yet\n", what);
    return 0;
}
static id<MTLBuffer> k3m_model(const void *map, uint64_t size, uint64_t off, uint64_t bytes, uint64_t *inner) {
    if (!map || off > size || bytes > size - off) { fprintf(stderr, "ds4: K3 on Metal: weight range outside the model\n"); return nil; }
    return ds4_gpu_wrap_model_range(map, size, off, bytes, inner);
}
static void k3m_batching_init(void) {
    if (g_k3m_batching >= 0) return;
    const char *e = getenv("DS4_K3_METAL_BATCH");
    g_k3m_batching = (e && strcmp(e, "0") == 0) ? 0 : 1;
    fprintf(stderr, "ds4: K3 on Metal: command batching %s\n", g_k3m_batching ? "on (host access completes the batch)" : "off (every op waits)");
}
static int k3m_run(const char *kernel, void (^encode)(id<MTLComputeCommandEncoder> enc), const char *label) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    k3m_batching_init();
    id<MTLComputePipelineState> p = ds4_gpu_get_pipeline(kernel);
    if (!p) { fprintf(stderr, "ds4: K3 on Metal: kernel %s missing\n", kernel); return 0; }
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:p];
    encode(enc);
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned, label);
}
#define K3M_BUF(enc, t, idx) [enc setBuffer:ds4_gpu_tensor_buffer(t) offset:ds4_gpu_tensor_offset(t) atIndex:(idx)]

/* ---- small kernels ---- */
typedef struct { uint32_t n_anchor, n; float eps; uint32_t anchor_stride; } k3m_mix_args;
static int k3m_residual_mix(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *anchors, uint32_t n_anchor, uint32_t n_tok,
                            const void *map, uint64_t size, uint64_t norm_off, uint64_t proj_off, uint32_t n, float eps, uint32_t stride) {
    if (!out || !x || n == 0u || n_anchor > 16u || (n_anchor && !anchors)) return 0;
    @autoreleasepool {
        uint64_t in_n = 0, in_p = 0;
        id<MTLBuffer> wn = k3m_model(map, size, norm_off, (uint64_t)n * 4u, &in_n);
        id<MTLBuffer> wp = k3m_model(map, size, proj_off, (uint64_t)n * 4u, &in_p);
        if (!wn || !wp) return 0;
        k3m_mix_args a = { n_anchor, n, eps, stride };
        const ds4_gpu_tensor *anc = anchors ? anchors : x;
        return k3m_run("kernel_k3_residual_mix", ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBytes:&a length:sizeof(a) atIndex:0];
            K3M_BUF(enc, x, 1); K3M_BUF(enc, anc, 2);
            [enc setBuffer:wn offset:(NSUInteger)in_n atIndex:3];
            [enc setBuffer:wp offset:(NSUInteger)in_p atIndex:4];
            K3M_BUF(enc, out, 5);
            [enc dispatchThreadgroups:MTLSizeMake(n_tok, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }, "K3 residual mix");
    }
}
int ds4_gpu_k3_residual_mix_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *anchors, uint32_t n_anchor,
                                   const void *model_map, uint64_t model_size, uint64_t norm_offset, uint64_t proj_offset, uint32_t n, float eps) {
    return k3m_residual_mix(out, x, anchors, n_anchor, 1u, model_map, model_size, norm_offset, proj_offset, n, eps, 0u);
}
int ds4_gpu_k3_residual_mix_batch_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *anchors, uint32_t n_anchor,
                                         uint32_t n_tok, const void *model_map, uint64_t model_size, uint64_t norm_offset, uint64_t proj_offset,
                                         uint32_t n, float eps) {
    return k3m_residual_mix(out, x, anchors, n_anchor, n_tok, model_map, model_size, norm_offset, proj_offset, n, eps, 16u * n);
}
typedef struct { uint32_t n; float b, bl; } k3m_ew_args;
static int k3m_ew(const char *kernel, ds4_gpu_tensor *out, const ds4_gpu_tensor *a0, const ds4_gpu_tensor *a1, uint32_t n, float b, float bl) {
    if (!out || !a0 || !a1 || n == 0u) return 0;
    k3m_ew_args a = { n, b, bl };
    return k3m_run(kernel, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        K3M_BUF(enc, a0, 1); K3M_BUF(enc, a1, 2); K3M_BUF(enc, out, 3);
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }, kernel);
}
int ds4_gpu_k3_situ_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float beta, float beta_linear) {
    return k3m_ew("kernel_k3_situ", out, gate, up, n, beta, beta_linear);
}
int ds4_gpu_k3_sigmoid_mul_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *gate, uint32_t n) {
    return k3m_ew("kernel_k3_sigmoid_mul", out, x, gate, n, 0.0f, 0.0f);
}
typedef struct { uint32_t heads, nope, vdim; } k3m_kvb_args;
static int k3m_split_kvb(ds4_gpu_tensor *k_nope, ds4_gpu_tensor *value, const ds4_gpu_tensor *kvb, uint32_t n_tok, uint32_t heads, uint32_t nope, uint32_t vdim) {
    if (!k_nope || !value || !kvb || n_tok == 0u) return 0;
    k3m_kvb_args a = { heads, nope, vdim };
    return k3m_run("kernel_k3_split_kvb", ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        K3M_BUF(enc, k_nope, 1); K3M_BUF(enc, value, 2); K3M_BUF(enc, kvb, 3);
        [enc dispatchThreads:MTLSizeMake(heads * (nope + vdim), n_tok, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }, "K3 kv_b split");
}
int ds4_gpu_k3_split_kvb_tensor(ds4_gpu_tensor *k_nope, ds4_gpu_tensor *value, const ds4_gpu_tensor *kvb, uint32_t heads, uint32_t nope, uint32_t vdim) {
    return k3m_split_kvb(k_nope, value, kvb, 1u, heads, nope, vdim);
}
int ds4_gpu_k3_split_kvb_batch_tensor(ds4_gpu_tensor *k_nope, ds4_gpu_tensor *value, const ds4_gpu_tensor *kvb, uint32_t n_tok, uint32_t heads,
                                      uint32_t nope, uint32_t vdim) {
    return k3m_split_kvb(k_nope, value, kvb, n_tok, heads, nope, vdim);
}
typedef struct { uint32_t pos0, heads, nope, rot, kv_lora, vdim; } k3m_store_args;
static int k3m_store(ds4_gpu_tensor *kc, ds4_gpu_tensor *vc, const ds4_gpu_tensor *k_nope, const ds4_gpu_tensor *kv_raw, const ds4_gpu_tensor *value,
                     uint32_t pos0, uint32_t n_tok, uint32_t cap, uint32_t heads, uint32_t nope, uint32_t rot, uint32_t kv_lora, uint32_t vdim) {
    if (!kc || !vc || !k_nope || !kv_raw || !value || n_tok == 0u || pos0 + n_tok > cap) return 0;
    k3m_store_args a = { pos0, heads, nope, rot, kv_lora, vdim };
    return k3m_run("kernel_k3_mla_store", ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        K3M_BUF(enc, kc, 1); K3M_BUF(enc, vc, 2); K3M_BUF(enc, k_nope, 3); K3M_BUF(enc, kv_raw, 4); K3M_BUF(enc, value, 5);
        [enc dispatchThreads:MTLSizeMake(heads * (nope + rot + vdim), n_tok, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }, "K3 MLA store");
}
int ds4_gpu_k3_mla_store_tensor(ds4_gpu_tensor *key_cache, ds4_gpu_tensor *value_cache, const ds4_gpu_tensor *k_nope, const ds4_gpu_tensor *kv_raw,
                                const ds4_gpu_tensor *value, uint32_t pos, uint32_t cache_cap, uint32_t heads, uint32_t nope, uint32_t rot,
                                uint32_t kv_lora, uint32_t vdim) {
    return k3m_store(key_cache, value_cache, k_nope, kv_raw, value, pos, 1u, cache_cap, heads, nope, rot, kv_lora, vdim);
}
int ds4_gpu_k3_mla_store_batch_tensor(ds4_gpu_tensor *key_cache, ds4_gpu_tensor *value_cache, const ds4_gpu_tensor *k_nope, const ds4_gpu_tensor *kv_raw,
                                      const ds4_gpu_tensor *value, uint32_t pos0, uint32_t n_tok, uint32_t cache_cap, uint32_t heads, uint32_t nope,
                                      uint32_t rot, uint32_t kv_lora, uint32_t vdim) {
    return k3m_store(key_cache, value_cache, k_nope, kv_raw, value, pos0, n_tok, cache_cap, heads, nope, rot, kv_lora, vdim);
}

/* ---- MLA attention, one query (decode); the batch launchers run it row by row (the CUDA verify path) ---- */
typedef struct { uint32_t n_pos, heads, qk, vdim; float scale; uint32_t use_mask; } k3m_attn_args;
static id<MTLBuffer> g_k3m_attn_scratch = nil;
static int k3m_attend(ds4_gpu_tensor *heads_out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *kc, const ds4_gpu_tensor *vc, const ds4_gpu_tensor *mask,
                      uint32_t n_pos, uint32_t heads, uint32_t qk, uint32_t vdim) {
    if (!heads_out || !q || !kc || !vc || n_pos == 0u) return 0;
    const size_t need = (size_t)heads * n_pos * sizeof(float);
    if (!g_k3m_attn_scratch || g_k3m_attn_scratch.length < need) {
        /* sized once for 32k positions (the command buffers use unretained references: never replace a buffer in flight) */
        const size_t want = need > (size_t)heads * 32768u * sizeof(float) ? need : (size_t)heads * 32768u * sizeof(float);
        if (!ds4_gpu_synchronize()) return 0;
        g_k3m_attn_scratch = [g_device newBufferWithLength:want options:MTLResourceStorageModeShared];
        if (!g_k3m_attn_scratch) return 0;
    }
    k3m_attn_args a = { n_pos, heads, qk, vdim, 1.0f / sqrtf((float)qk), mask ? 1u : 0u };
    const ds4_gpu_tensor *mk = mask ? mask : q;
    return k3m_run("kernel_k3_mla_attend", ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        K3M_BUF(enc, heads_out, 1); K3M_BUF(enc, q, 2); K3M_BUF(enc, kc, 3); K3M_BUF(enc, vc, 4);
        [enc setBuffer:g_k3m_attn_scratch offset:0 atIndex:5];
        K3M_BUF(enc, mk, 6);
        [enc dispatchThreadgroups:MTLSizeMake(heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }, "K3 MLA attention");
}
int ds4_gpu_k3_mla_attend_tensor(ds4_gpu_tensor *heads_out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *key_cache, const ds4_gpu_tensor *value_cache,
                                 uint32_t n_pos, uint32_t heads, uint32_t qk, uint32_t vdim) {
    return k3m_attend(heads_out, q, key_cache, value_cache, NULL, n_pos, heads, qk, vdim);
}
int ds4_gpu_k3_mla_attend_masked_tensor(ds4_gpu_tensor *heads_out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *key_cache, const ds4_gpu_tensor *value_cache,
                                        const ds4_gpu_tensor *mask, uint32_t n_pos, uint32_t heads, uint32_t qk, uint32_t vdim) {
    return k3m_attend(heads_out, q, key_cache, value_cache, mask, n_pos, heads, qk, vdim);
}
/* a view of rows of a tensor (Metal tensors are objects: views come from the public API and are freed after use) */
#define K3M_VIEWS 8
typedef struct { ds4_gpu_tensor *v[K3M_VIEWS]; int n; } k3m_views;
static ds4_gpu_tensor *k3m_view(k3m_views *vs, const ds4_gpu_tensor *t, uint64_t off, uint64_t bytes) {
    ds4_gpu_tensor *v = t ? ds4_gpu_tensor_view(t, off, bytes) : NULL;
    if (v && vs->n < K3M_VIEWS) vs->v[vs->n++] = v;
    return v;
}
static void k3m_views_free(k3m_views *vs) { for (int i = 0; i < vs->n; i++) ds4_gpu_tensor_free(vs->v[i]); vs->n = 0; }
int ds4_gpu_k3_mla_attend_batch_tensor(ds4_gpu_tensor *heads_out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *key_cache, const ds4_gpu_tensor *value_cache,
                                       uint32_t pos0, uint32_t n_tok, uint32_t heads, uint32_t qk, uint32_t vdim) {
    for (uint32_t t = 0; t < n_tok; t++) {
        k3m_views vs = {{0}, 0};
        ds4_gpu_tensor *ho = k3m_view(&vs, heads_out, (uint64_t)t * heads * vdim * 4u, (uint64_t)heads * vdim * 4u);
        ds4_gpu_tensor *qo = k3m_view(&vs, q, (uint64_t)t * heads * qk * 4u, (uint64_t)heads * qk * 4u);
        const int ok = ho && qo && k3m_attend(ho, qo, key_cache, value_cache, NULL, pos0 + t + 1u, heads, qk, vdim);
        k3m_views_free(&vs);
        if (!ok) return 0;
    }
    return 1;
}
int ds4_gpu_k3_mla_attend_batch_masked_tensor(ds4_gpu_tensor *heads_out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *key_cache,
                                              const ds4_gpu_tensor *value_cache, const ds4_gpu_tensor *mask, uint32_t pos0, uint32_t n_tok,
                                              uint32_t mask_stride, uint32_t top_k, uint32_t heads, uint32_t qk, uint32_t vdim) {
    for (uint32_t t = 0; t < n_tok; t++) {
        k3m_views vs = {{0}, 0};
        ds4_gpu_tensor *ho = k3m_view(&vs, heads_out, (uint64_t)t * heads * vdim * 4u, (uint64_t)heads * vdim * 4u);
        ds4_gpu_tensor *qo = k3m_view(&vs, q, (uint64_t)t * heads * qk * 4u, (uint64_t)heads * qk * 4u);
        ds4_gpu_tensor *mo = k3m_view(&vs, mask, (uint64_t)t * mask_stride, mask_stride);
        const uint32_t np = pos0 + t + 1u;
        const int ok = ho && qo && mo && k3m_attend(ho, qo, key_cache, value_cache, np <= top_k ? NULL : mo, np, heads, qk, vdim);
        k3m_views_free(&vs);
        if (!ok) return 0;
    }
    return 1;
}

/* ---- Kimi Delta Attention ---- */
typedef struct { uint32_t n_heads; float lower_bound, norm_eps; uint32_t per_channel; } k3m_kda_args;
int ds4_gpu_k3_kda_decode(ds4_gpu_tensor *out, ds4_gpu_tensor *conv_state, ds4_gpu_tensor *recurrent_state, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k,
                          const ds4_gpu_tensor *v, const ds4_gpu_tensor *raw_gate, const ds4_gpu_tensor *raw_beta, const ds4_gpu_tensor *output_gate,
                          const void *model_map, uint64_t model_size, uint64_t q_conv_offset, uint64_t k_conv_offset, uint64_t v_conv_offset,
                          uint64_t a_log_offset, uint64_t dt_bias_offset, uint64_t output_norm_offset, uint32_t n_heads, float gate_lower_bound,
                          float norm_eps, int a_log_per_channel) {
    if (!out || !conv_state || !recurrent_state || !q || !k || !v || !raw_gate || !raw_beta || !output_gate || n_heads == 0u) return 0;
    @autoreleasepool {
        const uint64_t projection = (uint64_t)n_heads * 128u;
        uint64_t oq = 0, ok_ = 0, ov = 0, oa = 0, od = 0, on = 0;
        id<MTLBuffer> wq = k3m_model(model_map, model_size, q_conv_offset, projection * 16u, &oq);
        id<MTLBuffer> wk = k3m_model(model_map, model_size, k_conv_offset, projection * 16u, &ok_);
        id<MTLBuffer> wv = k3m_model(model_map, model_size, v_conv_offset, projection * 16u, &ov);
        id<MTLBuffer> wa = k3m_model(model_map, model_size, a_log_offset, (uint64_t)(a_log_per_channel ? 128u : n_heads) * 4u, &oa);
        id<MTLBuffer> wd = k3m_model(model_map, model_size, dt_bias_offset, projection * 4u, &od);
        id<MTLBuffer> wn = k3m_model(model_map, model_size, output_norm_offset, 128u * 4u, &on);
        if (!wq || !wk || !wv || !wa || !wd || !wn) return 0;
        k3m_kda_args a = { n_heads, gate_lower_bound, norm_eps, a_log_per_channel ? 1u : 0u };
        return k3m_run("kernel_k3_kda_decode", ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBytes:&a length:sizeof(a) atIndex:0];
            K3M_BUF(enc, out, 1); K3M_BUF(enc, conv_state, 2); K3M_BUF(enc, recurrent_state, 3);
            K3M_BUF(enc, q, 4); K3M_BUF(enc, k, 5); K3M_BUF(enc, v, 6); K3M_BUF(enc, raw_gate, 7); K3M_BUF(enc, raw_beta, 8); K3M_BUF(enc, output_gate, 9);
            [enc setBuffer:wq offset:(NSUInteger)oq atIndex:10]; [enc setBuffer:wk offset:(NSUInteger)ok_ atIndex:11];
            [enc setBuffer:wv offset:(NSUInteger)ov atIndex:12]; [enc setBuffer:wa offset:(NSUInteger)oa atIndex:13];
            [enc setBuffer:wd offset:(NSUInteger)od atIndex:14]; [enc setBuffer:wn offset:(NSUInteger)on atIndex:15];
            [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }, "K3 KDA decode");
    }
}
int ds4_gpu_k3_kda_decode_batch(ds4_gpu_tensor *out, ds4_gpu_tensor *conv_state, ds4_gpu_tensor *recurrent_state, const ds4_gpu_tensor *q,
                                const ds4_gpu_tensor *k, const ds4_gpu_tensor *v, const ds4_gpu_tensor *raw_gate, const ds4_gpu_tensor *raw_beta,
                                const ds4_gpu_tensor *output_gate, const void *model_map, uint64_t model_size, uint64_t q_conv_offset,
                                uint64_t k_conv_offset, uint64_t v_conv_offset, uint64_t a_log_offset, uint64_t dt_bias_offset,
                                uint64_t output_norm_offset, uint32_t n_heads, float gate_lower_bound, float norm_eps, int a_log_per_channel,
                                uint32_t n_tok) {
    const uint64_t proj = (uint64_t)n_heads * 128u * 4u, heads_b = (uint64_t)n_heads * 4u;
    for (uint32_t t = 0; t < n_tok; t++) {   /* the decode step token by token (the recurrence is sequential) */
        k3m_views vs = {{0}, 0};
        ds4_gpu_tensor *o = k3m_view(&vs, out, t * proj, proj), *qq = k3m_view(&vs, q, t * proj, proj), *kk = k3m_view(&vs, k, t * proj, proj);
        ds4_gpu_tensor *vv = k3m_view(&vs, v, t * proj, proj), *rg = k3m_view(&vs, raw_gate, t * proj, proj);
        ds4_gpu_tensor *rb = k3m_view(&vs, raw_beta, t * heads_b, heads_b), *og = k3m_view(&vs, output_gate, t * proj, proj);
        const int ok = o && qq && kk && vv && rg && rb && og &&
                       ds4_gpu_k3_kda_decode(o, conv_state, recurrent_state, qq, kk, vv, rg, rb, og, model_map, model_size, q_conv_offset, k_conv_offset,
                                             v_conv_offset, a_log_offset, dt_bias_offset, output_norm_offset, n_heads, gate_lower_bound, norm_eps,
                                             a_log_per_channel);
        k3m_views_free(&vs);
        if (!ok) return 0;
    }
    return 1;
}

/* ---- routed experts: a unified-memory slab cache ---- */
typedef struct { uint32_t ff, latent; uint64_t w1p, w1s, w2p, w2s, w3p, w3s; float b, bl, weight; } k3m_moe_args;
typedef struct { uint32_t n_part, latent; } k3m_reduce_args;
static struct {
    int ready;
    uint32_t ff, latent, n_slabs, per_buf;
    uint64_t file_bytes, w1p, w1s, w2p, w2s, w3p, w3s;
    NSMutableArray<id<MTLBuffer>> *bufs;
    int32_t *slab_of;        /* (layer << 10 | expert) -> slab or -1 */
    uint32_t *key_of;        /* slab -> key or 0xffffffff */
    uint64_t *last_use, tick;
    uint64_t *epoch_of;      /* slab -> g_k3m_epoch of its last use in an encoded batch */
    _Atomic int *inflight;   /* slab -> 1 while a background prefetch reads into it, 2 if that read failed */
    float *val_of;           /* slab -> keep value for the heat eviction (step M9b) */
    id<MTLBuffer> mid, part;
    uint64_t hits, misses, bytes;
    double wait_s;
} g_k3m;
static int k3m_pool_init(uint32_t ff, uint32_t latent) {
    if (g_k3m.ready) return g_k3m.ff == ff && g_k3m.latent == latent;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint64_t gu_packed = (uint64_t)ff * latent / 2u, gu_scale = (uint64_t)ff * latent / 32u;
    g_k3m.ff = ff; g_k3m.latent = latent;
    g_k3m.w1p = 0; g_k3m.w1s = gu_packed; g_k3m.w2p = g_k3m.w1s + gu_scale; g_k3m.w2s = g_k3m.w2p + gu_packed;
    g_k3m.w3p = g_k3m.w2s + gu_scale; g_k3m.w3s = g_k3m.w3p + gu_packed; g_k3m.file_bytes = g_k3m.w3s + gu_scale;
    const uint64_t src = ds4_k3_expert_source_file_bytes();
    if (src && src != g_k3m.file_bytes) { fprintf(stderr, "ds4: K3 on Metal: expert files are %llu bytes, expected %llu\n", (unsigned long long)src, (unsigned long long)g_k3m.file_bytes); return 0; }
    const char *gb = getenv("DS4_K3_CACHE_RAM_GB");
    const uint64_t budget = (uint64_t)(gb && atof(gb) > 0 ? atof(gb) : 40.0) * (1ull << 30);
    g_k3m.per_buf = 64u;
    g_k3m.n_slabs = (uint32_t)(budget / g_k3m.file_bytes);
    if (g_k3m.n_slabs < 64u) g_k3m.n_slabs = 64u;
    g_k3m.n_slabs = (g_k3m.n_slabs + g_k3m.per_buf - 1u) / g_k3m.per_buf * g_k3m.per_buf;
    g_k3m.bufs = [NSMutableArray array];
    for (uint32_t i = 0; i < g_k3m.n_slabs / g_k3m.per_buf; i++) {
        id<MTLBuffer> b = [g_device newBufferWithLength:(NSUInteger)(g_k3m.file_bytes * g_k3m.per_buf) options:MTLResourceStorageModeShared];
        if (!b) { fprintf(stderr, "ds4: K3 on Metal: expert cache allocation failed after %u slabs\n", i * g_k3m.per_buf); g_k3m.n_slabs = i * g_k3m.per_buf; break; }
        [g_k3m.bufs addObject:b];
    }
    if (g_k3m.n_slabs < 32u) return 0;
    g_k3m.slab_of = malloc(sizeof(int32_t) * 128u * 1024u);
    g_k3m.key_of = malloc(sizeof(uint32_t) * g_k3m.n_slabs);
    g_k3m.last_use = calloc(g_k3m.n_slabs, sizeof(uint64_t));
    g_k3m.epoch_of = calloc(g_k3m.n_slabs, sizeof(uint64_t));
    g_k3m.inflight = calloc(g_k3m.n_slabs, sizeof(_Atomic int));
    g_k3m.val_of = calloc(g_k3m.n_slabs, sizeof(float));
    if (!g_k3m.inflight || !g_k3m.val_of) return 0;
    if (!g_k3m.slab_of || !g_k3m.key_of || !g_k3m.last_use || !g_k3m.epoch_of) return 0;
    for (uint32_t i = 0; i < g_k3m.n_slabs; i++) g_k3m.epoch_of[i] = UINT64_MAX;
    for (uint32_t i = 0; i < 128u * 1024u; i++) g_k3m.slab_of[i] = -1;
    for (uint32_t i = 0; i < g_k3m.n_slabs; i++) g_k3m.key_of[i] = 0xffffffffu;
    g_k3m.mid = [g_device newBufferWithLength:(NSUInteger)64u * ff * sizeof(float) options:MTLResourceStorageModeShared];
    g_k3m.part = [g_device newBufferWithLength:(NSUInteger)64u * latent * sizeof(float) options:MTLResourceStorageModeShared];
    if (!g_k3m.mid || !g_k3m.part) return 0;
    g_k3m.ready = 1;
    fprintf(stderr, "ds4: K3 on Metal: expert cache of %u slabs (%.1f GB of unified memory), files read straight into it\n",
            g_k3m.n_slabs, (double)g_k3m.n_slabs * g_k3m.file_bytes / 1e9);
    return 1;
}
static uint8_t *k3m_slab_ptr(uint32_t s, id<MTLBuffer> *buf, NSUInteger *off) {
    id<MTLBuffer> b = g_k3m.bufs[s / g_k3m.per_buf];
    const NSUInteger o = (NSUInteger)((uint64_t)(s % g_k3m.per_buf) * g_k3m.file_bytes);
    if (buf) *buf = b;
    if (off) *off = o;
    return (uint8_t *)[b contents] + o;
}
int ds4_gpu_k3_pool_prepare(uint32_t ff, uint32_t latent, uint32_t n_tok, uint32_t n_used) { (void)n_tok; (void)n_used; return k3m_pool_init(ff, latent); }
int ds4_gpu_k3_pool_prepare_staging(uint32_t ff, uint32_t latent, uint32_t n_tok, uint32_t n_used, const void *model_map, const uint64_t *offsets,
                                    const uint64_t *sizes, uint32_t n_ranges) {
    (void)model_map; (void)offsets; (void)sizes; (void)n_ranges;   /* unified memory: the mapped spine is read in place */
    return ds4_gpu_k3_pool_prepare(ff, latent, n_tok, n_used);
}
/* step M4: prefetch of the next layer's predicted experts. The driver's predictor (layer il+1's router applied to
   layer il's input) hands its ranked top-32 before layer il's experts are read; once those reads are done, while the
   GPU runs the rest of the layer, up to DS4_K3_METAL_PREFETCH (default 4, 0 = off) predicted experts that are not
   cached are read in the background into slabs marked in flight. A layer that routes to one waits for that read. */
static struct { int k, rank, cold_only; int have; uint32_t layer; int32_t hint[32]; uint32_t n; dispatch_group_t group; uint64_t issued, adopted, dual_rr, used, wasted; uint8_t *pf; } g_k3p = { .k = -1 };
static void k3p_init(void) {
    if (g_k3p.k >= 0) return;
    const char *e = getenv("DS4_K3_METAL_PREFETCH");
    g_k3p.k = e ? atoi(e) : 8;   /* 8 per layer in verify blocks (step M24) */
    if (g_k3p.k < 0) g_k3p.k = 0;
    if (g_k3p.k > 32) g_k3p.k = 32;
    const char *r = getenv("DS4_K3_METAL_PREFETCH_RANK");   /* only candidates ranked above this (precision falls with rank) */
    g_k3p.rank = r ? atoi(r) : 32;
    if (g_k3p.rank < 1) g_k3p.rank = 1;
    if (g_k3p.rank > 32) g_k3p.rank = 32;
    const char *co = getenv("DS4_K3_METAL_PREFETCH_COLD");   /* step M7, default on: only experts with no home on the SSDs (read
                                                               from the checkpoint drive, whose bandwidth is otherwise idle) */
    g_k3p.cold_only = co ? atoi(co) != 0 : 0;
    g_k3p.group = dispatch_group_create();
    fprintf(stderr, "ds4: K3 on Metal: next-layer prefetch of up to %d predicted experts per layer (ranks < %d%s)\n", g_k3p.k, g_k3p.rank,
            g_k3p.cold_only ? ", checkpoint-only experts" : "");
}
/* step M9: heat-aware eviction. DS4_K3_HEAT=<histogram from tools/kimi-k3/heat_from_routes.py> gives each expert its per-token
   use probability p (16 x heat / layer sum); the victim is the slab with the lowest sqrt(p) / (1 + age / 92), age in layer calls
   (92 = one token). Simulated on recorded routes: 53.8 % misses against LRU's 57.0 %. Without the file (or
   DS4_K3_METAL_EVICT=lru): least recently used. */
static float *g_k3m_psq = NULL;   /* (layer << 10 | expert) -> sqrt(p) */
static int g_k3m_evict = -1;      /* 0 LRU, 1 heat */
static int g_k3m_verify = 0;      /* step M14: a drafter verify block is running (ds4_gpu_k3_set_verify) */
static void k3m_evict_init(void) {
    if (g_k3m_evict >= 0) return;
    g_k3m_evict = 0;
    const char *mode = getenv("DS4_K3_METAL_EVICT"), *path = getenv("DS4_K3_HEAT");
    if ((mode && strcmp(mode, "lru") == 0) || !path || !path[0]) { fprintf(stderr, "ds4: K3 on Metal: slab eviction LRU\n"); return; }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ds4: K3 on Metal: heat file %s missing: slab eviction LRU\n", path); return; }
    static float h[82432];
    const int ok = fseek(f, 48, SEEK_SET) == 0 && fread(h, sizeof(float), 82432, f) == 82432;
    fclose(f);
    if (!ok) { fprintf(stderr, "ds4: K3 on Metal: heat file %s unreadable: slab eviction LRU\n", path); return; }
    g_k3m_psq = calloc(128u * 1024u, sizeof(float));
    if (!g_k3m_psq) return;
    for (uint32_t L = 0; L < 92u; L++) {
        double sum = 0.0;
        for (uint32_t e = 0; e < 896u; e++) sum += h[L * 896u + e];
        for (uint32_t e = 0; e < 896u; e++) g_k3m_psq[((L + 1u) << 10) | e] = sum > 0 ? (float)sqrt(16.0 * h[L * 896u + e] / sum) : 0.0f;
    }
    g_k3m_evict = 1;
    fprintf(stderr, "ds4: K3 on Metal: slab eviction by heat and recency (%s)\n", path);
}
/* step M9b: an expert with no home on the SSDs (read from the checkpoint drive at ~1 GB/s) keeps a fixed value of 2 (above every
   sqrt(p)), so the session's cold experts stay in memory; simulated on recorded routes: modelled read time -4.5 % (record 128) and
   -6.7 % (record 200) against LRU, where plain heat eviction was +3 / +4 % (it evicted the cold experts first) */
static float k3m_keep_value(uint32_t layer, uint32_t e) {
    if (!g_k3m_psq) return 0.0f;
    if (!ds4_k3_expert_source_has_tier(layer, e, 0) && !ds4_k3_expert_source_has_tier(layer, e, 1)) return 2.0f;
    return g_k3m_psq[(layer << 10) | e];
}
/* the slab to refill: no read, no encoded GPU work and no lookup of this tick may hold it; -1 if none */
static int32_t k3m_victim(int for_prefetch) {
    k3m_evict_init();
    uint32_t best = 0; int found = 0;
    if (g_k3m_evict == 1) {
        float bs = 0.0f;
        for (uint32_t t = 0; t < g_k3m.n_slabs; t++) {
            if (g_k3m.last_use[t] == g_k3m.tick) continue;
            if (atomic_load_explicit(&g_k3m.inflight[t], memory_order_acquire) == 1) continue;
            if (for_prefetch && g_batch_cb && g_k3m.epoch_of[t] == g_k3m_epoch) continue;
            const uint32_t key = g_k3m.key_of[t];
            const float sc = key == 0xffffffffu ? -1.0f : g_k3m.val_of[t] / (1.0f + (float)(g_k3m.tick - g_k3m.last_use[t]) / 92.0f);
            if (!found || sc < bs) { bs = sc; best = t; found = 1; }
        }
        return found ? (int32_t)best : -1;
    }
    uint64_t bu = UINT64_MAX;
    for (uint32_t t = 0; t < g_k3m.n_slabs; t++) {
        if (g_k3m.last_use[t] >= bu || g_k3m.last_use[t] == g_k3m.tick) continue;
        if (atomic_load_explicit(&g_k3m.inflight[t], memory_order_acquire) == 1) continue;
        if (for_prefetch && g_batch_cb && g_k3m.epoch_of[t] == g_k3m_epoch) continue;
        bu = g_k3m.last_use[t]; best = t; found = 1;
    }
    return found ? (int32_t)best : -1;
}
static void k3p_issue_k(uint32_t layer, int kmax);
static void k3p_issue(uint32_t layer) {   /* a plain step: DS4_K3_METAL_PREFETCH_ROW1 (default 0) predicted experts */
    static int k1 = -1;
    if (k1 < 0) { const char *e = getenv("DS4_K3_METAL_PREFETCH_ROW1"); k1 = e ? atoi(e) : 2; }
    k3p_issue_k(layer, k1);
}
static void k3p_issue_k(uint32_t layer, int kmax) {
    if (!g_k3p.have || g_k3p.layer != layer || g_k3p.k == 0) return;
    if (kmax > g_k3p.k) kmax = g_k3p.k;
    g_k3p.have = 0;
    const uint64_t S = g_k3m.file_bytes;
    int issued = 0;
    double pfl[3] = { 0.0, 0.0, 0.0 };   /* step M24: bytes given to each home in this call */
    static double prate[3] = { 0.0, 0.0, 0.0 };
    if (prate[0] == 0.0) {
        const char *ea = getenv("DS4_K3_RATE_A_GBS"), *eb = getenv("DS4_K3_RATE_B_GBS"), *ec = getenv("DS4_K3_RATE_C_GBS");
        prate[0] = ea && atof(ea) > 0 ? atof(ea) : 13.7; prate[1] = eb && atof(eb) > 0 ? atof(eb) : 7.1; prate[2] = ec && atof(ec) > 0 ? atof(ec) : 6.0;
    }
    for (uint32_t r = 0; r < g_k3p.n && r < (uint32_t)g_k3p.rank && issued < kmax; r++) {
        const int32_t e = g_k3p.hint[r];
        if (e < 0 || e >= 1024) continue;
        const uint32_t key = (layer << 10) | (uint32_t)e;
        if (g_k3m.slab_of[key] >= 0) continue;   /* cached or already in flight */
        if (g_k3p.cold_only && (ds4_k3_expert_source_has_tier(layer, (uint32_t)e, 0) || ds4_k3_expert_source_has_tier(layer, (uint32_t)e, 1))) continue;
        const int32_t v = k3m_victim(1);
        if (v < 0) break;
        if (g_k3m.key_of[v] != 0xffffffffu) g_k3m.slab_of[g_k3m.key_of[v]] = -1;
        g_k3m.key_of[v] = key; g_k3m.slab_of[key] = v; g_k3m.last_use[v] = g_k3m.tick;
        g_k3m.val_of[v] = g_k3m_verify ? 0.0f : k3m_keep_value(layer, (uint32_t)e);   /* step M24: transient in a block */
        atomic_store_explicit(&g_k3m.inflight[v], 1, memory_order_release);
        if (!g_k3p.pf) g_k3p.pf = calloc(g_k3m.n_slabs, 1);
        if (g_k3p.pf) { if (g_k3p.pf[v]) g_k3p.wasted++; g_k3p.pf[v] = 1; }   /* stats: a prefetched slab until first use */
        const int ha = ds4_k3_expert_source_has_tier(layer, (uint32_t)e, 0), hb = ds4_k3_expert_source_has_tier(layer, (uint32_t)e, 1);
        int tier = (hb && !ha) ? 1 : (ha && hb) ? ((g_k3p.dual_rr++ % 3u) == 2u ? 1 : 0) : (ha ? 0 : -1);
        if (ds4_k3_expert_source_has_dir_c()) {   /* step M24: the home that finishes this call's prefetches first */
            const int hh[3] = { ha, hb, ds4_k3_expert_source_has_tier(layer, (uint32_t)e, 2) };
            double best = 1e300; tier = -1;
            for (int d = 0; d < 3; d++) if (hh[d]) { const double f = (pfl[d] + (double)S) / prate[d]; if (f < best) { best = f; tier = d; } }
            if (tier >= 0) pfl[tier] += (double)S;
        }
        uint8_t *dst = k3m_slab_ptr((uint32_t)v, NULL, NULL);
        _Atomic int *flag = &g_k3m.inflight[v];
        const uint32_t ee = (uint32_t)e;
        dispatch_group_async(g_k3p.group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            const int ok = tier < 0 ? ds4_k3_expert_source_read_file_tier(layer, ee, dst, 1, 0)
                                    : ds4_k3_expert_source_read_range_tier(layer, ee, dst, 0, S, 1, tier, NULL);
            atomic_store_explicit(flag, ok ? 0 : 2, memory_order_release);
        });
        issued++; g_k3p.issued++;
    }
}
static int k3m_cmp_u32(const void *a, const void *b) { const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return x < y ? -1 : x > y; }
/* one token's routed experts: out[latent] = sum, in ascending expert id, of w_e * expert_e(x) (the CUDA deterministic order) */
/* One slab for (layer, e) at the current tick: a hit (waiting for a prefetch that is reading it), or a miss with a victim
   slab assigned; *miss tells which. A key assigned earlier in the same tick is a hit (its read is in the same batch). */
static int k3m_lookup(uint32_t layer, uint32_t e, int32_t *slab, int *miss) {
    const uint32_t key = (layer << 10) | e;
    int32_t s = g_k3m.slab_of[key];
    if (s >= 0) {
        int st = atomic_load_explicit(&g_k3m.inflight[s], memory_order_acquire);
        if (st == 1) {   /* step M4: a prefetch is reading it */
            const double tw = ds4_gpu_now_ms();
            while ((st = atomic_load_explicit(&g_k3m.inflight[s], memory_order_acquire)) == 1) usleep(20);
            g_k3m.wait_s += (ds4_gpu_now_ms() - tw) / 1e3;
            if (st == 0) g_k3p.adopted++;
        }
        if (st == 2) {   /* the prefetch failed: read it as a miss */
            atomic_store_explicit(&g_k3m.inflight[s], 0, memory_order_release);
            g_k3m.slab_of[key] = -1; g_k3m.key_of[s] = 0xffffffffu; g_k3m.last_use[s] = 0;
        } else {
            if (g_k3m.last_use[s] != g_k3m.tick) g_k3m.hits++;
            if (g_k3p.pf && g_k3p.pf[s]) { g_k3p.pf[s] = 0; g_k3p.used++; }
            if (!g_k3m_verify && g_k3m.val_of[s] == 0.0f) g_k3m.val_of[s] = k3m_keep_value(layer, e);   /* step M14: reused, no longer transient */
            g_k3m.last_use[s] = g_k3m.tick; *slab = s; *miss = 0; return 1;
        }
    }
    const int32_t v = k3m_victim(0);
    if (v < 0) return 0;
    if (g_k3p.pf && g_k3p.pf[v]) { g_k3p.pf[v] = 0; g_k3p.wasted++; }
    if (g_k3m.key_of[v] != 0xffffffffu) g_k3m.slab_of[g_k3m.key_of[v]] = -1;
    g_k3m.key_of[v] = key; g_k3m.slab_of[key] = v; g_k3m.last_use[v] = g_k3m.tick;
    g_k3m.val_of[v] = k3m_keep_value(layer, e);
    g_k3m.misses++;
    *slab = v; *miss = 1;
    return 1;
}
/* Read n missed experts into their slabs, all at once. Step M3: an expert with one home is read there; the experts on
   both the primary directory (tier 0) and dir_b (tier 1) are split at one 16 KB-aligned cut so both drives finish together at
   their rates (DS4_K3_RATE_A_GBS / DS4_K3_RATE_B_GBS, default 14 / 7; DS4_K3_BALANCE=0 reads the first home only). */
typedef void (^k3m_ready_fn)(const uint32_t *ms, uint32_t k, int last);   /* step M25b: miss indices whose reads completed */
static int k3m_read_misses_cb(uint32_t layer, const uint32_t *exps, const int32_t *slabs, uint32_t n, k3m_ready_fn ready, uint32_t chunk);
static int k3m_read_misses(uint32_t layer, const uint32_t *exps, const int32_t *slabs, uint32_t n) {
    return k3m_read_misses_cb(layer, exps, slabs, n, nil, 0u);
}
static int k3m_read_misses_cb(uint32_t layer, const uint32_t *exps, const int32_t *slabs, uint32_t n, k3m_ready_fn ready, uint32_t chunk) {
    if (n == 0u) return 1;
    int busy = g_k3m_batching != 1;   /* unbatched: synchronize as before */
    for (uint32_t m = 0; m < n && !busy; m++) busy = g_batch_cb && g_k3m.epoch_of[slabs[m]] == g_k3m_epoch;
    if (busy && !ds4_gpu_synchronize()) return 0;   /* nothing may still read a slab being refilled */
    static int bal = -1, three = 0; static double ra = 13.7, rb = 7.1, rc = 6.0;
    if (bal < 0) {
        const char *e0 = getenv("DS4_K3_BALANCE"), *ea = getenv("DS4_K3_RATE_A_GBS"), *eb = getenv("DS4_K3_RATE_B_GBS"), *ec = getenv("DS4_K3_RATE_C_GBS");
        bal = (e0 && strcmp(e0, "0") == 0) ? 0 : 1;
        if (ea && atof(ea) > 0) ra = atof(ea);
        if (eb && atof(eb) > 0) rb = atof(eb);
        if (ec && atof(ec) > 0) rc = atof(ec);
        three = bal && ds4_k3_expert_source_has_dir_c();
        if (three) fprintf(stderr, "ds4: K3 on Metal: reads balanced over three homes (primary %.1f GB/s, dir_b %.1f GB/s, dir_c %.1f GB/s; step M23)\n", ra, rb, rc);
        else fprintf(stderr, "ds4: K3 on Metal: reads %s (primary %.1f GB/s, dir_b %.1f GB/s)\n", bal ? "balanced over both homes" : "from the first home", ra, rb);
    }
    typedef struct { uint32_t e; int tier; uint64_t off, len; uint8_t *dst; uint32_t m; } k3m_job;
    k3m_job *jobs = malloc(sizeof(k3m_job) * 3u * n);
    uint32_t *dual = malloc(sizeof(uint32_t) * n);
    if (!jobs || !dual) { free(jobs); free(dual); return 0; }
    uint32_t nj = 0, nd = 0;
    const uint64_t S = g_k3m.file_bytes;
    double ba = 0.0, bb = 0.0;
    if (three) {   /* step M23: water-fill every expert into its homes, fewest homes first */
        const double rate[3] = { ra, rb, rc };
        double load[3] = { 0.0, 0.0, 0.0 };
        for (int want = 0; want <= 3; want++)
            for (uint32_t m = 0; m < n; m++) {
                const int h0 = ds4_k3_expert_source_has_tier(layer, exps[m], 0), h1 = ds4_k3_expert_source_has_tier(layer, exps[m], 1),
                          h2 = ds4_k3_expert_source_has_tier(layer, exps[m], 2);
                const int nh = h0 + h1 + h2;
                if (nh != want) continue;
                uint8_t *dst = k3m_slab_ptr((uint32_t)slabs[m], NULL, NULL);
                if (nh == 0) { jobs[nj++] = (k3m_job){ exps[m], -1, 0, S, dst, m }; continue; }
                int hs[3], k = 0;
                if (h0) hs[k++] = 0;
                if (h1) hs[k++] = 1;
                if (h2) hs[k++] = 2;
                for (int i = 1; i < k; i++)   /* by projected finish time */
                    for (int j = i; j > 0 && load[hs[j]] / rate[hs[j]] < load[hs[j - 1]] / rate[hs[j - 1]]; j--) { const int t = hs[j]; hs[j] = hs[j - 1]; hs[j - 1] = t; }
                int use = k; double L = 0.0;
                for (int q = 1; q <= k; q++) {
                    double rs = 0.0, fs = 0.0;
                    for (int i = 0; i < q; i++) { rs += rate[hs[i]]; fs += load[hs[i]]; }
                    L = ((double)S + fs) / rs;
                    if (q == k || L <= load[hs[q]] / rate[hs[q]]) { use = q; break; }
                }
                uint64_t off = 0;
                for (int i = 0; i < use; i++) {
                    uint64_t len;
                    if (i == use - 1) len = S - off;
                    else {
                        const double x = rate[hs[i]] * L - load[hs[i]];
                        uint64_t c = (uint64_t)(x / 16384.0 + 0.5) * 16384u;
                        if (c > S - off) c = S - off;
                        len = c;
                    }
                    if (len == 0) continue;
                    jobs[nj++] = (k3m_job){ exps[m], hs[i], off, len, dst, m };
                    load[hs[i]] += (double)len;
                    off += len;
                }
            }
    } else
    for (uint32_t m = 0; m < n; m++) {
        uint8_t *dst = k3m_slab_ptr((uint32_t)slabs[m], NULL, NULL);
        const int ha = ds4_k3_expert_source_has_tier(layer, exps[m], 0), hb = ds4_k3_expert_source_has_tier(layer, exps[m], 1);
        if (bal && ha && hb) { dual[nd++] = m; continue; }
        jobs[nj++] = (k3m_job){ exps[m], (hb && !ha) ? 1 : (ha ? 0 : -1), 0, S, dst, m };
        if (hb && !ha) bb += (double)S; else if (ha) ba += (double)S;
    }
    if (nd && !three) {
        const double Dd = (double)nd * (double)S;
        double xa = (ra * (bb + Dd) - rb * ba) / (ra + rb);   /* bytes of the two-home set read from the internal SSD */
        if (xa < 0.0) xa = 0.0;
        if (xa > Dd) xa = Dd;
        uint64_t cut = (uint64_t)(xa / (double)nd / 16384.0 + 0.5) * 16384u;
        if (cut > S) cut = S;
        for (uint32_t q = 0; q < nd; q++) {
            const uint32_t m = dual[q];
            uint8_t *dst = k3m_slab_ptr((uint32_t)slabs[m], NULL, NULL);
            if (cut > 0) jobs[nj++] = (k3m_job){ exps[m], 0, 0, cut, dst, m };
            if (cut < S) jobs[nj++] = (k3m_job){ exps[m], 1, cut, S - cut, dst, m };
        }
    }
    __block int fail = 0;
    const double t0 = ds4_gpu_now_ms();
    const k3m_job *jp = jobs;
    if (!ready) {
        dispatch_apply(nj, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t m) {
            const k3m_job jb = jp[m];
            const int ok = jb.tier < 0 ? ds4_k3_expert_source_read_file_tier(layer, jb.e, jb.dst, 1, 0)
                                       : ds4_k3_expert_source_read_range_tier(layer, jb.e, jb.dst, jb.off, jb.len, 1, jb.tier, NULL);
            if (!ok) fail = 1;
        });
    } else {   /* step M25b: the reads run on the workers; this thread encodes the completed experts in chunks */
        _Atomic uint32_t *left = calloc(n, sizeof(_Atomic uint32_t));
        uint32_t *doneq = malloc(sizeof(uint32_t) * n);
        if (!left || !doneq) { free(left); free(doneq); free(jobs); free(dual); return 0; }
        for (uint32_t j = 0; j < nj; j++) atomic_fetch_add_explicit(&left[jobs[j].m], 1u, memory_order_relaxed);
        typedef struct { os_unfair_lock lock; uint32_t ndone; } k3m_doneq;   /* on the heap: the workers and this thread share it */
        k3m_doneq *dq = calloc(1, sizeof(k3m_doneq));
        if (!dq) { free(left); free(doneq); free(jobs); free(dual); return 0; }
        dq->lock = OS_UNFAIR_LOCK_INIT;
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        _Atomic uint32_t *lp = left; uint32_t *qp = doneq;
        dispatch_group_t grp = dispatch_group_create();
        dispatch_group_async(grp, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            dispatch_apply(nj, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t j) {
                const k3m_job jb = jp[j];
                const int ok = jb.tier < 0 ? ds4_k3_expert_source_read_file_tier(layer, jb.e, jb.dst, 1, 0)
                                           : ds4_k3_expert_source_read_range_tier(layer, jb.e, jb.dst, jb.off, jb.len, 1, jb.tier, NULL);
                if (!ok) fail = 1;
                if (atomic_fetch_sub_explicit(&lp[jb.m], 1u, memory_order_acq_rel) == 1u) {
                    os_unfair_lock_lock(&dq->lock); qp[dq->ndone++] = jb.m; os_unfair_lock_unlock(&dq->lock);
                    dispatch_semaphore_signal(sem);
                }
            });
        });
        uint32_t consumed = 0, seen = 0;
        if (chunk == 0u) chunk = 1u;
        while (consumed < n) {
            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
            seen++;
            if (seen - consumed < chunk && seen < n) continue;
            os_unfair_lock_lock(&dq->lock); const uint32_t avail = dq->ndone; os_unfair_lock_unlock(&dq->lock);
            if (avail > consumed) { ready(doneq + consumed, avail - consumed, avail == n); consumed = avail; }
        }
        dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
        free(left); free(doneq); free(dq);
    }
    {
        static int tr = -1;
        if (tr < 0) tr = getenv("DS4_K3_METAL_READ_TRACE") != NULL;
        if (tr) {
            double qa = 0.0, qb = 0.0, qc = 0.0, qx = 0.0;
            for (uint32_t m = 0; m < nj; m++) { if (jobs[m].tier == 0) qa += (double)jobs[m].len; else if (jobs[m].tier == 1) qb += (double)jobs[m].len; else if (jobs[m].tier == 2) qc += (double)jobs[m].len; else qx += (double)jobs[m].len; }
            const double ms = ds4_gpu_now_ms() - t0;
            fprintf(stderr, "ds4: K3 read layer %u: %u experts, %u jobs (%u split), primary %.0f MB, dir_b %.0f MB, dir_c %.0f MB, other %.0f MB in %.1f ms = %.1f GB/s\n",
                    layer, n, nj, nd, qa / 1e6, qb / 1e6, qc / 1e6, qx / 1e6, ms, (qa + qb + qc + qx) / 1e6 / (ms > 0 ? ms : 1));
        }
    }
    free(jobs); free(dual);
    g_k3m.wait_s += (ds4_gpu_now_ms() - t0) / 1e3;
    g_k3m.bytes += (uint64_t)n * S;
    if (fail) {
        for (uint32_t m = 0; m < n; m++) { const int32_t sl = slabs[m]; if (g_k3m.key_of[sl] != 0xffffffffu) g_k3m.slab_of[g_k3m.key_of[sl]] = -1; g_k3m.key_of[sl] = 0xffffffffu; g_k3m.last_use[sl] = 0; }
        fprintf(stderr, "ds4: K3 on Metal: expert read failed (layer %u)\n", layer);
        return 0;
    }
    return 1;
}
/* One row's routed experts, encoded: out[latent] = sum, in ascending expert id, of w_e * expert_e(x) (the CUDA
   deterministic order); slab[] / w[] in that order. */
/* step M8: DS4_K3_MOE_V2 (default 1) runs a token's experts as two dispatches (gate-up, down) over all slots with vector loads;
   DS4_K3_MOE_CHECK=<n>: the first n calls also run the per-expert kernels into a side buffer and compare every float */
typedef struct { uint64_t slab[64]; float weight[64]; } k3m_moe_ptrs;
static id<MTLBuffer> g_k3m_part_ref = nil;
static int k3m_moe_v2(void) { static int v = -1; if (v < 0) { const char *e = getenv("DS4_K3_MOE_V2"); v = e ? atoi(e) : 3; } return v; }
static int k3m_moe_check_take(void) {
    static int left = -1;
    if (left < 0) { const char *e = getenv("DS4_K3_MOE_CHECK"); left = e ? atoi(e) : 0; if (left < 0) left = 0; }
    if (left <= 0) return 0;
    left--; return 1;
}
static int k3m_encode_row_v1(id<MTLComputeCommandEncoder> enc, const ds4_gpu_tensor *x, const int32_t *slab, const float *w, uint32_t n,
                             uint32_t ff, uint32_t latent, float beta, float beta_linear, id<MTLBuffer> part) {
    id<MTLComputePipelineState> pg = ds4_gpu_get_pipeline("kernel_k3_moe_gate_up"), pd = ds4_gpu_get_pipeline("kernel_k3_moe_down");
    if (!pg || !pd) return 0;
    for (uint32_t j = 0; j < n; j++) {
        id<MTLBuffer> sb = nil; NSUInteger so = 0;
        (void)k3m_slab_ptr((uint32_t)slab[j], &sb, &so);
        k3m_moe_args a = { ff, latent, g_k3m.w1p, g_k3m.w1s, g_k3m.w2p, g_k3m.w2s, g_k3m.w3p, g_k3m.w3s, beta, beta_linear, w[j] };
        [enc setComputePipelineState:pg];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:sb offset:so atIndex:1];
        K3M_BUF(enc, x, 2);
        [enc setBuffer:g_k3m.mid offset:(NSUInteger)j * ff * sizeof(float) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((ff + 7u) / 8u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc setComputePipelineState:pd];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:sb offset:so atIndex:1];
        [enc setBuffer:g_k3m.mid offset:(NSUInteger)j * ff * sizeof(float) atIndex:2];
        [enc setBuffer:part offset:(NSUInteger)j * latent * sizeof(float) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((latent + 7u) / 8u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    return 1;
}
static int k3m_encode_row_v2(id<MTLComputeCommandEncoder> enc, const ds4_gpu_tensor *x, const int32_t *slab, const float *w, uint32_t n,
                             uint32_t ff, uint32_t latent, float beta, float beta_linear, id<MTLBuffer> part) {
    const int v3 = k3m_moe_v2() >= 3;   /* DS4_K3_MOE_V2=3: x reuse over 4 rows per simdgroup (step M8b) */
    id<MTLComputePipelineState> pg = ds4_gpu_get_pipeline(v3 ? "kernel_k3_moe_gate_up3" : "kernel_k3_moe_gate_up2"),
                                pd = ds4_gpu_get_pipeline(v3 ? "kernel_k3_moe_down3" : "kernel_k3_moe_down2");
    if (!pg || !pd) return 0;
    static int rps_env = -1;
    if (rps_env < 0) { const char *e = getenv("DS4_K3_MOE_RPS"); rps_env = e ? atoi(e) : 4; }
    const uint32_t rps = v3 ? (uint32_t)rps_env : 1u;
    k3m_moe_ptrs ptrs;
    memset(&ptrs, 0, sizeof(ptrs));
    for (uint32_t j = 0; j < n; j++) {
        ptrs.weight[j] = w[j];
        if (slab[j] < 0) { ptrs.slab[j] = 0; continue; }   /* step M20: a slot of the other phase */
        id<MTLBuffer> sb = nil; NSUInteger so = 0;
        (void)k3m_slab_ptr((uint32_t)slab[j], &sb, &so);
        ptrs.slab[j] = (uint64_t)sb.gpuAddress + (uint64_t)so;
        [enc useResource:sb usage:MTLResourceUsageRead];   /* reached through a GPU address: must be made resident */
    }
    k3m_moe_args a = { ff, latent, g_k3m.w1p, g_k3m.w1s, g_k3m.w2p, g_k3m.w2s, g_k3m.w3p, g_k3m.w3s, beta, beta_linear, 0.0f };
    [enc setComputePipelineState:pg];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBytes:&ptrs length:sizeof(ptrs) atIndex:1];
    K3M_BUF(enc, x, 2);
    [enc setBuffer:g_k3m.mid offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((ff + 8u * rps - 1u) / (8u * rps), n, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc setComputePipelineState:pd];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBytes:&ptrs length:sizeof(ptrs) atIndex:1];
    [enc setBuffer:g_k3m.mid offset:0 atIndex:2];
    [enc setBuffer:part offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((latent + 8u * rps - 1u) / (8u * rps), n, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}
static int k3m_encode_row(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const int32_t *slab, const float *w, uint32_t n,
                          uint32_t ff, uint32_t latent, float beta, float beta_linear) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (k3m_moe_v2()) {
        const int check = k3m_moe_check_take();
        if (check && !g_k3m_part_ref) g_k3m_part_ref = [g_device newBufferWithLength:(NSUInteger)64u * latent * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLComputePipelineState> pr = ds4_gpu_get_pipeline("kernel_k3_moe_reduce");
        if (!pr) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (check && g_k3m_part_ref && !k3m_encode_row_v1(enc, x, slab, w, n, ff, latent, beta, beta_linear, g_k3m_part_ref)) return 0;
        if (!k3m_encode_row_v2(enc, x, slab, w, n, ff, latent, beta, beta_linear, g_k3m.part)) return 0;
        for (uint32_t j = 0; j < n; j++) g_k3m.epoch_of[slab[j]] = g_k3m_epoch;
        k3m_reduce_args ra = { n, latent };
        [enc setComputePipelineState:pr];
        [enc setBytes:&ra length:sizeof(ra) atIndex:0];
        [enc setBuffer:g_k3m.part offset:0 atIndex:1];
        K3M_BUF(enc, out, 2);
        [enc dispatchThreads:MTLSizeMake(latent, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        int ok = ds4_gpu_finish_command_buffer(cb, owned, "K3 routed experts");
        if (ok && check && g_k3m_part_ref) {
            if (g_batch_cb && !ds4_gpu_end_commands()) return 0;
            const float *a1 = (const float *)[g_k3m_part_ref contents], *a2 = (const float *)[g_k3m.part contents];
            uint64_t bad = 0, first = 0; const uint64_t nf = (uint64_t)n * latent;
            for (uint64_t i = 0; i < nf; i++) if (memcmp(&a1[i], &a2[i], 4) != 0) { if (!bad) first = i; bad++; }
            static uint64_t calls = 0, floats = 0, diffs = 0;
            calls++; floats += nf; diffs += bad;
            if (bad) fprintf(stderr, "ds4: K3 MoE v2 check: %llu of %llu floats DIFFER (first slot %llu col %llu: %a vs %a)\n", (unsigned long long)bad,
                             (unsigned long long)nf, (unsigned long long)(first / latent), (unsigned long long)(first % latent), (double)a1[first], (double)a2[first]);
            if (calls % 92u == 0u) fprintf(stderr, "ds4: K3 MoE v2 check: %llu calls, %llu floats compared, %llu differ\n",
                                           (unsigned long long)calls, (unsigned long long)floats, (unsigned long long)diffs);
        }
        return ok;
    }
    id<MTLComputePipelineState> pg = ds4_gpu_get_pipeline("kernel_k3_moe_gate_up"), pd = ds4_gpu_get_pipeline("kernel_k3_moe_down"),
                                pr = ds4_gpu_get_pipeline("kernel_k3_moe_reduce");
    if (!pg || !pd || !pr) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    for (uint32_t j = 0; j < n; j++) {
        id<MTLBuffer> sb = nil; NSUInteger so = 0;
        (void)k3m_slab_ptr((uint32_t)slab[j], &sb, &so);
        k3m_moe_args a = { ff, latent, g_k3m.w1p, g_k3m.w1s, g_k3m.w2p, g_k3m.w2s, g_k3m.w3p, g_k3m.w3s, beta, beta_linear, w[j] };
        [enc setComputePipelineState:pg];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:sb offset:so atIndex:1];
        K3M_BUF(enc, x, 2);
        [enc setBuffer:g_k3m.mid offset:(NSUInteger)j * ff * sizeof(float) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((ff + 7u) / 8u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc setComputePipelineState:pd];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:sb offset:so atIndex:1];
        [enc setBuffer:g_k3m.mid offset:(NSUInteger)j * ff * sizeof(float) atIndex:2];
        [enc setBuffer:g_k3m.part offset:(NSUInteger)j * latent * sizeof(float) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((latent + 7u) / 8u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        g_k3m.epoch_of[slab[j]] = g_k3m_epoch;
    }
    k3m_reduce_args ra = { n, latent };
    [enc setComputePipelineState:pr];
    [enc setBytes:&ra length:sizeof(ra) atIndex:0];
    [enc setBuffer:g_k3m.part offset:0 atIndex:1];
    K3M_BUF(enc, out, 2);
    [enc dispatchThreads:MTLSizeMake(latent, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned, "K3 routed experts");
}
/* a row's experts in ascending id, with their weights */
static int k3m_order_row(const int32_t *selected, const float *weights, uint32_t n_used, uint32_t *ord, float *w) {
    for (uint32_t i = 0; i < n_used; i++) {
        if (selected[i] < 0 || selected[i] >= 1024) return 0;
        ord[i] = ((uint32_t)selected[i] << 6) | i;
    }
    qsort(ord, n_used, sizeof(uint32_t), k3m_cmp_u32);
    for (uint32_t j = 0; j < n_used; j++) w[j] = weights[ord[j] & 63u];
    return 1;
}
int ds4_gpu_k3_moe_raw_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const int32_t *selected, const float *weights, uint32_t n_used,
                              uint32_t layer, uint32_t ff, uint32_t latent, float beta, float beta_linear) {
    if (!out || !x || !selected || !weights || n_used == 0u || n_used > 64u || layer >= 128u || !k3m_pool_init(ff, latent)) return 0;
    k3m_batching_init();
    @autoreleasepool {
        uint32_t ord[64], mexp[64]; float w[64]; int32_t slab[64], mslab[64]; uint32_t n_miss = 0;
        if (!k3m_order_row(selected, weights, n_used, ord, w)) return 0;
        g_k3m.tick++;
        for (uint32_t j = 0; j < n_used; j++) {
            int miss = 0;
            if (!k3m_lookup(layer, ord[j] >> 6, &slab[j], &miss)) return 0;
            if (miss) { mexp[n_miss] = ord[j] >> 6; mslab[n_miss++] = slab[j]; }
        }
        static int ovl = -1;
        if (ovl < 0) { const char *e = getenv("DS4_K3_OVERLAP"); ovl = e ? atoi(e) != 0 : 1; }
        if (ovl && g_k3m_batching == 1 && k3m_moe_v2() >= 2 && !getenv("DS4_K3_MOE_CHECK") && n_miss > 0u && n_miss < n_used) {
            /* step M20: the cached experts first, committed without a wait, then the reads, then the missed experts and the reduce */
            int32_t sa[64], sbm[64];
            for (uint32_t j = 0; j < n_used; j++) {
                int m = 0; for (uint32_t q = 0; q < n_miss; q++) if (mslab[q] == slab[j]) { m = 1; break; }
                sa[j] = m ? -1 : slab[j]; sbm[j] = m ? slab[j] : -1;
            }
            int owned = 0;
            id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
            if (!cb || owned) return 0;
            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            if (!k3m_encode_row_v2(enc, x, sa, w, n_used, ff, latent, beta, beta_linear, g_k3m.part)) return 0;
            for (uint32_t j = 0; j < n_used; j++) if (sa[j] >= 0) g_k3m.epoch_of[sa[j]] = g_k3m_epoch;
            ds4_gpu_end_compute_encoder(cb, enc);
            if (!ds4_gpu_flush_commands()) return 0;
            if (!k3m_read_misses(layer, mexp, mslab, n_miss)) return 0;
            id<MTLComputePipelineState> pr = ds4_gpu_get_pipeline("kernel_k3_moe_reduce");
            if (!pr) return 0;
            cb = ds4_gpu_command_buffer(&owned);
            if (!cb || owned) return 0;
            enc = ds4_gpu_compute_encoder(cb);
            if (!k3m_encode_row_v2(enc, x, sbm, w, n_used, ff, latent, beta, beta_linear, g_k3m.part)) return 0;
            for (uint32_t j = 0; j < n_used; j++) if (sbm[j] >= 0) g_k3m.epoch_of[sbm[j]] = g_k3m_epoch;
            k3m_reduce_args ra = { n_used, latent };
            [enc setComputePipelineState:pr];
            [enc setBytes:&ra length:sizeof(ra) atIndex:0];
            [enc setBuffer:g_k3m.part offset:0 atIndex:1];
            K3M_BUF(enc, out, 2);
            [enc dispatchThreads:MTLSizeMake(latent, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
            k3p_init();
            k3p_issue(layer + 1u);
            return 1;
        }
        if (!k3m_read_misses(layer, mexp, mslab, n_miss)) return 0;
        const int ok = k3m_encode_row(out, x, slab, w, n_used, ff, latent, beta, beta_linear);
        k3p_init();
        k3p_issue(layer + 1u);
        return ok;
    }
}
/* step M12: a batch's routed experts as two dispatches over the union of its experts (each weight block loaded once per group of
   tokens), then the per-token reduce in slot order. Per (row, token) the arithmetic is the single-row kernels'. */
typedef struct { uint64_t slab; uint32_t off, count; } k3m_mt_expert;
typedef struct { uint32_t token, slot; float weight; uint32_t pad; } k3m_mt_entry;
static id<MTLBuffer> g_k3m_mt_ex = nil, g_k3m_mt_ent = nil, g_k3m_mt_mid = nil, g_k3m_mt_part = nil;
static int k3m_mt_ensure(__strong id<MTLBuffer> *b, size_t bytes) {
    if (*b && (*b).length >= bytes) return 1;
    *b = [g_device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
    return *b != nil;
}
/* step M20: phase 0 = everything (as before); phase 1 = write the tables, encode the cached experts, commit without a wait;
   phase 2 = encode the missed experts and the reduce (the tables of phase 1). emiss[e]: expert e is read in this batch. */
static uint32_t g_k3m_mt_uh = 0, g_k3m_mt_u = 0, g_k3m_mt_ws[9], g_k3m_mt_nw = 1;   /* step M25: wave starts in ex[] */
static uint64_t g_k3m_mt_tab_epoch = UINT64_MAX;   /* the epoch of the last encode that reads the tables */
static int k3m_encode_batch_mt_ph(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const uint32_t *ord, const float *w, const int32_t *slab,
                                  uint32_t n_tok, uint32_t n_used, uint32_t ff, uint32_t latent, float beta, float beta_linear,
                                  int phase, const uint8_t *emiss, uint32_t wave);
static const uint32_t *g_k3m_mt_elist = NULL; static uint32_t g_k3m_mt_ne = 0;   /* step M25b: the chunk's experts */
static int k3m_encode_batch_mt(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const uint32_t *ord, const float *w, const int32_t *slab,
                               uint32_t n_tok, uint32_t n_used, uint32_t ff, uint32_t latent, float beta, float beta_linear) {
    return k3m_encode_batch_mt_ph(out, x, ord, w, slab, n_tok, n_used, ff, latent, beta, beta_linear, 0, NULL, 0u);
}
static int k3m_encode_batch_mt_ph(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const uint32_t *ord, const float *w, const int32_t *slab,
                                  uint32_t n_tok, uint32_t n_used, uint32_t ff, uint32_t latent, float beta, float beta_linear,
                                  int phase, const uint8_t *emiss, uint32_t wave) {
    static int32_t first[1024], nxt_dummy;
    (void)nxt_dummy;
    uint32_t cnt[1024]; memset(cnt, 0, sizeof(cnt));
    const size_t R = (size_t)n_tok * n_used;
    for (size_t q = 0; q < R; q++) cnt[ord[q] >> 6]++;
    uint32_t off[1025]; off[0] = 0;
    for (uint32_t e = 0; e < 1024u; e++) off[e + 1] = off[e] + cnt[e];
    if (!k3m_mt_ensure(&g_k3m_mt_ent, R * sizeof(k3m_mt_entry)) || !k3m_mt_ensure(&g_k3m_mt_ex, 1024u * sizeof(k3m_mt_expert)) ||
        !k3m_mt_ensure(&g_k3m_mt_mid, (size_t)n_tok * 16u * ff * 4u) || !k3m_mt_ensure(&g_k3m_mt_part, (size_t)n_tok * 16u * latent * 4u)) return 0;
    k3m_mt_entry *ent = (k3m_mt_entry *)[g_k3m_mt_ent contents];
    k3m_mt_expert *ex = (k3m_mt_expert *)[g_k3m_mt_ex contents];
    int32_t slab_of_e[1024]; for (uint32_t e = 0; e < 1024u; e++) slab_of_e[e] = -1;
    for (size_t q = 0; q < R; q++) slab_of_e[ord[q] >> 6] = slab[q];
    uint32_t U = 0, ex0 = 0, ex1 = 0;   /* this call encodes ex[ex0, ex1) */
    if (phase == 4 || phase == 5) {   /* step M25b: append the chunk's experts after the ones encoded so far */
        U = g_k3m_mt_u; ex0 = U;
        for (uint32_t i = 0; i < g_k3m_mt_ne; i++) {
            const uint32_t e = g_k3m_mt_elist[i];
            if (e >= 1024u || !cnt[e]) continue;
            id<MTLBuffer> sb = nil; NSUInteger so = 0;
            (void)k3m_slab_ptr((uint32_t)slab_of_e[e], &sb, &so);
            ex[U++] = (k3m_mt_expert){ (uint64_t)sb.gpuAddress + (uint64_t)so, off[e], cnt[e] };
        }
        g_k3m_mt_u = U; ex1 = U;
    } else
    if (phase != 2 && phase != 3) {
        /* the host writes these buffers: nothing encoded earlier may still read them (phase 1: only work of this epoch can,
           since every earlier encode was completed by a synchronization) */
        if (phase == 0) { if (g_batch_cb && !ds4_gpu_end_commands()) return 0; }
        else if (g_batch_cb && g_k3m_mt_tab_epoch == g_k3m_epoch && !ds4_gpu_end_commands()) return 0;
        uint32_t fill[1024]; memcpy(fill, off, sizeof(fill));
        for (uint32_t t = 0; t < n_tok; t++)
            for (uint32_t j = 0; j < n_used; j++) {
                const size_t q = (size_t)t * n_used + j; const uint32_t e = ord[q] >> 6;
                ent[fill[e]++] = (k3m_mt_entry){ t, j, w[q], 0u };
            }
        uint32_t uh = 0, nw = 1;
        if (phase == 1) for (uint32_t e = 0; e < 1024u; e++) if (cnt[e] && emiss[e] > nw) nw = emiss[e];
        if (nw > 8u) nw = 8u;
        const int passes = phase == 1 ? (wave == 0xffffffffu ? 1 : (int)nw + 1) : 1;   /* M25b (wave all ones): the cached only */
        for (int pass = 0; pass < passes; pass++) {   /* phase 1: the cached experts first, then wave 1, ..., wave W */
            if (phase == 1 && pass >= 1) g_k3m_mt_ws[pass - 1] = U;
            for (uint32_t e = 0; e < 1024u; e++) {
                if (!cnt[e]) continue;
                if (phase == 1 && emiss[e] != (uint8_t)pass) continue;
                id<MTLBuffer> sb = nil; NSUInteger so = 0;
                (void)k3m_slab_ptr((uint32_t)slab_of_e[e], &sb, &so);
                ex[U++] = (k3m_mt_expert){ (uint64_t)sb.gpuAddress + (uint64_t)so, off[e], cnt[e] };
            }
            if (pass == 0) uh = U;
        }
        if (phase == 1) { g_k3m_mt_ws[nw] = U; g_k3m_mt_nw = nw; }
        g_k3m_mt_u = U; g_k3m_mt_uh = uh;
        ex0 = 0; ex1 = phase == 1 ? uh : U;
    } else {
        U = g_k3m_mt_u;
        if (wave >= g_k3m_mt_nw) return 0;
        ex0 = g_k3m_mt_ws[wave]; ex1 = g_k3m_mt_ws[wave + 1u];
    }
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    for (uint32_t e = 0; e < 1024u; e++) {   /* residency and the slab epoch for the experts this call encodes */
        if (!cnt[e]) continue;
        const int m = emiss ? emiss[e] : 0;
        if (phase == 1 && m) continue;
        if ((phase == 2 || phase == 3) && m != (int)wave + 1) continue;
        if (phase == 4 || phase == 5) {
            int in = 0;
            for (uint32_t i = 0; i < g_k3m_mt_ne && !in; i++) in = g_k3m_mt_elist[i] == e;
            if (!in) continue;
        }
        id<MTLBuffer> sb = nil; NSUInteger so = 0;
        (void)k3m_slab_ptr((uint32_t)slab_of_e[e], &sb, &so);
        [enc useResource:sb usage:MTLResourceUsageRead];
        g_k3m.epoch_of[slab_of_e[e]] = g_k3m_epoch;
    }
    const uint32_t UE = ex1 - ex0;
    (void)first;
    static int rps_env = -1;
    if (rps_env < 0) { const char *e = getenv("DS4_K3_MOE_RPS"); rps_env = e ? atoi(e) : 4; }
    uint32_t rps = (uint32_t)rps_env;
    /* step M19: each weight word decoded once per (block, row) for the group's tokens (DS4_K3_MOE_MT_V=1: M12's kernels) */
    static int mtv = -1, rps2 = 4, tg2 = 4;
    if (mtv < 0) {
        const char *e = getenv("DS4_K3_MOE_MT_V"), *er = getenv("DS4_K3_MOE_MT_RPS"), *et = getenv("DS4_K3_MOE_MT_TG");
        mtv = e ? atoi(e) : 2; rps2 = er ? atoi(er) : (mtv == 3 ? 4 : 1); tg2 = et ? atoi(et) : (mtv == 3 ? 2 : 4);
        if (mtv == 3) {
            if (!(((rps2 == 2 || rps2 == 4) && (tg2 == 2 || tg2 == 4)) || (rps2 == 8 && tg2 == 2))) { rps2 = 4; tg2 = 2; }
        } else if (!((rps2 == 1 && tg2 == 4) || ((rps2 == 2 || rps2 == 4) && (tg2 == 4 || tg2 == 8)))) { rps2 = 2; tg2 = 4; }
        fprintf(stderr, "ds4: K3 on Metal: multi-token experts %s\n", mtv == 3 ? "with activations shared by the rows and weights decoded once (step M21)"
                        : mtv == 2 ? "decode each weight word once for the group (step M19)" : "as M12");
        if (mtv >= 2) fprintf(stderr, "ds4: K3 on Metal: multi-token expert shape: %d rows per simdgroup, %d tokens per group\n", rps2, tg2);
    }
    char kgu[64], kdn[64];
    if (mtv == 3) {
        snprintf(kgu, sizeof(kgu), "kernel_k3_moe_gate_up_mt3_r%d_t%d", rps2, tg2); snprintf(kdn, sizeof(kdn), "kernel_k3_moe_down_mt3_r%d_t%d", rps2, tg2);
        rps = (uint32_t)rps2;
    } else if (mtv == 2) {
        snprintf(kgu, sizeof(kgu), "kernel_k3_moe_gate_up_mt2_r%d_t%d", rps2, tg2); snprintf(kdn, sizeof(kdn), "kernel_k3_moe_down_mt2_r%d_t%d", rps2, tg2);
        rps = (uint32_t)rps2;
    } else { snprintf(kgu, sizeof(kgu), "kernel_k3_moe_gate_up_mt"); snprintf(kdn, sizeof(kdn), "kernel_k3_moe_down_mt"); }
    id<MTLComputePipelineState> pg = ds4_gpu_get_pipeline(kgu), pd = ds4_gpu_get_pipeline(kdn),
                                pr = ds4_gpu_get_pipeline("kernel_k3_moe_reduce_mt");
    if (!pg || !pd || !pr) return 0;
    k3m_moe_args a = { ff, latent, g_k3m.w1p, g_k3m.w1s, g_k3m.w2p, g_k3m.w2s, g_k3m.w3p, g_k3m.w3s, beta, beta_linear, 0.0f };
    const NSUInteger exo = (NSUInteger)ex0 * sizeof(k3m_mt_expert);
    if (UE) {
        [enc setComputePipelineState:pg];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:g_k3m_mt_ex offset:exo atIndex:1]; [enc setBuffer:g_k3m_mt_ent offset:0 atIndex:2];
        K3M_BUF(enc, x, 3); [enc setBuffer:g_k3m_mt_mid offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((ff + 8u * rps - 1u) / (8u * rps), UE, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc setComputePipelineState:pd];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:g_k3m_mt_ex offset:exo atIndex:1]; [enc setBuffer:g_k3m_mt_ent offset:0 atIndex:2];
        [enc setBuffer:g_k3m_mt_mid offset:0 atIndex:3]; [enc setBuffer:g_k3m_mt_part offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((latent + 8u * rps - 1u) / (8u * rps), UE, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    if (phase == 0 || phase == 2 || phase == 5) {
        struct { uint32_t n_used, latent; } ra = { n_used, latent };
        [enc setComputePipelineState:pr];
        [enc setBytes:&ra length:sizeof(ra) atIndex:0];
        [enc setBuffer:g_k3m_mt_part offset:0 atIndex:1];
        K3M_BUF(enc, out, 2);
        [enc dispatchThreads:MTLSizeMake(latent, n_tok, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    ds4_gpu_end_compute_encoder(cb, enc);
    g_k3m_mt_tab_epoch = g_k3m_epoch;
    if (phase == 1 || phase == 3 || phase == 4) { if (owned) return 0; return ds4_gpu_flush_commands(); }
    return ds4_gpu_finish_command_buffer(cb, owned, "K3 routed experts (multi-token)");
}
/* step M6: a batch of rows (prompt pass, verify block): one lookup tick for the layer, the union of the rows' missed
   experts read in one parallel batch, then each row encoded exactly as the single-row call */
int ds4_gpu_k3_moe_raw_batch_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const int32_t *selected, const float *weights, uint32_t n_tok,
                                    uint32_t n_used, uint32_t layer, uint32_t ff, uint32_t latent, float beta, float beta_linear) {
    if (!out || !x || !selected || !weights || n_tok == 0u || n_used == 0u || n_used > 64u || layer >= 128u || !k3m_pool_init(ff, latent)) return 0;
    k3m_batching_init();
    @autoreleasepool {
        const size_t R = (size_t)n_tok * n_used;
        uint32_t *ord = malloc(sizeof(uint32_t) * R), *mexp = malloc(sizeof(uint32_t) * R);
        float *w = malloc(sizeof(float) * R);
        int32_t *slab = malloc(sizeof(int32_t) * R), *mslab = malloc(sizeof(int32_t) * R);
        int ok = ord && mexp && w && slab && mslab;
        uint32_t n_miss = 0;
        /* step M14: in a drafter verify block a fill used by fewer than `admit` of the block's rows is transient (keep value 0):
           it is the first slab recycled, so the pool's hot experts survive the block. CUDA's DS4_K3_VERIFY_ADMIT; auto = 4 rows for
           blocks of 7 or more, 3 for 5-6, 2 below; DS4_K3_METAL_ADMIT=0 off, =N fixed. Only which slabs stay changes. */
        static int admit_env = -2;
        if (admit_env == -2) {
            const char *e = getenv("DS4_K3_METAL_ADMIT");
            admit_env = (!e || !strcmp(e, "auto")) ? -1 : atoi(e);
            fprintf(stderr, "ds4: K3 on Metal: verify-block admission %s\n", admit_env == 0 ? "off" : admit_env < 0 ? "auto (4 rows for blocks of 7+, 3 for 5-6, 2 below)" : "fixed");
        }
        const uint32_t admit = (!g_k3m_verify || admit_env == 0 || g_k3m_evict != 1) ? 0u : admit_env > 0 ? (uint32_t)admit_env
                             : (n_tok >= 7u ? 4u : n_tok >= 5u ? 3u : 2u);
        uint8_t cnt[1024];
        if (admit) {
            memset(cnt, 0, sizeof cnt);
            for (size_t q = 0; q < R; q++) { const int32_t e = selected[q]; if (e >= 0 && e < 1024 && cnt[e] < 255u) cnt[e]++; }
        }
        if (ok) {
            g_k3m.tick++;
            for (uint32_t t = 0; ok && t < n_tok; t++) {
                ok = k3m_order_row(selected + (size_t)t * n_used, weights + (size_t)t * n_used, n_used, ord + (size_t)t * n_used, w + (size_t)t * n_used);
                for (uint32_t j = 0; ok && j < n_used; j++) {
                    const size_t q = (size_t)t * n_used + j;
                    int miss = 0;
                    ok = k3m_lookup(layer, ord[q] >> 6, &slab[q], &miss);
                    if (ok && miss) {
                        mexp[n_miss] = ord[q] >> 6; mslab[n_miss++] = slab[q];
                        if (admit && cnt[ord[q] >> 6] < admit) g_k3m.val_of[slab[q]] = 0.0f;   /* step M14: transient fill */
                    }
                }
            }
        }
        static int mt = -1, mt_check = -1;
        if (mt < 0) { const char *e = getenv("DS4_K3_MOE_MT"); mt = e ? atoi(e) != 0 : 1; const char *c = getenv("DS4_K3_MOE_MT_CHECK"); mt_check = c ? atoi(c) : 0; }
        static int ovl = -1;
        if (ovl < 0) { const char *e = getenv("DS4_K3_OVERLAP"); ovl = e ? atoi(e) != 0 : 1; }
        const int ph = ok && ovl && g_k3m_batching == 1 && mt && mt_check <= 0 && n_tok > 1u && n_used <= 16u && k3m_moe_v2() >= 3 && n_miss > 0u;
        uint8_t emiss[1024];
        static int nw_env = -1, wmode = 0;
        if (nw_env < 0) { const char *e = getenv("DS4_K3_WAVES"), *m = getenv("DS4_K3_WAVE_MODE"); nw_env = e ? atoi(e) : 3; if (nw_env < 1) nw_env = 1; if (nw_env > 8) nw_env = 8;
                          wmode = m ? atoi(m) : 1;
                          if (nw_env > 1) fprintf(stderr, "ds4: K3 on Metal: missed experts of a batch %s %d %s (step %s)\n",
                                                  wmode ? "computed as their reads complete, in about" : "read and computed in", nw_env, wmode ? "chunks" : "waves", wmode ? "M25b" : "M25"); }
        uint32_t W = 1u;
        const int cmode = ph && wmode && nw_env > 1 && n_miss >= 2u * (uint32_t)nw_env;   /* step M25b */
        if (ph) {   /* step M20: the cached experts run while the missed ones are read; M25: then wave by wave */
            W = n_miss >= 2u * (uint32_t)nw_env ? (uint32_t)nw_env : 1u;
            memset(emiss, 0, sizeof emiss);
            for (uint32_t m = 0; m < n_miss; m++) emiss[mexp[m]] = (uint8_t)(cmode ? 1u : 1u + (uint32_t)(((uint64_t)m * W) / n_miss));
            ok = k3m_encode_batch_mt_ph(out, x, ord, w, slab, n_tok, n_used, ff, latent, beta, beta_linear, 1, emiss, cmode ? 0xffffffffu : 0u);
        }
        ds4_gpu_tensor *ref = NULL;
        if (ok && cmode) {
            __block int cok = 1;
            uint32_t *el = malloc(sizeof(uint32_t) * n_miss);
            if (!el) ok = 0;
            const uint32_t *mx = mexp; const uint32_t *ordp = ord; const float *wp = w; const int32_t *slp = slab; const uint8_t *emp = emiss;
            if (ok) ok = k3m_read_misses_cb(layer, mexp, mslab, n_miss, ^(const uint32_t *ms, uint32_t k, int last) {
                if (!cok) return;
                for (uint32_t i = 0; i < k; i++) el[i] = mx[ms[i]];
                g_k3m_mt_elist = el; g_k3m_mt_ne = k;
                cok = k3m_encode_batch_mt_ph(out, x, ordp, wp, slp, n_tok, n_used, ff, latent, beta, beta_linear, last ? 5 : 4, emp, 0u);
                g_k3m_mt_elist = NULL; g_k3m_mt_ne = 0;
            }, (n_miss + (uint32_t)nw_env - 1u) / (uint32_t)nw_env);
            if (ok) ok = cok;
            free(el);
        } else if (ok && ph) {
            uint32_t mb = 0;
            for (uint32_t wv = 0; ok && wv < W; wv++) {
                uint32_t me = mb;
                while (me < n_miss && emiss[mexp[me]] == wv + 1u) me++;
                ok = k3m_read_misses(layer, mexp + mb, mslab + mb, me - mb);
                if (ok) ok = k3m_encode_batch_mt_ph(out, x, ord, w, slab, n_tok, n_used, ff, latent, beta, beta_linear, wv + 1u < W ? 3 : 2, emiss, wv);
                mb = me;
            }
        } else if (ok && !(ok = k3m_read_misses(layer, mexp, mslab, n_miss))) {
        } else if (ok && mt && n_tok > 1u && n_used <= 16u && k3m_moe_v2() >= 3) {
            if (mt_check > 0) {   /* the per-row path into a side tensor first, compared after the batch */
                ref = ds4_gpu_tensor_alloc((uint64_t)n_tok * latent * 4u);
                for (uint32_t t = 0; ok && ref && t < n_tok; t++) {
                    k3m_views vs = {{0}, 0};
                    ds4_gpu_tensor *rt = k3m_view(&vs, ref, (uint64_t)t * latent * 4u, (uint64_t)latent * 4u);
                    ds4_gpu_tensor *xx = k3m_view(&vs, x, (uint64_t)t * latent * 4u, (uint64_t)latent * 4u);
                    ok = rt && xx && k3m_encode_row(rt, xx, slab + (size_t)t * n_used, w + (size_t)t * n_used, n_used, ff, latent, beta, beta_linear);
                    k3m_views_free(&vs);
                }
            }
            if (ok) ok = k3m_encode_batch_mt(out, x, ord, w, slab, n_tok, n_used, ff, latent, beta, beta_linear);
            if (ok && ref) {
                float *a1 = (float *)malloc((size_t)n_tok * latent * 4u), *a2 = (float *)malloc((size_t)n_tok * latent * 4u);
                if (a1 && a2 && ds4_gpu_tensor_read(ref, 0, a1, (uint64_t)n_tok * latent * 4u) && ds4_gpu_tensor_read(out, 0, a2, (uint64_t)n_tok * latent * 4u)) {
                    uint64_t bad = 0; const uint64_t nf = (uint64_t)n_tok * latent;
                    for (uint64_t i = 0; i < nf; i++) if (memcmp(&a1[i], &a2[i], 4) != 0) bad++;
                    static uint64_t calls = 0, floats = 0, diffs = 0;
                    calls++; floats += nf; diffs += bad;
                    if (bad || calls % 92u == 0u) fprintf(stderr, "ds4: K3 MoE multi-token check: %llu calls, %llu floats compared, %llu differ\n",
                                                         (unsigned long long)calls, (unsigned long long)floats, (unsigned long long)diffs);
                    mt_check--;
                }
                free(a1); free(a2);
            }
            if (ref) ds4_gpu_tensor_free(ref);
        } else {
            for (uint32_t t = 0; ok && t < n_tok; t++) {
                k3m_views vs = {{0}, 0};
                ds4_gpu_tensor *o = k3m_view(&vs, out, (uint64_t)t * latent * 4u, (uint64_t)latent * 4u), *xx = k3m_view(&vs, x, (uint64_t)t * latent * 4u, (uint64_t)latent * 4u);
                ok = o && xx && k3m_encode_row(o, xx, slab + (size_t)t * n_used, w + (size_t)t * n_used, n_used, ff, latent, beta, beta_linear);
                k3m_views_free(&vs);
            }
        }
        if (ok && n_tok <= 16u) { k3p_init(); k3p_issue_k(layer + 1u, g_k3p.k); }   /* step M24: the next layer's predicted experts */
        free(ord); free(mexp); free(w); free(slab); free(mslab);
        return ok;
    }
}
int ds4_gpu_k3_moe_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const int32_t *selected, const float *weights, uint32_t n_used,
                          uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t ff, uint32_t latent, float beta, float beta_linear) {
    (void)out; (void)x; (void)selected; (void)weights; (void)n_used; (void)gate_offset; (void)up_offset; (void)down_offset; (void)ff; (void)latent; (void)beta; (void)beta_linear;
    return k3m_unsupported("the GGUF-resident expert path");
}
void ds4_gpu_k3_moe_raw_stats(double *wait_s, uint64_t *bytes) { if (wait_s) *wait_s = g_k3m.wait_s; if (bytes) *bytes = g_k3m.bytes; }
void ds4_gpu_k3_cache_stats(uint64_t *hits_dev, uint64_t *hits_host, uint64_t *misses) {
    if (hits_dev) *hits_dev = g_k3m.hits;
    if (hits_host) *hits_host = 0;
    if (misses) *misses = g_k3m.misses;
}

/* ---- rows helpers of the prompt pass ---- */
typedef struct { uint32_t dst_stride, src_stride, width; } k3m_rows_args;
int ds4_gpu_k3_copy_rows_tensor(ds4_gpu_tensor *dst, uint32_t dst_stride, const ds4_gpu_tensor *src, uint32_t src_stride, uint32_t width, uint32_t n_rows) {
    if (!dst || !src || width == 0u || n_rows == 0u) return 0;
    k3m_rows_args a = { dst_stride, src_stride, width };
    return k3m_run("kernel_k3_copy_rows", ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        K3M_BUF(enc, dst, 1); K3M_BUF(enc, src, 2);
        [enc dispatchThreads:MTLSizeMake(width, n_rows, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }, "K3 copy rows");
}
int ds4_gpu_k3_anchor_push_batch_tensor(ds4_gpu_tensor *anchors, const ds4_gpu_tensor *cur, uint32_t slot, uint32_t n_tok, uint32_t n) {
    if (!anchors || !cur || slot >= 16u) return 0;   /* anchors [tok][16][n]: row slot of every token <- cur[tok] */
    k3m_views vs = {{0}, 0};
    ds4_gpu_tensor *dst = k3m_view(&vs, anchors, (uint64_t)slot * n * 4u, ds4_gpu_tensor_bytes(anchors) - (uint64_t)slot * n * 4u);
    const int ok = dst && ds4_gpu_k3_copy_rows_tensor(dst, 16u * n, cur, n, n, n_tok);
    k3m_views_free(&vs);
    return ok;
}

/* ---- not ported yet, or not needed on unified memory ---- */
/* step M13: K3 spine Q8_0 matmuls of more than 64 rows in fp32. The generic batched kernels round weight tiles (and the
   llama.cpp-style ones the activations) to half; here the weight is dequantized to fp32 in fixed slices of output rows
   (64 M values, exact) and multiplied by MPS's fp32 GEMM. Slices are fixed per matrix, so a pass is deterministic. */
static id<MTLBuffer> g_k3m_f32w = nil;
int ds4_gpu_k3_matmul_q8_0_f32gemm_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset,
                                          uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || in_dim == 0u || out_dim == 0u || n_tok == 0u || (in_dim % 32u) != 0u) return 0;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    k3m_batching_init();
    const uint64_t blocks = in_dim / 32u, wbytes = out_dim * blocks * 34u;
    const uint64_t slice_vals = 64ull << 20, slice_rows = slice_vals / in_dim;
    if (slice_rows == 0u) return 0;
    if (!g_k3m_f32w && !(g_k3m_f32w = [g_device newBufferWithLength:(NSUInteger)(slice_vals * 4u) options:MTLResourceStorageModePrivate])) return 0;
    @autoreleasepool {
        uint64_t inner = 0;
        id<MTLBuffer> wbuf = k3m_model(model_map, model_size, weight_offset, wbytes, &inner);
        id<MTLComputePipelineState> dq = ds4_gpu_get_pipeline("kernel_k3_q8_dequant_f32");
        if (!wbuf || !dq) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        MPSMatrixDescriptor *ad = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n_tok columns:(NSUInteger)in_dim
                                                                       rowBytes:(NSUInteger)in_dim * 4u dataType:MPSDataTypeFloat32];
        MPSMatrix *A = [[MPSMatrix alloc] initWithBuffer:ds4_gpu_tensor_buffer(x) offset:ds4_gpu_tensor_offset(x) descriptor:ad];
        for (uint64_t r0 = 0; r0 < out_dim; r0 += slice_rows) {
            const uint64_t rows = out_dim - r0 < slice_rows ? out_dim - r0 : slice_rows;
            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            struct { uint32_t in_dim, blocks, n_rows; } da = { (uint32_t)in_dim, (uint32_t)blocks, (uint32_t)rows };
            [enc setComputePipelineState:dq];
            [enc setBytes:&da length:sizeof(da) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)(inner + r0 * blocks * 34u) atIndex:1];
            [enc setBuffer:g_k3m_f32w offset:0 atIndex:2];
            [enc dispatchThreads:MTLSizeMake((NSUInteger)in_dim, (NSUInteger)rows, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
            if (g_batch_cb && cb == g_batch_cb) ds4_gpu_close_batch_encoder();   /* MPS encodes its own passes */
            MPSMatrixDescriptor *bd = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)rows columns:(NSUInteger)in_dim
                                                                           rowBytes:(NSUInteger)in_dim * 4u dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *cd = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n_tok columns:(NSUInteger)rows
                                                                           rowBytes:(NSUInteger)out_dim * 4u dataType:MPSDataTypeFloat32];
            MPSMatrix *B = [[MPSMatrix alloc] initWithBuffer:g_k3m_f32w offset:0 descriptor:bd];
            MPSMatrix *C = [[MPSMatrix alloc] initWithBuffer:ds4_gpu_tensor_buffer(out) offset:ds4_gpu_tensor_offset(out) + (NSUInteger)(r0 * 4u) descriptor:cd];
            MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:g_device transposeLeft:NO transposeRight:YES
                                                                               resultRows:(NSUInteger)n_tok resultColumns:(NSUInteger)rows
                                                                          interiorColumns:(NSUInteger)in_dim alpha:1.0 beta:0.0];
            [mm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C];
            if (g_batch_cb && cb == g_batch_cb) g_batch_has_work = YES;
        }
        return ds4_gpu_finish_command_buffer(cb, owned, "K3 q8_0 fp32 GEMM");
    }
}
/* step M5: the rows matmuls of the prompt pass and verify blocks: the single-row call per row on views, so every row is
   bit-identical to the decode step (the q8 rows reread the weight per row; a shared-load kernel can come later) */
static int k3m_rows_loop(int f32, ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset,
                         uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok);
/* step M15: Q8_0 rows in one kernel, each weight block read once per 8 rows (see kernel_k3_mul_mv_q8_0_f32_rows) */
static int k3m_rows_q8_kernel(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim,
                              uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if ((in_dim % 32u) != 0u || n_tok > 0xffffffffull) return -1;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    k3m_batching_init();
    const uint64_t row_bytes = (in_dim / 32u) * 34u;
    @autoreleasepool {
        uint64_t inner = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, out_dim * row_bytes, &inner);
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x), obuf = ds4_gpu_tensor_buffer(out);
        if (!wbuf || !xbuf || !obuf) return 0;
        ds4_gpu_q8_0_matvec_args a = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch dd = ds4_gpu_make_q8_0_mv_dispatch();
        if (out_dim > 65536u) dd.nsg = 8;   /* as the single-row call */
        a.nr0 = dd.nr0;
        /* step M16: R0 weight rows x RT activation rows per threadgroup (DS4_K3_METAL_ROWS_R0 / _RT; R0 2 = the M15 kernel) */
        static int r0e = -1, rte = -1, kind = 1;
        if (r0e < 0) {
            const char *e1 = getenv("DS4_K3_METAL_ROWS_R0"), *e2 = getenv("DS4_K3_METAL_ROWS_RT"), *e3 = getenv("DS4_K3_METAL_ROWS_KIND");
            kind = e3 ? atoi(e3) : 2;
            r0e = e1 ? atoi(e1) : (kind == 2 ? 4 : 8); rte = e2 ? atoi(e2) : (kind == 2 ? 4 : 8);
            if (kind == 2) {   /* step M18 */
                if (!((r0e == 2 || r0e == 4) && (rte == 4 || rte == 8))) { r0e = 2; rte = 4; }
            } else {
                kind = 1;
                if (!((r0e == 4 && rte == 8) || (r0e == 8 && (rte == 4 || rte == 8)) || (r0e == 16 && rte == 4))) { r0e = 2; rte = 8; }
            }
            fprintf(stderr, "ds4: K3 on Metal: q8_0 rows kernel %s (%d weight rows x %d activation rows per threadgroup)\n",
                    kind == 2 ? "M18 (weights converted once per block)" : r0e == 2 ? "M15" : "M16", r0e, rte);
        }
        int R0 = r0e, RT = rte;
        if (out_dim % (uint64_t)R0 != 0u) { R0 = 2; RT = 8; }
        char kname[64];
        if (kind == 2 && RT > 1 && out_dim % (uint64_t)R0 == 0u) snprintf(kname, sizeof(kname), "kernel_k3_mv_q8_rows2_r%d_t%d", R0, RT);
        else if (R0 == 2) snprintf(kname, sizeof(kname), "kernel_k3_mul_mv_q8_0_f32_rows");
        else snprintf(kname, sizeof(kname), "kernel_k3_mv_q8_rows_r%d_t%d", R0, RT);
        id<MTLComputePipelineState> p = ds4_gpu_get_mul_mv_pipeline(kname, dd.nsg);
        if (!p) return -1;
        const uint32_t nr = (uint32_t)n_tok;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:p];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:obuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setBytes:&nr length:sizeof(nr) atIndex:4];
        [enc setThreadgroupMemoryLength:(NSUInteger)R0 * (NSUInteger)RT * 32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)R0 - 1u) / (NSUInteger)R0, ((NSUInteger)n_tok + (NSUInteger)RT - 1u) / (NSUInteger)RT, 1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)dd.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_finish_command_buffer(cb, owned, "K3 q8_0 rows (step M15)");
    }
}
static int k3m_rows(int f32, ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim,
                    uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || n_tok == 0u) return 0;
    static int on = -1, check = 0;
    if (on < 0) {
        const char *e = getenv("DS4_K3_METAL_ROWS"), *c = getenv("DS4_K3_METAL_ROWS_CHECK");
        on = e ? atoi(e) != 0 : 1; check = c ? atoi(c) : 0;
        fprintf(stderr, "ds4: K3 on Metal: q8_0 rows matmuls %s%s\n", on ? "in one kernel, each weight block read once per 8 rows (step M15)" : "as the single-row call per row",
                on && check > 0 ? ", checked against the per-row call" : "");
    }
    if (!f32 && on && n_tok >= 2u) {
        ds4_gpu_tensor *ref = NULL;
        if (check > 0) {   /* the per-row path into a side tensor, compared after both ran */
            ref = ds4_gpu_tensor_alloc(n_tok * out_dim * 4u);
            if (!ref || !k3m_rows_loop(0, ref, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok)) { if (ref) ds4_gpu_tensor_free(ref); ref = NULL; }
        }
        const int r = k3m_rows_q8_kernel(out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok);
        if (r >= 0 && ref) {
            const uint64_t nf = n_tok * out_dim;
            float *a1 = (float *)malloc(nf * 4u), *a2 = (float *)malloc(nf * 4u);
            if (r && a1 && a2 && ds4_gpu_tensor_read(ref, 0, a1, nf * 4u) && ds4_gpu_tensor_read(out, 0, a2, nf * 4u)) {
                uint64_t bad = 0; for (uint64_t i = 0; i < nf; i++) if (memcmp(&a1[i], &a2[i], 4) != 0) bad++;
                static uint64_t calls = 0, floats = 0, diffs = 0;
                calls++; floats += nf; diffs += bad; check--;
                if (bad || (calls % 92u) == 0u || check == 0) fprintf(stderr, "ds4: K3 rows check (step M15): %llu calls, %llu floats compared, %llu differ\n",
                    (unsigned long long)calls, (unsigned long long)floats, (unsigned long long)diffs);
            }
            free(a1); free(a2);
        }
        if (ref) ds4_gpu_tensor_free(ref);
        if (r >= 0) return r;
    }
    return k3m_rows_loop(f32, out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok);
}
static int k3m_rows_loop(int f32, ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset,
                         uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || n_tok == 0u) return 0;
    for (uint64_t t = 0; t < n_tok; t++) {
        k3m_views vs = {{0}, 0};
        ds4_gpu_tensor *o = k3m_view(&vs, out, t * out_dim * 4u, out_dim * 4u), *xx = k3m_view(&vs, x, t * in_dim * 4u, in_dim * 4u);
        const int ok = o && xx && (f32 ? ds4_gpu_matmul_f32_tensor(o, model_map, model_size, weight_offset, in_dim, out_dim, xx, 1)
                                       : ds4_gpu_matmul_quant_tensor(o, model_map, model_size, weight_offset, DS4_METAL_TENSOR_Q8_0, in_dim, out_dim, xx, 1));
        k3m_views_free(&vs);
        if (!ok) {
            fprintf(stderr, "ds4: K3 on Metal: rows matmul failed at row %llu (%s, in %llu, out %llu, x %llu bytes, out %llu bytes, views %s)\n",
                    (unsigned long long)t, f32 ? "f32" : "q8_0", (unsigned long long)in_dim, (unsigned long long)out_dim,
                    (unsigned long long)ds4_gpu_tensor_bytes(x), (unsigned long long)ds4_gpu_tensor_bytes(out), (o && xx) ? "ok" : "failed");
            return 0;
        }
    }
    return 1;
}
int ds4_gpu_k3_matmul_q8_0_rows_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim,
                                       uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return k3m_rows(0, out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok);
}
int ds4_gpu_k3_matmul_f32_rows_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim,
                                      uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return k3m_rows(1, out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok);
}
/* ---- step M11: K3.1 sparse attention (DSA) and the latent attention cache ---- */
int ds4_gpu_k3_lat_store_kvb_tensor(ds4_gpu_tensor *key_cache, ds4_gpu_tensor *value_cache, const ds4_gpu_tensor *kvb, const ds4_gpu_tensor *kv_raw,
                                    const ds4_gpu_tensor *mask, uint32_t pos0, uint32_t n_tok, uint32_t cache_cap, uint32_t heads, uint32_t nope,
                                    uint32_t rot, uint32_t kv_lora, uint32_t vdim) {
    if (!key_cache || !value_cache || !kvb || !kv_raw || n_tok == 0u || pos0 + n_tok > cache_cap) return 0;
    if (mask && ds4_gpu_tensor_bytes(mask) < (uint64_t)pos0 + n_tok) return 0;
    if (ds4_gpu_tensor_bytes(kvb) < (uint64_t)n_tok * heads * (nope + vdim) * 4u || ds4_gpu_tensor_bytes(kv_raw) < (uint64_t)n_tok * (kv_lora + rot) * 4u) return 0;
    struct { uint32_t pos0, heads, nope, rot, kv_lora, vdim, use_mask; } a = { pos0, heads, nope, rot, kv_lora, vdim, mask ? 1u : 0u };
    const uint32_t n = heads * (nope + rot) + heads * vdim;
    const ds4_gpu_tensor *mk = mask ? mask : kvb;
    return k3m_run("kernel_k3_lat_store_kvb", ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        K3M_BUF(enc, key_cache, 1); K3M_BUF(enc, value_cache, 2); K3M_BUF(enc, kvb, 3); K3M_BUF(enc, kv_raw, 4); K3M_BUF(enc, mk, 5);
        [enc dispatchThreads:MTLSizeMake(n, n_tok, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }, "K3 latent K/V store");
}
/* the indexer's f32 weights become a model view of their own, so the plain f32 matmul serves them (as the CUDA device range does) */
int ds4_gpu_k3_register_aux_weights(const void *host, uint64_t bytes) {
    if (!host || bytes == 0u) return 0;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    for (uint32_t i = 0; i < g_model_view_count; i++) if (g_model_views[i].model_map == host && g_model_views[i].model_size == bytes) return 1;
    if (g_model_view_count >= DS4_METAL_MAX_MODEL_VIEWS) return 0;
    id<MTLBuffer> buf = [g_device newBufferWithBytes:host length:(NSUInteger)bytes options:MTLResourceStorageModeShared];
    if (!buf) { fprintf(stderr, "ds4: K3 on Metal: no buffer for the DSA weights (%.1f MB)\n", (double)bytes / 1e6); return 0; }
    ds4_gpu_model_view *v = &g_model_views[g_model_view_count++];
    v->buffer = buf; v->model_map = host; v->model_size = bytes; v->model_offset = 0; v->bytes = bytes;
    fprintf(stderr, "ds4: K3 on Metal: DSA indexer weights in unified memory (%.1f MB)\n", (double)bytes / 1e6);
    return 1;
}
/* every row the single-row call, so verify rows equal plain steps and a prompt pass is deterministic */
int ds4_gpu_k3_dsa_matmul_tensor(ds4_gpu_tensor *out, const void *aux, uint64_t aux_size, uint64_t offset, uint32_t in_dim, uint32_t out_dim,
                                 const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !x || !aux || n_tok == 0u) return 0;
    if (n_tok == 1u) return ds4_gpu_matmul_f32_tensor(out, aux, aux_size, offset, in_dim, out_dim, x, 1);
    return k3m_rows(1, out, aux, aux_size, offset, in_dim, out_dim, x, n_tok);
}
int ds4_gpu_k3_dsa_ln_store_tensor(ds4_gpu_tensor *key_cache, const ds4_gpu_tensor *raw, const void *aux, uint64_t aux_size, uint64_t w_off,
                                   uint64_t b_off, uint32_t pos0, uint32_t n_tok, uint32_t cap, float eps) {
    if (!key_cache || !raw || !aux || n_tok == 0u || pos0 + n_tok > cap || w_off + 512u > aux_size || b_off + 512u > aux_size ||
        ds4_gpu_tensor_bytes(raw) < (uint64_t)n_tok * 128u * 4u || ds4_gpu_tensor_bytes(key_cache) < (uint64_t)cap * 128u * 4u) return 0;
    @autoreleasepool {
        uint64_t iw = 0, ib = 0;
        id<MTLBuffer> wb = k3m_model(aux, aux_size, w_off, 512u, &iw), bb = k3m_model(aux, aux_size, b_off, 512u, &ib);
        if (!wb || !bb) return 0;
        struct { uint32_t pos0; float eps; } a = { pos0, eps };
        return k3m_run("kernel_k3_dsa_ln_store", ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBytes:&a length:sizeof(a) atIndex:0];
            K3M_BUF(enc, key_cache, 1); K3M_BUF(enc, raw, 2);
            [enc setBuffer:wb offset:(NSUInteger)iw atIndex:3]; [enc setBuffer:bb offset:(NSUInteger)ib atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(n_tok, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }, "K3 DSA key store");
    }
}
int ds4_gpu_k3_dsa_scores_tensor(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *key_cache,
                                 uint32_t n_rows, uint32_t n_tokens, uint32_t pos0, uint32_t n_head, float scale) {
    if (!scores || !q || !weights || !key_cache || n_rows == 0u || n_tokens == 0u || n_head == 0u || n_head > 64u || !isfinite(scale) || scale <= 0.0f ||
        ds4_gpu_tensor_bytes(q) < (uint64_t)n_tokens * n_head * 128u * 4u || ds4_gpu_tensor_bytes(weights) < (uint64_t)n_tokens * n_head * 4u ||
        ds4_gpu_tensor_bytes(key_cache) < (uint64_t)n_rows * 128u * 4u || ds4_gpu_tensor_bytes(scores) < (uint64_t)n_tokens * n_rows * 4u) return 0;
    struct { uint32_t n_rows, n_tokens, pos0, n_head; float scale; } a = { n_rows, n_tokens, pos0, n_head, scale };
    return k3m_run("kernel_k3_dsa_scores", ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        K3M_BUF(enc, scores, 1); K3M_BUF(enc, q, 2); K3M_BUF(enc, weights, 3); K3M_BUF(enc, key_cache, 4);
        [enc setThreadgroupMemoryLength:(NSUInteger)((n_head * 128u + n_head + 3u) & ~3u) * 4u atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((n_rows + 255u) / 256u, n_tokens, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }, "K3 DSA scores");
}
int ds4_gpu_k3_dsa_mask_tensor(ds4_gpu_tensor *mask, const ds4_gpu_tensor *selected, uint32_t stride, uint32_t n_tokens, uint32_t pos0, uint32_t top_k) {
    if (!mask || !selected || n_tokens == 0u || top_k == 0u || pos0 + n_tokens > stride ||
        ds4_gpu_tensor_bytes(mask) < (uint64_t)n_tokens * stride || ds4_gpu_tensor_bytes(selected) < (uint64_t)n_tokens * top_k * 4u) return 0;
    struct { uint32_t stride, pos0, top_k; } a = { stride, pos0, top_k };
    return k3m_run("kernel_k3_dsa_mask", ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        K3M_BUF(enc, mask, 1); K3M_BUF(enc, selected, 2);
        [enc dispatchThreadgroups:MTLSizeMake(n_tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }, "K3 DSA mask");
}
int ds4_gpu_k3_route_hist(uint32_t *counts, const ds4_gpu_tensor *logits, const float *bias, uint32_t n_tok, uint32_t n_expert, uint32_t n_used) {
    (void)counts; (void)logits; (void)bias; (void)n_tok; (void)n_expert; (void)n_used; return 0;
}
int ds4_gpu_k3_cache_grow(void) { return 1; }
uint64_t ds4_gpu_k3_cache_lend(void) { return 0; }
/* step M5: what the prompt pass may still allocate: the device's recommended working set less what is allocated
   (the mapped spine and the expert slabs count); DS4_K3_METAL_FREE_GB overrides */
uint64_t ds4_gpu_metal_free_bytes(void) {
    const char *o = getenv("DS4_K3_METAL_FREE_GB");
    if (o && atof(o) > 0) return (uint64_t)(atof(o) * 1e9);
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint64_t cap = (uint64_t)g_device.recommendedMaxWorkingSetSize, used = (uint64_t)g_device.currentAllocatedSize;
    return cap > used ? cap - used : 0;
}
uint32_t ds4_gpu_k3_attn_max_pos(void) { return 32768u; }
uint64_t ds4_gpu_k3_attn_scratch_need(uint32_t n_tok, uint32_t n_cols, uint32_t heads) { (void)n_tok; (void)n_cols; (void)heads; return 0; }
uint64_t ds4_gpu_k3_pass_scratch_need(uint32_t n_tok, uint32_t n_used, uint32_t ff, uint32_t latent) { (void)n_tok; (void)n_used; (void)ff; (void)latent; return 0; }
int ds4_gpu_k3_skip_mode(void) { return 0; }
void ds4_gpu_k3_skip_stats(uint64_t *skipped, double *mass, uint64_t *obs_n, uint64_t obs[6]) {
    if (skipped) *skipped = 0; if (mass) *mass = 0; if (obs_n) *obs_n = 0; if (obs) for (int i = 0; i < 6; i++) obs[i] = 0;
}
uint64_t ds4_gpu_k3_cache_copydowns(void) { return 0; }
void ds4_gpu_k3_prefetch_hint(uint32_t layer, const int32_t *experts, uint32_t n) {
    k3p_init();
    if (g_k3p.k == 0 || !experts || layer >= 128u) return;
    g_k3p.n = n > 32u ? 32u : n;
    memcpy(g_k3p.hint, experts, sizeof(int32_t) * g_k3p.n);
    g_k3p.layer = layer; g_k3p.have = 1;
}
void ds4_gpu_k3_prefetch_stats(uint64_t *issued, uint64_t *adopted) { if (issued) *issued = g_k3p.issued; if (adopted) *adopted = g_k3p.adopted; }
void ds4_gpu_k3_lookahead_report(void) {   /* step M24 stats */
    if (g_k3p.issued) fprintf(stderr, "ds4: K3 lookahead prefetch: issued %llu, used %llu (%.1f %%), evicted unused %llu, waited on in flight %llu\n",
                              (unsigned long long)g_k3p.issued, (unsigned long long)g_k3p.used, 100.0 * (double)g_k3p.used / (double)g_k3p.issued,
                              (unsigned long long)g_k3p.wasted, (unsigned long long)g_k3p.adopted);
}
void ds4_gpu_k3_set_verify(int on) { g_k3m_verify = on; }   /* step M14 */
int ds4_gpu_k3_mla_cache_elem_bytes(void) { return 4; }
void ds4_gpu_k3_queue_stats(double *wait_s, uint64_t *n_waits, uint64_t *pf_hit) { if (wait_s) *wait_s = 0; if (n_waits) *n_waits = 0; if (pf_hit) *pf_hit = 0; }
void ds4_gpu_k3_set_spine_gemm_max(uint32_t n) { (void)n; }
void ds4_gpu_k3_argodrive_stats(uint64_t *chunks3, double *stream_gbps3) { if (chunks3) for (int i = 0; i < 3; i++) chunks3[i] = 0; if (stream_gbps3) for (int i = 0; i < 3; i++) stream_gbps3[i] = 0; }
void ds4_gpu_k3_source_stats(uint64_t *bytes3, uint64_t *fallback_reads, uint64_t *fallback_bytes, int *tier_c_configured) {
    if (bytes3) for (int i = 0; i < 3; i++) bytes3[i] = 0; if (fallback_reads) *fallback_reads = 0; if (fallback_bytes) *fallback_bytes = 0;
    if (tier_c_configured) *tier_c_configured = 0;
}
int ds4_gpu_k3_draft_load(const char *dir, uint32_t cache_cap) { (void)dir; (void)cache_cap; return k3m_unsupported("the DSpark drafter"); }
int ds4_gpu_k3_draft_capture(int ci, const ds4_gpu_tensor *cur, uint32_t T) { (void)ci; (void)cur; (void)T; return 0; }
int ds4_gpu_k3_draft_commit(uint32_t pos0, uint32_t T) { (void)pos0; (void)T; return 0; }
int ds4_gpu_k3_draft_truncate(uint32_t len) { (void)len; return 0; }
uint32_t ds4_gpu_k3_draft_len(void) { return 0; }
int ds4_gpu_k3_draft_backbone(const ds4_gpu_tensor *embed7, uint32_t cache_len, ds4_gpu_tensor *hidden7_out) { (void)embed7; (void)cache_len; (void)hidden7_out; return 0; }
int ds4_gpu_k3_draft_finish(ds4_gpu_tensor *logits7, int anchor, int ids[7], float conf[7]) { (void)logits7; (void)anchor; (void)ids; (void)conf; return 0; }
double ds4_gpu_k3_draft_gpu_bytes(void) { return 0.0; }

/* ---- the CUDA engine's steps 106 / 107 entry points ---- */
void ds4_gpu_k3_set_f32_act(int on) { (void)on; }   /* the Metal K3 path computes with fp32 activations throughout */
int ds4_gpu_k3_memset_tensor(ds4_gpu_tensor *t, uint64_t offset, int value, uint64_t bytes) {
    if (!t || offset > ds4_gpu_tensor_bytes(t) || bytes > ds4_gpu_tensor_bytes(t) - offset) return 0;
    if (bytes == 0u) return 1;
    if (!ds4_gpu_synchronize()) return 0;   /* the host writes shared memory: nothing encoded may still use it */
    memset((uint8_t *)[ds4_gpu_tensor_buffer(t) contents] + ds4_gpu_tensor_offset(t) + offset, value, (size_t)bytes);
    return 1;
}
int ds4_gpu_k3_kvb_chain_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset,
                                uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x, uint32_t n_tok) {
    (void)out; (void)model_map; (void)model_size; (void)weight_offset; (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return k3m_unsupported("the kv_b chain kernel (CUDA step 107)");
}
int ds4_gpu_k3_lat_replay_sel_tensor(ds4_gpu_tensor *key_cache, ds4_gpu_tensor *value_cache, ds4_gpu_tensor *kvb,
        const ds4_gpu_tensor *norm_rows, const ds4_gpu_tensor *raw_rows, const ds4_gpu_tensor *cls, const ds4_gpu_tensor *mask,
        uint32_t q0, uint32_t q1, uint32_t max_rows, uint32_t cache_cap, const void *model_map, uint64_t model_size,
        uint64_t kvb_offset, uint32_t heads, uint32_t nope, uint32_t rot, uint32_t kv_lora, uint32_t vdim) {
    (void)key_cache; (void)value_cache; (void)kvb; (void)norm_rows; (void)raw_rows; (void)cls; (void)mask; (void)q0; (void)q1; (void)max_rows;
    (void)cache_cap; (void)model_map; (void)model_size; (void)kvb_offset; (void)heads; (void)nope; (void)rot; (void)kv_lora; (void)vdim;
    return k3m_unsupported("the selected-position latent replay (CUDA step 107)");
}
