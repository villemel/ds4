/* ds4_k3_draft.cuh — the Kimi K3 DSpark drafter (Inferact/Kimi-K3-DSpark) natively on the GPU (step 46).
 *
 * Included at the end of ds4_cuda.cu. BF16 weights are read straight from the checkpoint's safetensors file (the K3
 * embedding and head are the target's own, supplied by ds4.c); the five decoder layers run with cuBLAS BF16 GEMMs,
 * K3's MLA store/attend kernels in the expanded (per-head key/value) form, and small kernels for the norms, the YaRN
 * rotation (interleaved pairs), SiLU, the Markov bigram correction and the confidence head. The arithmetic follows
 * deltafin's provider_dspark.cpp: fp32 norms and softmax, BF16 rounding at every linear output, no mask among the
 * seven query rows (anchor + six mask tokens), proposals for the seven positions after the anchor.
 *
 * Context: rows captured by ds4.c (the residual stream after decoder layers 2, 23, 47, 71, 89) are staged as
 * [T][5][n] bf16 (ds4_gpu_k3_draft_capture), and the committed ones become per-layer key/value rows
 * (ds4_gpu_k3_draft_commit). Only target-verified tokens ever enter the context.
 */
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace {

constexpr uint32_t K3D_H = 7168u, K3D_I = 14336u, K3D_NH = 64u, K3D_QR = 1536u, K3D_KR = 512u, K3D_NOPE = 128u, K3D_ROPE = 64u;
constexpr uint32_t K3D_VD = 128u, K3D_V = 163840u, K3D_MR = 256u, K3D_MASK = 163837u, K3D_NL = 5u, K3D_NT = 5u, K3D_Q = 7u;
constexpr uint32_t K3D_QK = K3D_NOPE + K3D_ROPE, K3D_KVW = K3D_KR + K3D_ROPE;   /* 192, 576 */
constexpr float K3D_EPS = 1e-5f;
constexpr double K3D_THETA = 50000.0, K3D_FACTOR = 32.0, K3D_ORIG = 32768.0, K3D_BFAST = 32.0, K3D_BSLOW = 1.0;

struct k3d_layer { __nv_bfloat16 *in_norm, *post_norm, *q_a, *q_a_norm, *q_b, *kv_a, *kv_a_norm, *kv_b, *o, *gate, *up, *down; };
struct k3d_model {
    bool ready = false;
    uint32_t cap = 0, len = 0, stage_cap = 0;
    __nv_bfloat16 *ctx_proj = nullptr, *ctx_norm = nullptr, *final_norm = nullptr, *mk_w1 = nullptr, *mk_w2 = nullptr, *conf_w = nullptr;
    float conf_b = 0.0f;
    k3d_layer L[K3D_NL];
    __nv_bfloat16 *fused_kv_a = nullptr;
    float *key_cache[K3D_NL], *value_cache[K3D_NL];
    __nv_bfloat16 *stage = nullptr;
    float *inv_freq = nullptr;
    /* proposal scratch (7 rows) */
    __nv_bfloat16 *hid = nullptr, *acc = nullptr, *normed = nullptr, *ql = nullptr, *latent = nullptr, *heads16 = nullptr, *acc2 = nullptr, *post = nullptr, *mid = nullptr, *resid = nullptr, *scored = nullptr, *embs = nullptr;
    float *qa = nullptr, *q = nullptr, *proj = nullptr, *kvb = nullptr, *k_nope = nullptr, *value = nullptr, *heads = nullptr, *attn = nullptr, *g = nullptr, *u = nullptr, *mlp = nullptr, *conf = nullptr;
    int *idx = nullptr;
    /* context-append scratch (chunks of K3D_CHUNK rows) */
    float *c_ctx = nullptr, *c_stacked = nullptr, *c_kvb = nullptr, *c_knope = nullptr, *c_value = nullptr, *c_kvraw = nullptr;
    __nv_bfloat16 *c_ctx16 = nullptr, *c_latent = nullptr;
    double gpu_bytes = 0.0;
};
constexpr uint32_t K3D_CHUNK = 256u;
static k3d_model g_k3d;

/* ---- kernels ------------------------------------------------------------------------------------------------- */
__device__ __forceinline__ float k3d_r(float x) { return __bfloat162float(__float2bfloat16(x)); }   /* round through bf16 */

__global__ static void k3d_rows_to_bf16_kernel(__nv_bfloat16 *stage, const float *cur, uint32_t T, uint32_t n, uint32_t ci) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * n) return;
    const uint32_t t = (uint32_t)(i / n), c = (uint32_t)(i % n);
    stage[((size_t)t * K3D_NT + ci) * n + c] = __float2bfloat16(cur[i]);
}
__global__ static void k3d_f32_to_bf16_kernel(__nv_bfloat16 *out, const float *in, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}
__global__ static void k3d_bf16_to_f32_kernel(float *out, const __nv_bfloat16 *in, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}
/* out = bf16(a + bf16(b)); b is a GEMM's fp32 output */
__global__ static void k3d_add_kernel(__nv_bfloat16 *out, const __nv_bfloat16 *a, const float *b, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(a[i]) + k3d_r(b[i]));
}
__global__ static void k3d_add16_kernel(__nv_bfloat16 *out, const __nv_bfloat16 *a, const __nv_bfloat16 *b, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}
/* RMS norm per row: the input row is fp32 (rounded through bf16 first) or bf16; one block per row.
   out[r][i] = bf16(bf16(x_i * rsqrt(mean(x^2) + eps)) * w_i) */
__global__ static void k3d_rms_kernel(__nv_bfloat16 *out, const float *in32, const __nv_bfloat16 *in16, uint32_t in_stride,
                                      const __nv_bfloat16 *w, uint32_t n, float eps) {
    __shared__ float red[32];
    const uint32_t r = blockIdx.x;
    const float *x32 = in32 ? in32 + (size_t)r * in_stride : nullptr;
    const __nv_bfloat16 *x16 = in16 ? in16 + (size_t)r * in_stride : nullptr;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) { const float v = x32 ? k3d_r(x32[i]) : __bfloat162float(x16[i]); ss += v * v; }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31u) == 0u) red[threadIdx.x >> 5u] = ss;
    __syncthreads();
    if (threadIdx.x == 0u) { float s = 0.0f; for (uint32_t k = 0; k < (blockDim.x >> 5u); k++) s += red[k]; red[0] = s; }
    __syncthreads();
    const float inv = rsqrtf(red[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x32 ? k3d_r(x32[i]) : __bfloat162float(x16[i]);
        out[(size_t)r * n + i] = __float2bfloat16(k3d_r(v * inv) * __bfloat162float(w[i]));
    }
}
/* YaRN rotation of interleaved pairs on `rope` dims at `off` within each head of each row, in place on fp32 (values are
   rounded through bf16 first, the result is written as an fp32 holding a bf16 value scaled by qscale). One block per
   (row, head), 32 threads = pairs. */
__global__ static void k3d_rope_kernel(float *x, uint32_t row_stride, uint32_t n_heads, uint32_t head_stride, uint32_t off,
                                       const float *inv, uint32_t pos0, float qscale) {
    const uint32_t t = blockIdx.x / n_heads, h = blockIdx.x % n_heads, j = threadIdx.x;
    if (j >= K3D_ROPE / 2u) return;
    float *p = x + (size_t)t * row_stride + (size_t)h * head_stride + off + 2u * j;
    const float phase = (float)(pos0 + t) * inv[j];
    const float c = cosf(phase), s = sinf(phase);
    const float e = k3d_r(p[0]), o = k3d_r(p[1]);
    p[0] = k3d_r(e * c - o * s) * qscale;
    p[1] = k3d_r(o * c + e * s) * qscale;
}
/* the nope part of the query: rounded through bf16 and scaled (the rope kernel does its own part) */
__global__ static void k3d_q_nope_kernel(float *q, uint32_t rows, float qscale) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)rows * K3D_NH * K3D_QK;
    if (i >= total) return;
    const uint32_t d = (uint32_t)(i % K3D_QK);
    if (d < K3D_NOPE) q[i] = k3d_r(q[i]) * qscale;
}
__global__ static void k3d_silu_mul_kernel(__nv_bfloat16 *out, const float *gate, const float *up, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = k3d_r(gate[i]), u = k3d_r(up[i]);
    const float s = k3d_r(g / (1.0f + __expf(-g)));
    out[i] = __float2bfloat16(s * u);
}
/* logits[v] += bf16(sum_r w2[v][r] * e[r]) */
__global__ static void k3d_markov_bias_kernel(float *logits, const __nv_bfloat16 *w2, const __nv_bfloat16 *e, uint32_t vocab, uint32_t rank) {
    const uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= vocab) return;
    const __nv_bfloat16 *row = w2 + (size_t)v * rank;
    float s = 0.0f;
    for (uint32_t r = 0; r < rank; r++) s += __bfloat162float(row[r]) * __bfloat162float(e[r]);
    logits[v] += k3d_r(s);
}
/* argmax of one row (lowest index on ties): one block of 1024 threads */
__global__ static void k3d_argmax_kernel(int *out, const float *row, uint32_t n) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    float best = -3.4e38f; int bidx = 0x7fffffff;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) { const float v = row[i]; if (v > best || (v == best && (int)i < bidx)) { best = v; bidx = (int)i; } }
    bv[threadIdx.x] = best; bi[threadIdx.x] = bidx;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2u; s > 0u; s >>= 1) {
        if (threadIdx.x < s) {
            const float v2 = bv[threadIdx.x + s]; const int i2 = bi[threadIdx.x + s];
            if (v2 > bv[threadIdx.x] || (v2 == bv[threadIdx.x] && i2 < bi[threadIdx.x])) { bv[threadIdx.x] = v2; bi[threadIdx.x] = i2; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) *out = bi[0];
}
/* confidence logit per row: dot(scored[r], w[0..H)) + dot(embs[r], w[H..H+MR)) + b */
__global__ static void k3d_conf_kernel(float *out, const __nv_bfloat16 *scored, const __nv_bfloat16 *embs, const __nv_bfloat16 *w, float b) {
    __shared__ float red[32];
    const uint32_t r = blockIdx.x;
    float s = 0.0f;
    for (uint32_t i = threadIdx.x; i < K3D_H; i += blockDim.x) s += __bfloat162float(scored[(size_t)r * K3D_H + i]) * __bfloat162float(w[i]);
    for (uint32_t i = threadIdx.x; i < K3D_MR; i += blockDim.x) s += __bfloat162float(embs[(size_t)r * K3D_MR + i]) * __bfloat162float(w[K3D_H + i]);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31u) == 0u) red[threadIdx.x >> 5u] = s;
    __syncthreads();
    if (threadIdx.x == 0u) { float t = 0.0f; for (uint32_t k = 0; k < (blockDim.x >> 5u); k++) t += red[k]; out[r] = t + b; }
}

/* ---- helpers ------------------------------------------------------------------------------------------------- */
static inline unsigned k3d_grid(size_t n) { return (unsigned)((n + 255u) / 256u); }
static bool k3d_gemm(const __nv_bfloat16 *W, const __nv_bfloat16 *X, float *C, uint32_t out_f, uint32_t rows, uint32_t in_f) {
    cublasHandle_t h = cuda_cublas_for_tier(0);
    cublasSetStream(h, cuda_decode_stream());
    return k3_gemm_bf16(h, W, X, C, (int)out_f, (int)rows, (int)in_f);
}
static void k3d_rms(__nv_bfloat16 *out, const float *in32, const __nv_bfloat16 *in16, uint32_t in_stride, const __nv_bfloat16 *w, uint32_t rows, uint32_t n) {
    k3d_rms_kernel<<<rows, 256, 0, cuda_decode_stream()>>>(out, in32, in16, in_stride, w, n, K3D_EPS);
}
static double k3d_yarn_mscale(double factor, double mult) { return factor <= 1.0 ? 1.0 : 0.1 * mult * log(factor) + 1.0; }

/* safetensors: header scan for one tensor's byte range */
struct k3d_st { int fd = -1; uint64_t data_start = 0; std::string header; };
static bool k3d_st_open(const char *path, k3d_st &st) {
    st.fd = open(path, O_RDONLY | O_CLOEXEC);
    if (st.fd < 0) { fprintf(stderr, "ds4: K3 drafter: cannot open %s\n", path); return false; }
    uint64_t hl = 0;
    if (pread(st.fd, &hl, 8, 0) != 8 || hl < 2 || hl > (64u << 20)) { fprintf(stderr, "ds4: K3 drafter: bad safetensors header\n"); return false; }
    st.header.resize(hl);
    if (pread(st.fd, &st.header[0], hl, 8) != (ssize_t)hl) return false;
    st.data_start = 8 + hl;
    return true;
}
static bool k3d_st_find(const k3d_st &st, const char *name, uint64_t &off, uint64_t &len) {
    const std::string key = std::string("\"") + name + "\":";
    const size_t p = st.header.find(key);
    if (p == std::string::npos) { fprintf(stderr, "ds4: K3 drafter: tensor %s missing\n", name); return false; }
    const size_t q = st.header.find("\"data_offsets\"", p);
    const size_t e = st.header.find("}", p);
    if (q == std::string::npos || e == std::string::npos || q > e) return false;
    unsigned long long a = 0, b = 0;
    if (sscanf(st.header.c_str() + q, "\"data_offsets\":[%llu,%llu]", &a, &b) != 2) return false;
    off = a; len = b - a;
    return true;
}
static __nv_bfloat16 *k3d_load(const k3d_st &st, const char *name, uint64_t expect_elems, std::vector<uint8_t> &buf) {
    uint64_t off = 0, len = 0;
    if (!k3d_st_find(st, name, off, len)) return nullptr;
    if (len != expect_elems * 2u) { fprintf(stderr, "ds4: K3 drafter: tensor %s has %llu bytes, expected %llu\n", name, (unsigned long long)len, (unsigned long long)(expect_elems * 2u)); return nullptr; }
    __nv_bfloat16 *d = nullptr;
    if (cudaMalloc(&d, len) != cudaSuccess) { cudaGetLastError(); fprintf(stderr, "ds4: K3 drafter: out of device memory for %s\n", name); return nullptr; }
    const size_t chunk = buf.size();
    for (uint64_t done = 0; done < len; done += chunk) {
        const size_t n = (size_t)(len - done < chunk ? len - done : chunk);
        if (pread(st.fd, buf.data(), n, (off_t)(st.data_start + off + done)) != (ssize_t)n) { fprintf(stderr, "ds4: K3 drafter: read of %s failed\n", name); cudaFree(d); return nullptr; }
        if (cudaMemcpy((uint8_t *)d + done, buf.data(), n, cudaMemcpyHostToDevice) != cudaSuccess) { cudaGetLastError(); cudaFree(d); return nullptr; }
        posix_fadvise(st.fd, (off_t)(st.data_start + off + done), (off_t)n, POSIX_FADV_DONTNEED);
    }
    g_k3d.gpu_bytes += (double)len;
    return d;
}
static float *k3d_alloc_f32(size_t n) { float *p = nullptr; if (cudaMalloc(&p, n * sizeof(float)) != cudaSuccess) { cudaGetLastError(); return nullptr; } g_k3d.gpu_bytes += (double)n * 4; return p; }
static __nv_bfloat16 *k3d_alloc_bf16(size_t n) { __nv_bfloat16 *p = nullptr; if (cudaMalloc(&p, n * 2) != cudaSuccess) { cudaGetLastError(); return nullptr; } g_k3d.gpu_bytes += (double)n * 2; return p; }

}  // namespace

/* ---- API ----------------------------------------------------------------------------------------------------- */
extern "C" int ds4_gpu_k3_draft_load(const char *dir, uint32_t cache_cap) {
    k3d_model &m = g_k3d;
    if (m.ready) return 1;
    const double t0 = k3_now_sec();
    std::string path = std::string(dir) + "/model.safetensors";
    k3d_st st;
    if (!k3d_st_open(path.c_str(), st)) return 0;
    std::vector<uint8_t> buf(64u << 20);
    bool ok = true;
    m.ctx_proj = k3d_load(st, "context_proj.weight", (uint64_t)K3D_H * K3D_NT * K3D_H, buf); ok = ok && m.ctx_proj;
    m.ctx_norm = k3d_load(st, "context_norm.weight", K3D_H, buf); ok = ok && m.ctx_norm;
    m.final_norm = k3d_load(st, "final_norm.weight", K3D_H, buf); ok = ok && m.final_norm;
    m.mk_w1 = k3d_load(st, "markov_head.markov_w1.weight", (uint64_t)K3D_V * K3D_MR, buf); ok = ok && m.mk_w1;
    m.mk_w2 = k3d_load(st, "markov_head.markov_w2.weight", (uint64_t)K3D_V * K3D_MR, buf); ok = ok && m.mk_w2;
    m.conf_w = k3d_load(st, "confidence_head.proj.weight", K3D_H + K3D_MR, buf); ok = ok && m.conf_w;
    {
        uint64_t off = 0, len = 0; uint16_t bits = 0;
        if (k3d_st_find(st, "confidence_head.proj.bias", off, len) && len == 2 && pread(st.fd, &bits, 2, (off_t)(st.data_start + off)) == 2) {
            const uint32_t u = (uint32_t)bits << 16; memcpy(&m.conf_b, &u, 4);
        } else ok = false;
    }
    for (uint32_t l = 0; l < K3D_NL && ok; l++) {
        char nm[128]; k3d_layer &L = m.L[l];
#define K3D_T(field, suffix, elems) snprintf(nm, sizeof(nm), "layers.%u.%s", l, suffix); L.field = k3d_load(st, nm, (elems), buf); ok = ok && L.field;
        K3D_T(in_norm, "input_layernorm.weight", K3D_H)
        K3D_T(post_norm, "post_attention_layernorm.weight", K3D_H)
        K3D_T(q_a, "self_attn.q_a_proj.weight", (uint64_t)K3D_QR * K3D_H)
        K3D_T(q_a_norm, "self_attn.q_a_layernorm.weight", K3D_QR)
        K3D_T(q_b, "self_attn.q_b_proj.weight", (uint64_t)K3D_NH * K3D_QK * K3D_QR)
        K3D_T(kv_a, "self_attn.kv_a_proj_with_mqa.weight", (uint64_t)K3D_KVW * K3D_H)
        K3D_T(kv_a_norm, "self_attn.kv_a_layernorm.weight", K3D_KR)
        K3D_T(kv_b, "self_attn.kv_b_proj.weight", (uint64_t)K3D_NH * (K3D_NOPE + K3D_VD) * K3D_KR)
        K3D_T(o, "self_attn.o_proj.weight", (uint64_t)K3D_H * K3D_NH * K3D_VD)
        K3D_T(gate, "mlp.gate_proj.weight", (uint64_t)K3D_I * K3D_H)
        K3D_T(up, "mlp.up_proj.weight", (uint64_t)K3D_I * K3D_H)
        K3D_T(down, "mlp.down_proj.weight", (uint64_t)K3D_H * K3D_I)
#undef K3D_T
    }
    close(st.fd);
    if (!ok) { fprintf(stderr, "ds4: K3 drafter: checkpoint load failed (%s)\n", path.c_str()); return 0; }
    /* the fused five-layer kv_a projection for the context rows */
    m.fused_kv_a = k3d_alloc_bf16((size_t)K3D_NL * K3D_KVW * K3D_H);
    ok = m.fused_kv_a != nullptr;
    for (uint32_t l = 0; l < K3D_NL && ok; l++)
        ok = cudaMemcpy(m.fused_kv_a + (size_t)l * K3D_KVW * K3D_H, m.L[l].kv_a, (size_t)K3D_KVW * K3D_H * 2, cudaMemcpyDeviceToDevice) == cudaSuccess;
    /* YaRN inverse frequencies (provider_dspark.cpp) */
    {
        float inv[K3D_ROPE / 2];
        const double half = K3D_ROPE / 2;
        auto corr = [](double rot) { return (double)K3D_ROPE * log(K3D_ORIG / (rot * 2.0 * M_PI)) / (2.0 * log(K3D_THETA)); };
        const double low = fmax(floor(corr(K3D_BFAST)), 0.0);
        double high = fmin(ceil(corr(K3D_BSLOW)), (double)K3D_ROPE - 1.0);
        if (low == high) high += 0.001;
        for (uint32_t j = 0; j < (uint32_t)half; j++) {
            const double exponent = (double)j * (2.0 / (double)K3D_ROPE);
            const double extrap = 1.0 / exp(exponent * log(K3D_THETA)), interp = extrap / K3D_FACTOR;
            double ramp = ((double)j - low) / (high - low); ramp = ramp < 0.0 ? 0.0 : ramp > 1.0 ? 1.0 : ramp;
            const double ef = 1.0 - ramp;
            inv[j] = (float)(interp * (1.0 - ef) + extrap * ef);
        }
        m.inv_freq = k3d_alloc_f32(K3D_ROPE / 2);
        ok = ok && m.inv_freq && cudaMemcpy(m.inv_freq, inv, sizeof(inv), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    /* caches and scratch */
    m.cap = cache_cap + K3D_Q + 8u;
    for (uint32_t l = 0; l < K3D_NL && ok; l++) {
        m.key_cache[l] = k3d_alloc_f32((size_t)m.cap * K3D_NH * K3D_QK);
        m.value_cache[l] = k3d_alloc_f32((size_t)m.cap * K3D_NH * K3D_VD);
        ok = m.key_cache[l] && m.value_cache[l];
    }
    m.stage_cap = cache_cap;
    m.stage = k3d_alloc_bf16((size_t)m.stage_cap * K3D_NT * K3D_H); ok = ok && m.stage;
    const size_t Q = K3D_Q;
    m.hid = k3d_alloc_bf16(Q * K3D_H); m.acc = k3d_alloc_bf16(Q * K3D_H); m.normed = k3d_alloc_bf16(Q * K3D_H); m.ql = k3d_alloc_bf16(Q * K3D_QR);
    m.latent = k3d_alloc_bf16(Q * K3D_KR); m.heads16 = k3d_alloc_bf16(Q * K3D_NH * K3D_VD); m.acc2 = k3d_alloc_bf16(Q * K3D_H); m.post = k3d_alloc_bf16(Q * K3D_H);
    m.mid = k3d_alloc_bf16(Q * K3D_I); m.resid = k3d_alloc_bf16(Q * K3D_H); m.scored = k3d_alloc_bf16(Q * K3D_H); m.embs = k3d_alloc_bf16(Q * K3D_MR);
    m.qa = k3d_alloc_f32(Q * K3D_QR); m.q = k3d_alloc_f32(Q * K3D_NH * K3D_QK); m.proj = k3d_alloc_f32(Q * K3D_KVW); m.kvb = k3d_alloc_f32(Q * K3D_NH * (K3D_NOPE + K3D_VD));
    m.k_nope = k3d_alloc_f32(Q * K3D_NH * K3D_NOPE); m.value = k3d_alloc_f32(Q * K3D_NH * K3D_VD); m.heads = k3d_alloc_f32(Q * K3D_NH * K3D_VD); m.attn = k3d_alloc_f32(Q * K3D_H);
    m.g = k3d_alloc_f32(Q * K3D_I); m.u = k3d_alloc_f32(Q * K3D_I); m.mlp = k3d_alloc_f32(Q * K3D_H); m.conf = k3d_alloc_f32(Q);
    if (cudaMalloc(&m.idx, sizeof(int) * Q) != cudaSuccess) { cudaGetLastError(); ok = false; }
    const size_t C = K3D_CHUNK;
    m.c_ctx = k3d_alloc_f32(C * K3D_H); m.c_ctx16 = k3d_alloc_bf16(C * K3D_H); m.c_stacked = k3d_alloc_f32(C * K3D_NL * K3D_KVW);
    m.c_latent = k3d_alloc_bf16(C * K3D_KR); m.c_kvb = k3d_alloc_f32(C * K3D_NH * (K3D_NOPE + K3D_VD)); m.c_knope = k3d_alloc_f32(C * K3D_NH * K3D_NOPE);
    m.c_value = k3d_alloc_f32(C * K3D_NH * K3D_VD); m.c_kvraw = k3d_alloc_f32(C * K3D_KVW);
    ok = ok && m.hid && m.acc && m.normed && m.ql && m.latent && m.heads16 && m.acc2 && m.post && m.mid && m.resid && m.scored && m.embs &&
         m.qa && m.q && m.proj && m.kvb && m.k_nope && m.value && m.heads && m.attn && m.g && m.u && m.mlp && m.conf &&
         m.c_ctx && m.c_ctx16 && m.c_stacked && m.c_latent && m.c_kvb && m.c_knope && m.c_value && m.c_kvraw;
    if (!ok) { fprintf(stderr, "ds4: K3 drafter: device allocation failed\n"); return 0; }
    m.len = 0;
    m.ready = true;
    fprintf(stderr, "ds4: K3 drafter: native DSpark loaded in %.1f s (%.2f GB on the device: weights + caches for %u positions)\n",
            k3_now_sec() - t0, m.gpu_bytes / 1e9, m.cap);
    return 1;
}
extern "C" double ds4_gpu_k3_draft_gpu_bytes(void) { return g_k3d.gpu_bytes; }
extern "C" uint32_t ds4_gpu_k3_draft_len(void) { return g_k3d.len; }
extern "C" int ds4_gpu_k3_draft_truncate(uint32_t len) { if (!g_k3d.ready) return 0; if (len < g_k3d.len) g_k3d.len = len; return 1; }

/* rows [T][n] fp32 on the device (the residual stream after capture layer ci) -> staged bf16 column ci */
extern "C" int ds4_gpu_k3_draft_capture(int ci, const ds4_gpu_tensor *cur, uint32_t T) {
    k3d_model &m = g_k3d;
    if (!m.ready || ci < 0 || ci >= (int)K3D_NT || !cur || T == 0u || T > m.stage_cap) return 0;
    const size_t n = (size_t)T * K3D_H;
    k3d_rows_to_bf16_kernel<<<k3d_grid(n), 256, 0, cuda_decode_stream()>>>(m.stage, (const float *)cur->ptr, T, K3D_H, (uint32_t)ci);
    return cuda_ok(cudaGetLastError(), "K3 drafter capture");
}
/* the first T staged rows become context positions pos0..pos0+T-1 */
extern "C" int ds4_gpu_k3_draft_commit(uint32_t pos0, uint32_t T) {
    k3d_model &m = g_k3d;
    if (!m.ready || T == 0u || T > m.stage_cap) return 0;
    if (pos0 != m.len) { fprintf(stderr, "ds4: K3 drafter: commit at %u but the context holds %u rows\n", pos0, m.len); return 0; }
    if (pos0 + T + K3D_Q >= m.cap) { fprintf(stderr, "ds4: K3 drafter: context full (%u)\n", m.cap); return 0; }
    cudaStream_t stream = cuda_decode_stream();
    for (uint32_t c0 = 0; c0 < T; c0 += K3D_CHUNK) {
        const uint32_t C = T - c0 < K3D_CHUNK ? T - c0 : K3D_CHUNK;
        const __nv_bfloat16 *rows = m.stage + (size_t)c0 * K3D_NT * K3D_H;
        if (!k3d_gemm(m.ctx_proj, rows, m.c_ctx, K3D_H, C, K3D_NT * K3D_H)) return 0;
        k3d_rms(m.c_ctx16, m.c_ctx, nullptr, K3D_H, m.ctx_norm, C, K3D_H);
        if (!k3d_gemm(m.fused_kv_a, m.c_ctx16, m.c_stacked, K3D_NL * K3D_KVW, C, K3D_H)) return 0;
        for (uint32_t l = 0; l < K3D_NL; l++) {
            const float *slice = m.c_stacked + (size_t)l * K3D_KVW;   /* row stride K3D_NL*K3D_KVW */
            k3d_rms(m.c_latent, slice, nullptr, K3D_NL * K3D_KVW, m.L[l].kv_a_norm, C, K3D_KR);
            if (!k3d_gemm(m.L[l].kv_b, m.c_latent, m.c_kvb, K3D_NH * (K3D_NOPE + K3D_VD), C, K3D_KR)) return 0;
            k3_split_kvb_batch_kernel<<<dim3((K3D_NH * (K3D_NOPE + K3D_VD) + 255u) / 256u, C, 1u), 256, 0, stream>>>(m.c_knope, m.c_value, m.c_kvb, K3D_NH, K3D_NOPE, K3D_VD);
            if (cudaMemcpy2DAsync(m.c_kvraw, K3D_KVW * sizeof(float), slice, (size_t)K3D_NL * K3D_KVW * sizeof(float), K3D_KVW * sizeof(float), C, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 0;
            k3d_rope_kernel<<<C, 32, 0, stream>>>(m.c_kvraw, K3D_KVW, 1u, 0u, K3D_KR, m.inv_freq, pos0 + c0, 1.0f);
            k3_mla_store_batch_kernel<<<dim3((K3D_NH * K3D_QK + K3D_NH * K3D_VD + 255u) / 256u, C, 1u), 256, 0, stream>>>(
                m.key_cache[l], m.value_cache[l], m.c_knope, m.c_kvraw, m.c_value, pos0 + c0, K3D_NH, K3D_NOPE, K3D_ROPE, K3D_KR, K3D_VD);
        }
    }
    if (!cuda_ok(cudaGetLastError(), "K3 drafter commit")) return 0;
    m.len = pos0 + T;
    return 1;
}
/* seven K3 embedding rows [7][n] fp32 (anchor + six mask tokens) -> final-normed hidden rows [7][n] fp32 for the K3 head */
extern "C" int ds4_gpu_k3_draft_backbone(const ds4_gpu_tensor *embed7, uint32_t cache_len, ds4_gpu_tensor *hidden7_out) {
    k3d_model &m = g_k3d;
    if (!m.ready || !embed7 || !hidden7_out || cache_len != m.len || cache_len + K3D_Q >= m.cap) return 0;
    cudaStream_t stream = cuda_decode_stream();
    const uint32_t Q = K3D_Q;
    const float qscale = (float)(k3d_yarn_mscale(K3D_FACTOR, 1.0) * k3d_yarn_mscale(K3D_FACTOR, 1.0));
    k3d_f32_to_bf16_kernel<<<k3d_grid((size_t)Q * K3D_H), 256, 0, stream>>>(m.hid, (const float *)embed7->ptr, (size_t)Q * K3D_H);
    const __nv_bfloat16 *residual = nullptr;
    for (uint32_t l = 0; l < K3D_NL; l++) {
        const k3d_layer &L = m.L[l];
        const __nv_bfloat16 *acc = m.hid;
        if (residual) { k3d_add16_kernel<<<k3d_grid((size_t)Q * K3D_H), 256, 0, stream>>>(m.acc, residual, m.hid, (size_t)Q * K3D_H); acc = m.acc; }
        k3d_rms(m.normed, nullptr, acc, K3D_H, L.in_norm, Q, K3D_H);
        if (!k3d_gemm(L.q_a, m.normed, m.qa, K3D_QR, Q, K3D_H)) return 0;
        k3d_rms(m.ql, m.qa, nullptr, K3D_QR, L.q_a_norm, Q, K3D_QR);
        if (!k3d_gemm(L.q_b, m.ql, m.q, K3D_NH * K3D_QK, Q, K3D_QR)) return 0;
        k3d_q_nope_kernel<<<k3d_grid((size_t)Q * K3D_NH * K3D_QK), 256, 0, stream>>>(m.q, Q, qscale);
        k3d_rope_kernel<<<Q * K3D_NH, 32, 0, stream>>>(m.q, K3D_NH * K3D_QK, K3D_NH, K3D_QK, K3D_NOPE, m.inv_freq, cache_len, qscale);
        if (!k3d_gemm(L.kv_a, m.normed, m.proj, K3D_KVW, Q, K3D_H)) return 0;
        k3d_rms(m.latent, m.proj, nullptr, K3D_KVW, L.kv_a_norm, Q, K3D_KR);
        k3d_rope_kernel<<<Q, 32, 0, stream>>>(m.proj, K3D_KVW, 1u, 0u, K3D_KR, m.inv_freq, cache_len, 1.0f);
        if (!k3d_gemm(L.kv_b, m.latent, m.kvb, K3D_NH * (K3D_NOPE + K3D_VD), Q, K3D_KR)) return 0;
        k3_split_kvb_batch_kernel<<<dim3((K3D_NH * (K3D_NOPE + K3D_VD) + 255u) / 256u, Q, 1u), 256, 0, stream>>>(m.k_nope, m.value, m.kvb, K3D_NH, K3D_NOPE, K3D_VD);
        k3_mla_store_batch_kernel<<<dim3((K3D_NH * K3D_QK + K3D_NH * K3D_VD + 255u) / 256u, Q, 1u), 256, 0, stream>>>(
            m.key_cache[l], m.value_cache[l], m.k_nope, m.proj, m.value, cache_len, K3D_NH, K3D_NOPE, K3D_ROPE, K3D_KR, K3D_VD);
        const uint32_t n_pos = cache_len + Q;   /* every query row sees the context and all seven rows */
        for (uint32_t r = 0; r < Q; r++) {
            k3_mla_attend_kernel<<<K3D_NH, 256, n_pos * sizeof(float), stream>>>(m.heads + (size_t)r * K3D_NH * K3D_VD, m.q + (size_t)r * K3D_NH * K3D_QK,
                m.key_cache[l], m.value_cache[l], n_pos, K3D_NH, K3D_QK, K3D_VD, 1.0f / sqrtf((float)K3D_QK));
        }
        k3d_f32_to_bf16_kernel<<<k3d_grid((size_t)Q * K3D_NH * K3D_VD), 256, 0, stream>>>(m.heads16, m.heads, (size_t)Q * K3D_NH * K3D_VD);
        if (!k3d_gemm(L.o, m.heads16, m.attn, K3D_H, Q, K3D_NH * K3D_VD)) return 0;
        k3d_add_kernel<<<k3d_grid((size_t)Q * K3D_H), 256, 0, stream>>>(m.acc2, acc, m.attn, (size_t)Q * K3D_H);
        k3d_rms(m.post, nullptr, m.acc2, K3D_H, L.post_norm, Q, K3D_H);
        if (!k3d_gemm(L.gate, m.post, m.g, K3D_I, Q, K3D_H) || !k3d_gemm(L.up, m.post, m.u, K3D_I, Q, K3D_H)) return 0;
        k3d_silu_mul_kernel<<<k3d_grid((size_t)Q * K3D_I), 256, 0, stream>>>(m.mid, m.g, m.u, (size_t)Q * K3D_I);
        if (!k3d_gemm(L.down, m.mid, m.mlp, K3D_H, Q, K3D_I)) return 0;
        k3d_f32_to_bf16_kernel<<<k3d_grid((size_t)Q * K3D_H), 256, 0, stream>>>(m.hid, m.mlp, (size_t)Q * K3D_H);
        if (cudaMemcpyAsync(m.resid, m.acc2, (size_t)Q * K3D_H * 2, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 0;
        residual = m.resid;
    }
    k3d_add16_kernel<<<k3d_grid((size_t)Q * K3D_H), 256, 0, stream>>>(m.acc, residual, m.hid, (size_t)Q * K3D_H);
    k3d_rms(m.scored, nullptr, m.acc, K3D_H, m.final_norm, Q, K3D_H);
    k3d_bf16_to_f32_kernel<<<k3d_grid((size_t)Q * K3D_H), 256, 0, stream>>>((float *)hidden7_out->ptr, m.scored, (size_t)Q * K3D_H);
    return cuda_ok(cudaGetLastError(), "K3 drafter backbone");
}
/* logits [7][vocab] fp32 from the K3 head -> the seven proposals (Markov-corrected argmax chain) and their confidence */
extern "C" int ds4_gpu_k3_draft_finish(ds4_gpu_tensor *logits7, int anchor, int ids[7], float conf[7]) {
    k3d_model &m = g_k3d;
    if (!m.ready || !logits7 || anchor < 0 || anchor >= (int)K3D_V) return 0;
    cudaStream_t stream = cuda_decode_stream();
    float *logits = (float *)logits7->ptr;
    int prev = anchor;
    for (uint32_t r = 0; r < K3D_Q; r++) {
        const __nv_bfloat16 *e = m.mk_w1 + (size_t)prev * K3D_MR;
        if (cudaMemcpyAsync(m.embs + (size_t)r * K3D_MR, e, K3D_MR * 2, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 0;
        k3d_markov_bias_kernel<<<k3d_grid(K3D_V), 256, 0, stream>>>(logits + (size_t)r * K3D_V, m.mk_w2, e, K3D_V, K3D_MR);
        k3d_argmax_kernel<<<1, 1024, 0, stream>>>(m.idx + r, logits + (size_t)r * K3D_V, K3D_V);
        int v = 0;
        if (cudaMemcpyAsync(&v, m.idx + r, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) return 0;
        ids[r] = v; prev = v;
    }
    k3d_conf_kernel<<<K3D_Q, 256, 0, stream>>>(m.conf, m.scored, m.embs, m.conf_w, m.conf_b);
    float c[K3D_Q];
    if (cudaMemcpyAsync(c, m.conf, sizeof(c), cudaMemcpyDeviceToHost, stream) != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) return 0;
    for (uint32_t r = 0; r < K3D_Q; r++) conf[r] = 1.0f / (1.0f + expf(-c[r]));
    return cuda_ok(cudaGetLastError(), "K3 drafter finish");
}
