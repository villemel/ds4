// Kimi K3 / K3.1 Pro on Metal (8 Nov 2026): the decode-path kernels of the CUDA port (ds4_cuda.cu, "Kimi K3 CUDA primitives"),
// one for one, with the same per-thread work split. Arithmetic follows the CUDA kernels' order where the structure allows; Metal
// and CUDA results are close, not bit-identical (different GPUs, different math libraries). Within Metal every kernel is
// deterministic: no atomics, fixed reduction trees.

#include <metal_stdlib>
using namespace metal;

constant float k3_e2m1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                              -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
// step M22: the table's floats from the code's bits (no memory access)
static inline float k3_e2m1_fast(uint c) {
    const uint mag = c & 7u;
    const uint bits = mag == 0u ? 0u : ((mag + (mag == 1u ? 251u : 252u)) << 22);
    return as_type<float>(bits | ((c & 8u) << 28));
}
#if defined(K3_E2M1_BITS)
#define K3_E2M1(c) k3_e2m1_fast(c)
#else
#define K3_E2M1(c) k3_e2m1[c]
#endif

static inline float k3_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float k3_silu(float x) { return x * k3_sigmoid(x); }
static inline float k3_situ(float g, float u, float b, float bl) { return (b * tanh(g / b) * k3_sigmoid(g)) * (bl * tanh(u / bl)); }
static inline float k3_simd_xor_sum(float v) {   // the CUDA butterfly (xor 16, 8, 4, 2, 1)
    v += simd_shuffle_xor(v, 16); v += simd_shuffle_xor(v, 8); v += simd_shuffle_xor(v, 4);
    v += simd_shuffle_xor(v, 2);  v += simd_shuffle_xor(v, 1);
    return v;
}

struct k3_mix_args { uint n_anchor; uint n; float eps; uint anchor_stride; };   // anchor_stride: floats between tokens' anchor sets

// Residual-anchor mix: values = [anchors 0..n_anchor-1, x]; score_i = sum_j rmsnorm(v_i)[j] * norm[j] * proj[j]; out = softmax(score) . values.
// One threadgroup of 1024 threads per token (grid.x = tokens).
kernel void kernel_k3_residual_mix(constant k3_mix_args &a [[buffer(0)]], device const float *x [[buffer(1)]],
                                   device const float *anchors [[buffer(2)]], device const float *res_norm [[buffer(3)]],
                                   device const float *res_proj [[buffer(4)]], device float *out [[buffer(5)]],
                                   uint tid [[thread_position_in_threadgroup]], uint t [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    threadgroup float scores[17];
    threadgroup float probs[17];
    const uint n = a.n, count = a.n_anchor + 1u;
    device const float *xt = x + (ulong)t * n;
    device const float *at = anchors + (ulong)t * a.anchor_stride;
    for (uint i = 0; i < count; i++) {
        device const float *v = i < a.n_anchor ? at + (ulong)i * n : xt;
        float ss = 0.0f;
        for (uint j = tid; j < n; j += 1024u) ss += v[j] * v[j];
        ss = k3_simd_xor_sum(ss);
        if (lane == 0u) red[warp] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tt = red[lane]; tt = k3_simd_xor_sum(tt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = rsqrt(tt / (float)n + a.eps);
        float sc = 0.0f;
        for (uint j = tid; j < n; j += 1024u) sc += v[j] * inv * res_norm[j] * res_proj[j];
        sc = k3_simd_xor_sum(sc);
        if (lane == 0u) red[warp] = sc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s2 = red[lane]; s2 = k3_simd_xor_sum(s2);
        if (tid == 0u) scores[i] = s2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float mx = scores[0];
        for (uint i = 1; i < count; i++) mx = max(mx, scores[i]);
        float denom = 0.0f;
        for (uint i = 0; i < count; i++) { probs[i] = exp(scores[i] - mx); denom += probs[i]; }
        for (uint i = 0; i < count; i++) probs[i] /= denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *ot = out + (ulong)t * n;
    for (uint j = tid; j < n; j += 1024u) {
        float acc = 0.0f;
        for (uint i = 0; i < a.n_anchor; i++) acc += probs[i] * at[(ulong)i * n + j];
        acc += probs[a.n_anchor] * xt[j];
        ot[j] = acc;
    }
}

struct k3_ew_args { uint n; float b; float bl; };
kernel void kernel_k3_situ(constant k3_ew_args &a [[buffer(0)]], device const float *gate [[buffer(1)]], device const float *up [[buffer(2)]],
                           device float *out [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) out[i] = k3_situ(gate[i], up[i], a.b, a.bl);
}
kernel void kernel_k3_sigmoid_mul(constant k3_ew_args &a [[buffer(0)]], device const float *x [[buffer(1)]], device const float *gate [[buffer(2)]],
                                  device float *out [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) out[i] = x[i] * k3_sigmoid(gate[i]);
}

struct k3_kvb_args { uint heads; uint nope; uint vdim; };
// kv_b rows [heads][nope + vdim] -> k_nope [heads][nope], value [heads][vdim]; grid.y = token
kernel void kernel_k3_split_kvb(constant k3_kvb_args &a [[buffer(0)]], device float *k_nope [[buffer(1)]], device float *value [[buffer(2)]],
                                device const float *kvb [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, t = gid.y, per_head = a.nope + a.vdim;
    if (i >= a.heads * per_head) return;
    const uint h = i / per_head, d = i % per_head;
    const float v = kvb[(ulong)t * a.heads * per_head + i];
    if (d < a.nope) k_nope[(ulong)t * a.heads * a.nope + h * a.nope + d] = v;
    else value[(ulong)t * a.heads * a.vdim + h * a.vdim + (d - a.nope)] = v;
}

struct k3_store_args { uint pos0; uint heads; uint nope; uint rot; uint kv_lora; uint vdim; };
// dense f32 caches: key [cap][heads][nope+rot], value [cap][heads][vdim]; the rope dims copied unrotated from kv_raw; grid.y = token
kernel void kernel_k3_mla_store(constant k3_store_args &a [[buffer(0)]], device float *key_cache [[buffer(1)]], device float *value_cache [[buffer(2)]],
                                device const float *k_nope [[buffer(3)]], device const float *kv_raw [[buffer(4)]], device const float *value [[buffer(5)]],
                                uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, t = gid.y, pos = a.pos0 + t, qk = a.nope + a.rot, key_elems = a.heads * qk;
    device const float *kn = k_nope + (ulong)t * a.heads * a.nope;
    device const float *kr = kv_raw + (ulong)t * (a.kv_lora + a.rot);
    device const float *vv = value + (ulong)t * a.heads * a.vdim;
    if (i < key_elems) {
        const uint h = i / qk, d = i % qk;
        key_cache[((ulong)pos * a.heads + h) * qk + d] = d < a.nope ? kn[h * a.nope + d] : kr[a.kv_lora + (d - a.nope)];
    } else if (i < key_elems + a.heads * a.vdim) {
        const uint j = i - key_elems;
        value_cache[(ulong)pos * a.heads * a.vdim + j] = vv[j];
    }
}

struct k3_attn_args { uint n_pos; uint heads; uint qk; uint vdim; float scale; uint use_mask; };
// One query over n_pos cached positions (decode). One threadgroup of 256 per head; scores in a device scratch row per head
// (threadgroup memory is 32 KB on Apple GPUs). Same per-thread split as the CUDA kernel: thread t takes positions t, t+256, ...
kernel void kernel_k3_mla_attend(constant k3_attn_args &a [[buffer(0)]], device float *heads_out [[buffer(1)]], device const float *q [[buffer(2)]],
                                 device const float *key_cache [[buffer(3)]], device const float *value_cache [[buffer(4)]],
                                 device float *scratch [[buffer(5)]], device const uchar *mask [[buffer(6)]],
                                 uint tid [[thread_position_in_threadgroup]], uint h [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    threadgroup float stats[2];
    device float *sc = scratch + (ulong)h * a.n_pos;
    device const float *qh = q + (ulong)h * a.qk;
    float local_max = -1e30f;
    for (uint t = tid; t < a.n_pos; t += 256u) {
        float s = -INFINITY;
        if (!a.use_mask || mask[t]) {
            device const float *kt = key_cache + ((ulong)t * a.heads + h) * a.qk;
            s = 0.0f;
            for (uint d = 0; d < a.qk; d++) s = fma(qh[d], kt[d], s);
            s *= a.scale;
        }
        sc[t] = s;
        local_max = max(local_max, s);
    }
    local_max = max(local_max, simd_shuffle_xor(local_max, 16)); local_max = max(local_max, simd_shuffle_xor(local_max, 8));
    local_max = max(local_max, simd_shuffle_xor(local_max, 4));  local_max = max(local_max, simd_shuffle_xor(local_max, 2));
    local_max = max(local_max, simd_shuffle_xor(local_max, 1));
    if (lane == 0u) red[warp] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (tid == 0u) { float m = -1e30f; for (uint w = 0; w < 8u; w++) m = max(m, red[w]); stats[0] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m = stats[0];
    float local_sum = 0.0f;
    for (uint t = tid; t < a.n_pos; t += 256u) { const float e = exp(sc[t] - m); sc[t] = e; local_sum += e; }
    local_sum = k3_simd_xor_sum(local_sum);
    if (lane == 0u) red[warp] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (tid == 0u) { float s = 0.0f; for (uint w = 0; w < 8u; w++) s += red[w]; stats[1] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / stats[1];
    device float *o = heads_out + (ulong)h * a.vdim;
    for (uint v = tid; v < a.vdim; v += 256u) {
        float acc = 0.0f;
        for (uint t = 0; t < a.n_pos; t++) { const float w = sc[t]; if (w != 0.0f || acc == 0.0f) acc = fma(w, value_cache[((ulong)t * a.heads + h) * a.vdim + v], acc); }
        o[v] = acc * inv;
    }
}

// ---- Kimi Delta Attention, one token step for one head (threadgroup = 128 threads), the CUDA k3_kda_step ----
struct k3_kda_args { uint n_heads; float lower_bound; float norm_eps; uint per_channel; };
kernel void kernel_k3_kda_decode(constant k3_kda_args &a [[buffer(0)]], device float *out [[buffer(1)]], device float *conv_state [[buffer(2)]],
                                 device float *state [[buffer(3)]], device const float *q_in [[buffer(4)]], device const float *k_in [[buffer(5)]],
                                 device const float *v_in [[buffer(6)]], device const float *raw_gate [[buffer(7)]], device const float *raw_beta [[buffer(8)]],
                                 device const float *output_gate [[buffer(9)]], device const float *q_conv [[buffer(10)]], device const float *k_conv [[buffer(11)]],
                                 device const float *v_conv [[buffer(12)]], device const float *a_log [[buffer(13)]], device const float *dt_bias [[buffer(14)]],
                                 device const float *output_norm [[buffer(15)]],
                                 uint tid [[thread_position_in_threadgroup]], uint head [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    constexpr uint DIM = 128u, HIST = 3u;
    threadgroup float sq[128], sk[128], sd[128], sv[128], so[128];
    threadgroup float reduce_q[4], reduce_k[4], reduce_o[4], beta_shared[1];
    const uint projection = a.n_heads * DIM, channel = head * DIM + tid;
    device float *q_state = conv_state;
    device float *k_state = q_state + HIST * projection;
    device float *v_state = k_state + HIST * projection;
    float q_acc = 0.0f, k_acc = 0.0f, v_acc = 0.0f;
    for (uint w = 0; w < HIST; w++) {
        q_acc = fma(q_state[(ulong)w * projection + channel], q_conv[(ulong)channel * 4u + w], q_acc);
        k_acc = fma(k_state[(ulong)w * projection + channel], k_conv[(ulong)channel * 4u + w], k_acc);
        v_acc = fma(v_state[(ulong)w * projection + channel], v_conv[(ulong)channel * 4u + w], v_acc);
    }
    const float q_new = q_in[head * DIM + tid], k_new = k_in[head * DIM + tid], v_new = v_in[head * DIM + tid];
    q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
    k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
    v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);
    q_state[channel] = q_state[projection + channel]; q_state[projection + channel] = q_state[2u * projection + channel]; q_state[2u * projection + channel] = q_new;
    k_state[channel] = k_state[projection + channel]; k_state[projection + channel] = k_state[2u * projection + channel]; k_state[2u * projection + channel] = k_new;
    v_state[channel] = v_state[projection + channel]; v_state[projection + channel] = v_state[2u * projection + channel]; v_state[2u * projection + channel] = v_new;
    sq[tid] = k3_silu(q_acc); sk[tid] = k3_silu(k_acc); sv[tid] = k3_silu(v_acc);
    const float gate = raw_gate[head * DIM + tid] + dt_bias[channel];
    const float al = a.per_channel ? a_log[tid] : a_log[head];
    sd[tid] = exp(a.lower_bound * k3_sigmoid(exp(al) * gate));
    if (tid == 0u) beta_shared[0] = k3_sigmoid(raw_beta[head]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float q_sumsq = k3_simd_xor_sum(sq[tid] * sq[tid]);
    const float k_sumsq = k3_simd_xor_sum(sk[tid] * sk[tid]);
    if (lane == 0u) { reduce_q[warp] = q_sumsq; reduce_k[warp] = k_sumsq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
    float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
    q_total = simd_broadcast(k3_simd_xor_sum(q_total), 0);
    k_total = simd_broadcast(k3_simd_xor_sum(k_total), 0);
    const float qn = sq[tid] * (rsqrt(q_total + 1.0e-6f) * 0.08838834764831845f);
    const float kn = sk[tid] * rsqrt(k_total + 1.0e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sq[tid] = qn; sk[tid] = kn;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint k0 = lane * 4u;
    const float4 q4 = float4(sq[k0], sq[k0 + 1], sq[k0 + 2], sq[k0 + 3]);
    const float4 k4 = float4(sk[k0], sk[k0 + 1], sk[k0 + 2], sk[k0 + 3]);
    const float4 decay4 = float4(sd[k0], sd[k0 + 1], sd[k0 + 2], sd[k0 + 3]);
    const ulong state_head = (ulong)head * DIM * DIM;
    for (uint value = warp; value < DIM; value += 4u) {
        device float4 *hptr = (device float4 *)(state + state_head + (ulong)value * DIM + k0);
        float4 hv = *hptr;
        hv *= decay4;
        const float hk = simd_broadcast(k3_simd_xor_sum(hv.x * k4.x + hv.y * k4.y + hv.z * k4.z + hv.w * k4.w), 0);
        const float delta_v = (sv[value] - hk) * beta_shared[0];
        hv.x = fma(k4.x, delta_v, hv.x); hv.y = fma(k4.y, delta_v, hv.y); hv.z = fma(k4.z, delta_v, hv.z); hv.w = fma(k4.w, delta_v, hv.w);
        *hptr = hv;
        const float hq = simd_broadcast(k3_simd_xor_sum(hv.x * q4.x + hv.y * q4.y + hv.z * q4.z + hv.w * q4.w), 0);
        if (lane == 0u) so[value] = hq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float o_sumsq = k3_simd_xor_sum(so[tid] * so[tid]);
    if (lane == 0u) reduce_o[warp] = o_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o_total = lane < 4u ? reduce_o[lane] : 0.0f;
    o_total = simd_broadcast(k3_simd_xor_sum(o_total), 0);
    const float o_scale = rsqrt(o_total / (float)DIM + a.norm_eps);
    const float out_gate = k3_sigmoid(output_gate[head * DIM + tid]);
    out[head * DIM + tid] = so[tid] * o_scale * output_norm[tid] * out_gate;
}

// ---- routed experts: one MXFP4 slab = [w1 packed][w1 scales][w2 packed][w2 scales][w3 packed][w3 scales] ----
struct k3_moe_args { uint ff; uint latent; ulong w1p; ulong w1s; ulong w2p; ulong w2s; ulong w3p; ulong w3s; float b; float bl; float weight; };
// one simdgroup per output row; lane takes 32-element blocks j = lane, lane + 32, ... (the CUDA k3_raw_row_dot_x order)
static inline float k3_mxfp4_row_dot(device const uchar *packed_row, device const uchar *scale_row, uint blocks,
                                     device const float *x, uint lane) {
    float acc = 0.0f;
    for (uint j = lane; j < blocks; j += 32u) {
        device const uchar *pb = packed_row + (ulong)j * 16u;
        device const float *xb = x + (ulong)j * 32u;
        const uint e = scale_row[j];
        const float d = e == 255u ? as_type<float>(0x7fffffffu) : exp2((float)e - 127.0f);
        float s = 0.0f;
        for (uint i = 0; i < 16u; i++) {
            const uint byte = pb[i];
            s = fma(K3_E2M1(byte & 0x0fu), xb[2u * i], s);
            s = fma(K3_E2M1(byte >> 4), xb[2u * i + 1u], s);
        }
        acc = fma(d, s, acc);
    }
    return k3_simd_xor_sum(acc);
}
// mid[r] = situ(w1[r] . x, w3[r] . x); 8 rows per threadgroup of 256
kernel void kernel_k3_moe_gate_up(constant k3_moe_args &a [[buffer(0)]], device const uchar *slab [[buffer(1)]], device const float *x [[buffer(2)]],
                                  device float *mid [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint r = tg * 8u + warp;
    if (r >= a.ff) return;
    const uint blocks = a.latent / 32u;
    const ulong prow = (ulong)a.latent / 2u;
    const float g = k3_mxfp4_row_dot(slab + a.w1p + (ulong)r * prow, slab + a.w1s + (ulong)r * blocks, blocks, x, lane);
    const float u = k3_mxfp4_row_dot(slab + a.w3p + (ulong)r * prow, slab + a.w3s + (ulong)r * blocks, blocks, x, lane);
    if (lane == 0u) mid[r] = k3_situ(g, u, a.b, a.bl);
}
// part[c] = weight * (w2[c] . mid); 8 rows per threadgroup of 256
kernel void kernel_k3_moe_down(constant k3_moe_args &a [[buffer(0)]], device const uchar *slab [[buffer(1)]], device const float *mid [[buffer(2)]],
                               device float *part [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint c = tg * 8u + warp;
    if (c >= a.latent) return;
    const uint blocks = a.ff / 32u;
    const float v = k3_mxfp4_row_dot(slab + a.w2p + (ulong)c * ((ulong)a.ff / 2u), slab + a.w2s + (ulong)c * blocks, blocks, mid, lane);
    if (lane == 0u) part[c] = a.weight * v;
}
// step M8: all of a token's experts in one dispatch (grid.y = expert slot, slab pointers from GPU addresses) and vector loads;
// each row's arithmetic is the same fma sequence as k3_mxfp4_row_dot (16 bytes = 4 words, byte i = word i/4 >> 8*(i%4)).
struct k3_moe_ptrs { device const uchar *slab[64]; float weight[64]; };
// e2m1 code (sign, 2-bit exponent, 1 mantissa bit) -> the same float as K3_E2M1(), built from bits (no table lookup):
// 0 -> 0, 1 -> 0.5, (e > 0) -> 2^(e-1) * (1 + m/2)
static inline float k3_e2m1_bits(uint c) {
    const uint mag = c & 7u, e = mag >> 1, m = mag & 1u;
    const uint bits = mag == 0u ? 0u : (mag == 1u ? (126u << 23) : (((e + 126u) << 23) | (m << 22)));
    return as_type<float>(bits | ((c & 8u) << 28));
}
static inline float k3_mxfp4_row_dot_v(device const uchar *packed_row, device const uchar *scale_row, uint blocks,
                                       device const float *x, uint lane) {
    float acc = 0.0f;
    for (uint j = lane; j < blocks; j += 32u) {
        const uint4 pk = *((device const uint4 *)(packed_row + (ulong)j * 16u));
        device const float4 *x4 = (device const float4 *)(x + (ulong)j * 32u);
        const uint e = scale_row[j];
        const float d = e == 255u ? as_type<float>(0x7fffffffu) : exp2((float)e - 127.0f);
        float s = 0.0f;
        for (uint q = 0; q < 4u; q++) {
            const uint w = pk[q];
            const float4 xa = x4[2u * q], xb = x4[2u * q + 1u];
            s = fma(K3_E2M1(w & 0x0fu), xa.x, s);         s = fma(K3_E2M1((w >> 4) & 0x0fu), xa.y, s);
            s = fma(K3_E2M1((w >> 8) & 0x0fu), xa.z, s);  s = fma(K3_E2M1((w >> 12) & 0x0fu), xa.w, s);
            s = fma(K3_E2M1((w >> 16) & 0x0fu), xb.x, s); s = fma(K3_E2M1((w >> 20) & 0x0fu), xb.y, s);
            s = fma(K3_E2M1((w >> 24) & 0x0fu), xb.z, s); s = fma(K3_E2M1(w >> 28), xb.w, s);
        }
        acc = fma(d, s, acc);
    }
    return k3_simd_xor_sum(acc);
}
kernel void kernel_k3_moe_gate_up2(constant k3_moe_args &a [[buffer(0)]], constant k3_moe_ptrs &p [[buffer(1)]], device const float *x [[buffer(2)]],
                                   device float *mid [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint r = tg.x * 8u + warp, e = tg.y;
    if (r >= a.ff) return;
    device const uchar *slab = p.slab[e];
    const uint blocks = a.latent / 32u;
    const ulong prow = (ulong)a.latent / 2u;
    const float g = k3_mxfp4_row_dot_v(slab + a.w1p + (ulong)r * prow, slab + a.w1s + (ulong)r * blocks, blocks, x, lane);
    const float u = k3_mxfp4_row_dot_v(slab + a.w3p + (ulong)r * prow, slab + a.w3s + (ulong)r * blocks, blocks, x, lane);
    if (lane == 0u) mid[(ulong)e * a.ff + r] = k3_situ(g, u, a.b, a.bl);
}
kernel void kernel_k3_moe_down2(constant k3_moe_args &a [[buffer(0)]], constant k3_moe_ptrs &p [[buffer(1)]], device const float *mid [[buffer(2)]],
                                device float *part [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint c = tg.x * 8u + warp, e = tg.y;
    if (c >= a.latent) return;
    device const uchar *slab = p.slab[e];
    const uint blocks = a.ff / 32u;
    const float v = k3_mxfp4_row_dot_v(slab + a.w2p + (ulong)c * ((ulong)a.ff / 2u), slab + a.w2s + (ulong)c * blocks, blocks,
                                       mid + (ulong)e * a.ff, lane);
    if (lane == 0u) part[(ulong)e * a.latent + c] = p.weight[e] * v;
}
// step M8b: x reuse. One 32-value block of x is loaded once per lane and serves K3_RPS rows (gate-up: both matrices of each row),
// so the per-row arithmetic is k3_mxfp4_row_dot's exactly: per row, lane's blocks in order j = lane, lane+32, ..., s by i, acc = fma(d, s, acc).
#ifndef K3_RPS
#define K3_RPS 4
#endif
static inline float k3_blk(uint4 pk, thread const float4 *xv) {
    float s = 0.0f;
    for (uint q = 0; q < 4u; q++) {
        const uint w = pk[q];
        const float4 xa = xv[2u * q], xb = xv[2u * q + 1u];
        s = fma(K3_E2M1(w & 0x0fu), xa.x, s);         s = fma(K3_E2M1((w >> 4) & 0x0fu), xa.y, s);
        s = fma(K3_E2M1((w >> 8) & 0x0fu), xa.z, s);  s = fma(K3_E2M1((w >> 12) & 0x0fu), xa.w, s);
        s = fma(K3_E2M1((w >> 16) & 0x0fu), xb.x, s); s = fma(K3_E2M1((w >> 20) & 0x0fu), xb.y, s);
        s = fma(K3_E2M1((w >> 24) & 0x0fu), xb.z, s); s = fma(K3_E2M1(w >> 28), xb.w, s);
    }
    return s;
}
static inline float k3_scale(uint e) { return e == 255u ? as_type<float>(0x7fffffffu) : exp2((float)e - 127.0f); }
// grid (ceil(ff / (8 * K3_RPS)), n_experts); simdgroup w of threadgroup t takes rows (t*8 + w)*K3_RPS .. +K3_RPS-1
kernel void kernel_k3_moe_gate_up3(constant k3_moe_args &a [[buffer(0)]], constant k3_moe_ptrs &p [[buffer(1)]], device const float *x [[buffer(2)]],
                                   device float *mid [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint r0 = (tg.x * 8u + warp) * K3_RPS, e = tg.y;
    if (r0 >= a.ff) return;
    device const uchar *slab = p.slab[e];
    if (slab == nullptr) return;   /* step M20: this slot runs in the other phase */
    const uint blocks = a.latent / 32u;
    const ulong prow = (ulong)a.latent / 2u;
    float ag[K3_RPS], au[K3_RPS];
    for (uint k = 0; k < K3_RPS; k++) { ag[k] = 0.0f; au[k] = 0.0f; }
    for (uint j = lane; j < blocks; j += 32u) {
        float4 xv[8];
        device const float4 *x4 = (device const float4 *)(x + (ulong)j * 32u);
        for (uint q = 0; q < 8u; q++) xv[q] = x4[q];
        for (uint k = 0; k < K3_RPS; k++) {
            const uint r = r0 + k;
            if (r >= a.ff) break;
            const uint4 pg = *((device const uint4 *)(slab + a.w1p + (ulong)r * prow + (ulong)j * 16u));
            const uint4 pu = *((device const uint4 *)(slab + a.w3p + (ulong)r * prow + (ulong)j * 16u));
            const float dg = k3_scale(slab[a.w1s + (ulong)r * blocks + j]), du = k3_scale(slab[a.w3s + (ulong)r * blocks + j]);
            ag[k] = fma(dg, k3_blk(pg, xv), ag[k]);
            au[k] = fma(du, k3_blk(pu, xv), au[k]);
        }
    }
    for (uint k = 0; k < K3_RPS; k++) {
        const uint r = r0 + k;
        if (r >= a.ff) break;
        const float g = k3_simd_xor_sum(ag[k]), u = k3_simd_xor_sum(au[k]);
        if (lane == 0u) mid[(ulong)e * a.ff + r] = k3_situ(g, u, a.b, a.bl);
    }
}
kernel void kernel_k3_moe_down3(constant k3_moe_args &a [[buffer(0)]], constant k3_moe_ptrs &p [[buffer(1)]], device const float *mid [[buffer(2)]],
                                device float *part [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint c0 = (tg.x * 8u + warp) * K3_RPS, e = tg.y;
    if (c0 >= a.latent) return;
    device const uchar *slab = p.slab[e];
    if (slab == nullptr) return;   /* step M20: this slot runs in the other phase */
    device const float *xm = mid + (ulong)e * a.ff;
    const uint blocks = a.ff / 32u;
    const ulong prow = (ulong)a.ff / 2u;
    float ac[K3_RPS];
    for (uint k = 0; k < K3_RPS; k++) ac[k] = 0.0f;
    for (uint j = lane; j < blocks; j += 32u) {
        float4 xv[8];
        device const float4 *x4 = (device const float4 *)(xm + (ulong)j * 32u);
        for (uint q = 0; q < 8u; q++) xv[q] = x4[q];
        for (uint k = 0; k < K3_RPS; k++) {
            const uint c = c0 + k;
            if (c >= a.latent) break;
            const uint4 pk = *((device const uint4 *)(slab + a.w2p + (ulong)c * prow + (ulong)j * 16u));
            ac[k] = fma(k3_scale(slab[a.w2s + (ulong)c * blocks + j]), k3_blk(pk, xv), ac[k]);
        }
    }
    for (uint k = 0; k < K3_RPS; k++) {
        const uint c = c0 + k;
        if (c >= a.latent) break;
        const float v = k3_simd_xor_sum(ac[k]);
        if (lane == 0u) part[(ulong)e * a.latent + c] = p.weight[e] * v;
    }
}
struct k3_reduce_args { uint n_part; uint latent; };
// out[c] = sum over the token's experts, in ascending expert id, of part rows (the CUDA k3_part_reduce_kernel order)
kernel void kernel_k3_moe_reduce(constant k3_reduce_args &a [[buffer(0)]], device const float *part [[buffer(1)]], device float *out [[buffer(2)]],
                                 uint c [[thread_position_in_grid]]) {
    if (c >= a.latent) return;
    float acc = 0.0f;
    for (uint i = 0; i < a.n_part; i++) acc += part[(ulong)i * a.latent + c];
    out[c] = acc;
}

struct k3_rows_args { uint dst_stride; uint src_stride; uint width; };
kernel void kernel_k3_copy_rows(constant k3_rows_args &a [[buffer(0)]], device float *dst [[buffer(1)]], device const float *src [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x < a.width) dst[(ulong)gid.y * a.dst_stride + gid.x] = src[(ulong)gid.y * a.src_stride + gid.x];
}

// ---- step M11: K3.1 sparse attention (DSA) and the latent attention cache: the CUDA step 95 / 96c / 101 kernels ----
// index keys: LayerNorm over 128 values (tree s = 64 .. 1 in threadgroup memory, the CUDA order), stored at row pos0 + t
struct k3_dsa_ln_args { uint pos0; float eps; };
kernel void kernel_k3_dsa_ln_store(constant k3_dsa_ln_args &a [[buffer(0)]], device float *cache [[buffer(1)]], device const float *raw [[buffer(2)]],
                                   device const float *w [[buffer(3)]], device const float *bias [[buffer(4)]],
                                   uint t [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]]) {
    threadgroup float red[128];
    const float v = raw[(ulong)t * 128u + i];
    red[i] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 64u; s > 0u; s >>= 1u) { if (i < s) red[i] += red[i + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    const float mean = red[0] * (1.0f / 128.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float d = v - mean;
    red[i] = d * d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 64u; s > 0u; s >>= 1u) { if (i < s) red[i] += red[i + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    const float rstd = rsqrt(red[0] * (1.0f / 128.0f) + a.eps);
    cache[(ulong)(a.pos0 + t) * 128u + i] = d * rstd * w[i] + bias[i];
}
// scores[t][row] = scale * sum_h relu(q_h . k_row) * w_h; one simdgroup per (row, token), lane l holds elements l, l+32, l+64, l+96
// (the CUDA warp kernel's tree: levels 64 and 32 in the lane, 16 .. 1 by shuffles); rows past the token's position score -inf
struct k3_dsa_score_args { uint n_rows; uint n_tokens; uint pos0; uint n_head; float scale; };
kernel void kernel_k3_dsa_scores(constant k3_dsa_score_args &a [[buffer(0)]], device float *scores [[buffer(1)]], device const float *q [[buffer(2)]],
                                 device const float *weights [[buffer(3)]], device const float *key [[buffer(4)]], threadgroup float *qs [[threadgroup(0)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint token = tg.y;
    threadgroup float *ws = qs + a.n_head * 128u;
    for (uint i = tid; i < a.n_head * 128u; i += 256u) qs[i] = q[(ulong)token * a.n_head * 128u + i];
    for (uint i = tid; i < a.n_head; i += 256u) ws[i] = weights[(ulong)token * a.n_head + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint visible = a.pos0 + token + 1u, lim = visible < a.n_rows ? visible : a.n_rows;
    const uint row0 = tg.x * 256u;
    for (uint r = warp; r < 256u; r += 8u) {
        const uint row = row0 + r;
        if (row >= a.n_rows) break;
        if (row >= lim) { if (lane == 0u) scores[(ulong)token * a.n_rows + row] = -INFINITY; continue; }
        device const float *kr = key + (ulong)row * 128u;
        const float k0 = kr[lane], k1 = kr[lane + 32u], k2 = kr[lane + 64u], k3 = kr[lane + 96u];
        float total = 0.0f;
        for (uint h = 0; h < a.n_head; h++) {
            threadgroup const float *qh = qs + h * 128u;
            const float p0 = qh[lane] * k0, p2 = qh[lane + 64u] * k2, p1 = qh[lane + 32u] * k1, p3 = qh[lane + 96u] * k3;
            float v = (p0 + p2) + (p1 + p3);
            v += simd_shuffle_down(v, 16u);
            v += simd_shuffle_down(v, 8u);
            v += simd_shuffle_down(v, 4u);
            v += simd_shuffle_down(v, 2u);
            v += simd_shuffle_down(v, 1u);
            if (lane == 0u) total += max(v, 0.0f) * ws[h];
        }
        if (lane == 0u) scores[(ulong)token * a.n_rows + row] = total * a.scale;
    }
}
// selection mask: every visible position while there are at most top_k, else the selected ones
struct k3_dsa_mask_args { uint stride; uint pos0; uint top_k; };
kernel void kernel_k3_dsa_mask(constant k3_dsa_mask_args &a [[buffer(0)]], device uchar *mask [[buffer(1)]], device const uint *selected [[buffer(2)]],
                               uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint visible = a.pos0 + t + 1u;
    device uchar *m = mask + (ulong)t * a.stride;
    if (visible <= a.top_k) { for (uint p = tid; p < a.stride; p += 256u) m[p] = p < visible ? (uchar)1 : (uchar)0; return; }
    for (uint p = tid; p < a.stride; p += 256u) m[p] = (uchar)0;
    threadgroup_barrier(mem_flags::mem_device);
    device const uint *sel = selected + (ulong)t * a.top_k;
    for (uint i = tid; i < a.top_k; i += 256u) { const uint p = sel[i]; if (p < visible) m[p] = (uchar)1; }
}
// latent cache (step 96c): kv_b rows straight into the shared dense K/V; with a mask only the selected positions
struct k3_lat_args { uint pos0; uint heads; uint nope; uint rot; uint kv_lora; uint vdim; uint use_mask; };
kernel void kernel_k3_lat_store_kvb(constant k3_lat_args &a [[buffer(0)]], device float *key_cache [[buffer(1)]], device float *value_cache [[buffer(2)]],
                                    device const float *kvb [[buffer(3)]], device const float *kv_raw [[buffer(4)]], device const uchar *mask [[buffer(5)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    const ulong t = gid.y;
    const uint pos = a.pos0 + (uint)t;
    if (a.use_mask != 0u && mask[pos] == 0) return;
    const uint per = a.nope + a.vdim, qk = a.nope + a.rot;
    device const float *row = kvb + t * a.heads * per;
    device const float *raw = kv_raw + t * (a.kv_lora + a.rot);
    const uint i = gid.x, key_elems = a.heads * qk;
    if (i < key_elems) {
        const uint h = i / qk, d = i % qk;
        key_cache[((ulong)pos * a.heads + h) * qk + d] = d < a.nope ? row[h * per + d] : raw[a.kv_lora + (d - a.nope)];
    } else if (i < key_elems + a.heads * a.vdim) {
        const uint j = i - key_elems, h = j / a.vdim, d = j % a.vdim;
        value_cache[(ulong)pos * a.heads * a.vdim + j] = row[h * per + a.nope + d];
    }
}

// ---- step M12: multi-token routed experts for batches (prompt passes, verify blocks). Each union expert u has a list of
// (token, slot, weight) entries; a simdgroup takes K3_RPS rows and walks the expert's tokens in groups of K3_MT_TG, loading each
// weight block once per group. Per (row, token) the arithmetic is kernel_k3_moe_gate_up3 / down3's: k3_blk then
// acc = fma(d, s, acc) over the lane's blocks in order, then k3_simd_xor_sum.
#define K3_MT_TG 4
struct k3_mt_expert { device const uchar *slab; uint off; uint count; };
struct k3_mt_entry { uint token; uint slot; float weight; uint pad; };
kernel void kernel_k3_moe_gate_up_mt(constant k3_moe_args &a [[buffer(0)]], device const k3_mt_expert *ex [[buffer(1)]],
                                     device const k3_mt_entry *ent [[buffer(2)]], device const float *x [[buffer(3)]],
                                     device float *mid [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint r0 = (tg.x * 8u + warp) * K3_RPS;
    if (r0 >= a.ff) return;
    const k3_mt_expert e = ex[tg.y];
    device const uchar *slab = e.slab;
    const uint blocks = a.latent / 32u;
    const ulong prow = (ulong)a.latent / 2u;
    for (uint g0 = 0; g0 < e.count; g0 += K3_MT_TG) {
        const uint ng = min((uint)K3_MT_TG, e.count - g0);
        float ag[K3_RPS][K3_MT_TG], au[K3_RPS][K3_MT_TG];
        for (uint k = 0; k < K3_RPS; k++) for (uint t = 0; t < K3_MT_TG; t++) { ag[k][t] = 0.0f; au[k][t] = 0.0f; }
        for (uint j = lane; j < blocks; j += 32u) {
            for (uint k = 0; k < K3_RPS; k++) {
                const uint r = r0 + k;
                if (r >= a.ff) break;
                const uint4 pg = *((device const uint4 *)(slab + a.w1p + (ulong)r * prow + (ulong)j * 16u));
                const uint4 pu = *((device const uint4 *)(slab + a.w3p + (ulong)r * prow + (ulong)j * 16u));
                const float dg = k3_scale(slab[a.w1s + (ulong)r * blocks + j]), du = k3_scale(slab[a.w3s + (ulong)r * blocks + j]);
                for (uint t = 0; t < K3_MT_TG; t++) {
                    if (t >= ng) break;
                    float4 xv[8];
                    device const float4 *x4 = (device const float4 *)(x + (ulong)ent[e.off + g0 + t].token * a.latent + (ulong)j * 32u);
                    for (uint q = 0; q < 8u; q++) xv[q] = x4[q];
                    ag[k][t] = fma(dg, k3_blk(pg, xv), ag[k][t]);
                    au[k][t] = fma(du, k3_blk(pu, xv), au[k][t]);
                }
            }
        }
        for (uint k = 0; k < K3_RPS; k++) {
            const uint r = r0 + k;
            if (r >= a.ff) break;
            for (uint t = 0; t < K3_MT_TG; t++) {
                const float gs = k3_simd_xor_sum(ag[k][t]), us = k3_simd_xor_sum(au[k][t]);   /* all lanes shuffle, also t >= ng */
                if (lane == 0u && t < ng) {
                    const k3_mt_entry en = ent[e.off + g0 + t];
                    mid[((ulong)en.token * 16u + en.slot) * a.ff + r] = k3_situ(gs, us, a.b, a.bl);
                }
            }
        }
    }
}
kernel void kernel_k3_moe_down_mt(constant k3_moe_args &a [[buffer(0)]], device const k3_mt_expert *ex [[buffer(1)]],
                                  device const k3_mt_entry *ent [[buffer(2)]], device const float *mid [[buffer(3)]],
                                  device float *part [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint c0 = (tg.x * 8u + warp) * K3_RPS;
    if (c0 >= a.latent) return;
    const k3_mt_expert e = ex[tg.y];
    device const uchar *slab = e.slab;
    const uint blocks = a.ff / 32u;
    const ulong prow = (ulong)a.ff / 2u;
    for (uint g0 = 0; g0 < e.count; g0 += K3_MT_TG) {
        const uint ng = min((uint)K3_MT_TG, e.count - g0);
        float ac[K3_RPS][K3_MT_TG];
        for (uint k = 0; k < K3_RPS; k++) for (uint t = 0; t < K3_MT_TG; t++) ac[k][t] = 0.0f;
        for (uint j = lane; j < blocks; j += 32u) {
            for (uint k = 0; k < K3_RPS; k++) {
                const uint c = c0 + k;
                if (c >= a.latent) break;
                const uint4 pk = *((device const uint4 *)(slab + a.w2p + (ulong)c * prow + (ulong)j * 16u));
                const float d = k3_scale(slab[a.w2s + (ulong)c * blocks + j]);
                for (uint t = 0; t < K3_MT_TG; t++) {
                    if (t >= ng) break;
                    const k3_mt_entry en = ent[e.off + g0 + t];
                    float4 xv[8];
                    device const float4 *x4 = (device const float4 *)(mid + ((ulong)en.token * 16u + en.slot) * a.ff + (ulong)j * 32u);
                    for (uint q = 0; q < 8u; q++) xv[q] = x4[q];
                    ac[k][t] = fma(d, k3_blk(pk, xv), ac[k][t]);
                }
            }
        }
        for (uint k = 0; k < K3_RPS; k++) {
            const uint c = c0 + k;
            if (c >= a.latent) break;
            for (uint t = 0; t < K3_MT_TG; t++) {
                const float v = k3_simd_xor_sum(ac[k][t]);
                if (lane == 0u && t < ng) {
                    const k3_mt_entry en = ent[e.off + g0 + t];
                    part[((ulong)en.token * 16u + en.slot) * a.latent + c] = en.weight * v;
                }
            }
        }
    }
}
// step M19: kernel_k3_moe_gate_up_mt / down_mt with each weight word decoded once per (block, row) for the group's tokens.
#define K3_FMA8(S, F, XA, XB) S = fma(F[0], XA.x, S); S = fma(F[1], XA.y, S); S = fma(F[2], XA.z, S); S = fma(F[3], XA.w, S); \
                              S = fma(F[4], XB.x, S); S = fma(F[5], XB.y, S); S = fma(F[6], XB.z, S); S = fma(F[7], XB.w, S);
template<uint RPS, uint TG>
static inline void k3_moe_gate_up_mt2_impl(constant k3_moe_args &a, device const k3_mt_expert *ex, device const k3_mt_entry *ent,
                                           device const float *x, device float *mid, uint2 tg, uint lane, uint warp) {
    const uint r0 = (tg.x * 8u + warp) * RPS;
    if (r0 >= a.ff) return;
    const k3_mt_expert e = ex[tg.y];
    device const uchar *slab = e.slab;
    const uint blocks = a.latent / 32u;
    const ulong prow = (ulong)a.latent / 2u;
    for (uint g0 = 0; g0 < e.count; g0 += TG) {
        const uint ng = min(TG, e.count - g0);
        device const float *xt[TG];
        for (uint t = 0; t < TG; t++) xt[t] = x + (ulong)ent[e.off + g0 + min(t, ng - 1u)].token * a.latent;
        float ag[RPS][TG], au[RPS][TG];
        for (uint k = 0; k < RPS; k++) for (uint t = 0; t < TG; t++) { ag[k][t] = 0.0f; au[k][t] = 0.0f; }
        for (uint j = lane; j < blocks; j += 32u) {
            for (uint k = 0; k < RPS; k++) {
                const uint r = r0 + k;
                if (r >= a.ff) break;
                const uint4 pg = *((device const uint4 *)(slab + a.w1p + (ulong)r * prow + (ulong)j * 16u));
                const uint4 pu = *((device const uint4 *)(slab + a.w3p + (ulong)r * prow + (ulong)j * 16u));
                const float dg = k3_scale(slab[a.w1s + (ulong)r * blocks + j]), du = k3_scale(slab[a.w3s + (ulong)r * blocks + j]);
                float sg[TG], su[TG];
                for (uint t = 0; t < TG; t++) { sg[t] = 0.0f; su[t] = 0.0f; }
                for (uint q = 0; q < 4u; q++) {
                    const uint wg = pg[q], wu = pu[q];
                    float fg[8], fu[8];
                    for (uint n = 0; n < 8u; n++) { fg[n] = K3_E2M1((wg >> (4u * n)) & 0x0fu); fu[n] = K3_E2M1((wu >> (4u * n)) & 0x0fu); }
                    for (uint t = 0; t < TG; t++) {
                        if (t >= ng) break;
                        device const float4 *x4 = (device const float4 *)(xt[t] + (ulong)j * 32u);
                        const float4 xa = x4[2u * q], xb = x4[2u * q + 1u];
                        float s1 = sg[t], s2 = su[t];
                        K3_FMA8(s1, fg, xa, xb)
                        K3_FMA8(s2, fu, xa, xb)
                        sg[t] = s1; su[t] = s2;
                    }
                }
                for (uint t = 0; t < TG; t++) {
                    if (t >= ng) break;
                    ag[k][t] = fma(dg, sg[t], ag[k][t]);
                    au[k][t] = fma(du, su[t], au[k][t]);
                }
            }
        }
        for (uint k = 0; k < RPS; k++) {
            const uint r = r0 + k;
            if (r >= a.ff) break;
            for (uint t = 0; t < TG; t++) {
                const float gs = k3_simd_xor_sum(ag[k][t]), us = k3_simd_xor_sum(au[k][t]);   /* all lanes shuffle, also t >= ng */
                if (lane == 0u && t < ng) {
                    const k3_mt_entry en = ent[e.off + g0 + t];
                    mid[((ulong)en.token * 16u + en.slot) * a.ff + r] = k3_situ(gs, us, a.b, a.bl);
                }
            }
        }
    }
}
template<uint RPS, uint TG>
static inline void k3_moe_down_mt2_impl(constant k3_moe_args &a, device const k3_mt_expert *ex, device const k3_mt_entry *ent,
                                        device const float *mid, device float *part, uint2 tg, uint lane, uint warp) {
    const uint c0 = (tg.x * 8u + warp) * RPS;
    if (c0 >= a.latent) return;
    const k3_mt_expert e = ex[tg.y];
    device const uchar *slab = e.slab;
    const uint blocks = a.ff / 32u;
    const ulong prow = (ulong)a.ff / 2u;
    for (uint g0 = 0; g0 < e.count; g0 += TG) {
        const uint ng = min(TG, e.count - g0);
        device const float *xt[TG];
        for (uint t = 0; t < TG; t++) {
            const k3_mt_entry en = ent[e.off + g0 + min(t, ng - 1u)];
            xt[t] = mid + ((ulong)en.token * 16u + en.slot) * a.ff;
        }
        float ac[RPS][TG];
        for (uint k = 0; k < RPS; k++) for (uint t = 0; t < TG; t++) ac[k][t] = 0.0f;
        for (uint j = lane; j < blocks; j += 32u) {
            for (uint k = 0; k < RPS; k++) {
                const uint c = c0 + k;
                if (c >= a.latent) break;
                const uint4 pk = *((device const uint4 *)(slab + a.w2p + (ulong)c * prow + (ulong)j * 16u));
                const float d = k3_scale(slab[a.w2s + (ulong)c * blocks + j]);
                float sk[TG];
                for (uint t = 0; t < TG; t++) sk[t] = 0.0f;
                for (uint q = 0; q < 4u; q++) {
                    const uint w = pk[q];
                    float f[8];
                    for (uint n = 0; n < 8u; n++) f[n] = K3_E2M1((w >> (4u * n)) & 0x0fu);
                    for (uint t = 0; t < TG; t++) {
                        if (t >= ng) break;
                        device const float4 *x4 = (device const float4 *)(xt[t] + (ulong)j * 32u);
                        const float4 xa = x4[2u * q], xb = x4[2u * q + 1u];
                        float s1 = sk[t];
                        K3_FMA8(s1, f, xa, xb)
                        sk[t] = s1;
                    }
                }
                for (uint t = 0; t < TG; t++) {
                    if (t >= ng) break;
                    ac[k][t] = fma(d, sk[t], ac[k][t]);
                }
            }
        }
        for (uint k = 0; k < RPS; k++) {
            const uint c = c0 + k;
            if (c >= a.latent) break;
            for (uint t = 0; t < TG; t++) {
                const float v = k3_simd_xor_sum(ac[k][t]);
                if (lane == 0u && t < ng) {
                    const k3_mt_entry en = ent[e.off + g0 + t];
                    part[((ulong)en.token * 16u + en.slot) * a.latent + c] = en.weight * v;
                }
            }
        }
    }
}
#define K3_MT2_KERNELS(RPS, TG) \
kernel void kernel_k3_moe_gate_up_mt2_r##RPS##_t##TG(constant k3_moe_args &a [[buffer(0)]], device const k3_mt_expert *ex [[buffer(1)]], \
        device const k3_mt_entry *ent [[buffer(2)]], device const float *x [[buffer(3)]], device float *mid [[buffer(4)]], \
        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) { \
    k3_moe_gate_up_mt2_impl<RPS, TG>(a, ex, ent, x, mid, tg, lane, warp); \
} \
kernel void kernel_k3_moe_down_mt2_r##RPS##_t##TG(constant k3_moe_args &a [[buffer(0)]], device const k3_mt_expert *ex [[buffer(1)]], \
        device const k3_mt_entry *ent [[buffer(2)]], device const float *mid [[buffer(3)]], device float *part [[buffer(4)]], \
        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) { \
    k3_moe_down_mt2_impl<RPS, TG>(a, ex, ent, mid, part, tg, lane, warp); \
}
K3_MT2_KERNELS(2, 4)
K3_MT2_KERNELS(4, 4)
K3_MT2_KERNELS(2, 8)
K3_MT2_KERNELS(4, 8)
K3_MT2_KERNELS(1, 4)

// step M21: activations once per (block, word, token) for all RPS rows; weights decoded once per (block, word, row) for the group.
template<uint RPS, uint TG>
static inline void k3_moe_gate_up_mt3_impl(constant k3_moe_args &a, device const k3_mt_expert *ex, device const k3_mt_entry *ent,
                                           device const float *x, device float *mid, uint2 tg, uint lane, uint warp) {
    const uint r0 = (tg.x * 8u + warp) * RPS;
    if (r0 >= a.ff) return;
    const uint nk = min(RPS, a.ff - r0);
    const k3_mt_expert e = ex[tg.y];
    device const uchar *slab = e.slab;
    const uint blocks = a.latent / 32u;
    const ulong prow = (ulong)a.latent / 2u;
    for (uint g0 = 0; g0 < e.count; g0 += TG) {
        const uint ng = min(TG, e.count - g0);
        device const float *xt[TG];
        for (uint t = 0; t < TG; t++) xt[t] = x + (ulong)ent[e.off + g0 + min(t, ng - 1u)].token * a.latent;
        float ag[RPS][TG], au[RPS][TG];
        for (uint k = 0; k < RPS; k++) for (uint t = 0; t < TG; t++) { ag[k][t] = 0.0f; au[k][t] = 0.0f; }
        for (uint j = lane; j < blocks; j += 32u) {
            float sg[RPS][TG], su[RPS][TG];
            for (uint k = 0; k < RPS; k++) for (uint t = 0; t < TG; t++) { sg[k][t] = 0.0f; su[k][t] = 0.0f; }
            for (uint q = 0; q < 4u; q++) {
                float4 xa[TG], xb[TG];
                for (uint t = 0; t < TG; t++) {
                    device const float4 *x4 = (device const float4 *)(xt[t] + (ulong)j * 32u);
                    xa[t] = x4[2u * q]; xb[t] = x4[2u * q + 1u];
                }
                for (uint k = 0; k < RPS; k++) {
                    if (k >= nk) break;
                    const uint r = r0 + k;
                    const uint wg = *((device const uint *)(slab + a.w1p + (ulong)r * prow + (ulong)j * 16u + 4u * q));
                    const uint wu = *((device const uint *)(slab + a.w3p + (ulong)r * prow + (ulong)j * 16u + 4u * q));
                    float fg[8], fu[8];
                    for (uint n = 0; n < 8u; n++) { fg[n] = K3_E2M1((wg >> (4u * n)) & 0x0fu); fu[n] = K3_E2M1((wu >> (4u * n)) & 0x0fu); }
                    for (uint t = 0; t < TG; t++) {
                        if (t >= ng) break;
                        float s1 = sg[k][t], s2 = su[k][t];
                        K3_FMA8(s1, fg, xa[t], xb[t])
                        K3_FMA8(s2, fu, xa[t], xb[t])
                        sg[k][t] = s1; su[k][t] = s2;
                    }
                }
            }
            for (uint k = 0; k < RPS; k++) {
                if (k >= nk) break;
                const uint r = r0 + k;
                const float dg = k3_scale(slab[a.w1s + (ulong)r * blocks + j]), du = k3_scale(slab[a.w3s + (ulong)r * blocks + j]);
                for (uint t = 0; t < TG; t++) {
                    if (t >= ng) break;
                    ag[k][t] = fma(dg, sg[k][t], ag[k][t]);
                    au[k][t] = fma(du, su[k][t], au[k][t]);
                }
            }
        }
        for (uint k = 0; k < RPS; k++) {
            if (k >= nk) break;
            const uint r = r0 + k;
            for (uint t = 0; t < TG; t++) {
                const float gs = k3_simd_xor_sum(ag[k][t]), us = k3_simd_xor_sum(au[k][t]);
                if (lane == 0u && t < ng) {
                    const k3_mt_entry en = ent[e.off + g0 + t];
                    mid[((ulong)en.token * 16u + en.slot) * a.ff + r] = k3_situ(gs, us, a.b, a.bl);
                }
            }
        }
    }
}
template<uint RPS, uint TG>
static inline void k3_moe_down_mt3_impl(constant k3_moe_args &a, device const k3_mt_expert *ex, device const k3_mt_entry *ent,
                                        device const float *mid, device float *part, uint2 tg, uint lane, uint warp) {
    const uint c0 = (tg.x * 8u + warp) * RPS;
    if (c0 >= a.latent) return;
    const uint nk = min(RPS, a.latent - c0);
    const k3_mt_expert e = ex[tg.y];
    device const uchar *slab = e.slab;
    const uint blocks = a.ff / 32u;
    const ulong prow = (ulong)a.ff / 2u;
    for (uint g0 = 0; g0 < e.count; g0 += TG) {
        const uint ng = min(TG, e.count - g0);
        device const float *xt[TG];
        for (uint t = 0; t < TG; t++) {
            const k3_mt_entry en = ent[e.off + g0 + min(t, ng - 1u)];
            xt[t] = mid + ((ulong)en.token * 16u + en.slot) * a.ff;
        }
        float ac[RPS][TG];
        for (uint k = 0; k < RPS; k++) for (uint t = 0; t < TG; t++) ac[k][t] = 0.0f;
        for (uint j = lane; j < blocks; j += 32u) {
            float sk[RPS][TG];
            for (uint k = 0; k < RPS; k++) for (uint t = 0; t < TG; t++) sk[k][t] = 0.0f;
            for (uint q = 0; q < 4u; q++) {
                float4 xa[TG], xb[TG];
                for (uint t = 0; t < TG; t++) {
                    device const float4 *x4 = (device const float4 *)(xt[t] + (ulong)j * 32u);
                    xa[t] = x4[2u * q]; xb[t] = x4[2u * q + 1u];
                }
                for (uint k = 0; k < RPS; k++) {
                    if (k >= nk) break;
                    const uint c = c0 + k;
                    const uint w = *((device const uint *)(slab + a.w2p + (ulong)c * prow + (ulong)j * 16u + 4u * q));
                    float f[8];
                    for (uint n = 0; n < 8u; n++) f[n] = K3_E2M1((w >> (4u * n)) & 0x0fu);
                    for (uint t = 0; t < TG; t++) {
                        if (t >= ng) break;
                        float s1 = sk[k][t];
                        K3_FMA8(s1, f, xa[t], xb[t])
                        sk[k][t] = s1;
                    }
                }
            }
            for (uint k = 0; k < RPS; k++) {
                if (k >= nk) break;
                const uint c = c0 + k;
                const float d = k3_scale(slab[a.w2s + (ulong)c * blocks + j]);
                for (uint t = 0; t < TG; t++) {
                    if (t >= ng) break;
                    ac[k][t] = fma(d, sk[k][t], ac[k][t]);
                }
            }
        }
        for (uint k = 0; k < RPS; k++) {
            if (k >= nk) break;
            const uint c = c0 + k;
            for (uint t = 0; t < TG; t++) {
                const float v = k3_simd_xor_sum(ac[k][t]);
                if (lane == 0u && t < ng) {
                    const k3_mt_entry en = ent[e.off + g0 + t];
                    part[((ulong)en.token * 16u + en.slot) * a.latent + c] = en.weight * v;
                }
            }
        }
    }
}
#define K3_MT3_KERNELS(RPS, TG) \
kernel void kernel_k3_moe_gate_up_mt3_r##RPS##_t##TG(constant k3_moe_args &a [[buffer(0)]], device const k3_mt_expert *ex [[buffer(1)]], \
        device const k3_mt_entry *ent [[buffer(2)]], device const float *x [[buffer(3)]], device float *mid [[buffer(4)]], \
        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) { \
    k3_moe_gate_up_mt3_impl<RPS, TG>(a, ex, ent, x, mid, tg, lane, warp); \
} \
kernel void kernel_k3_moe_down_mt3_r##RPS##_t##TG(constant k3_moe_args &a [[buffer(0)]], device const k3_mt_expert *ex [[buffer(1)]], \
        device const k3_mt_entry *ent [[buffer(2)]], device const float *mid [[buffer(3)]], device float *part [[buffer(4)]], \
        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]) { \
    k3_moe_down_mt3_impl<RPS, TG>(a, ex, ent, mid, part, tg, lane, warp); \
}
K3_MT3_KERNELS(2, 2)
K3_MT3_KERNELS(4, 2)
K3_MT3_KERNELS(2, 4)
K3_MT3_KERNELS(4, 4)
K3_MT3_KERNELS(8, 2)

// out[t][c] = sum over the token's slots in order (ascending expert id) of part[t][slot][c]: kernel_k3_moe_reduce per token
struct k3_mt_reduce_args { uint n_used; uint latent; };
kernel void kernel_k3_moe_reduce_mt(constant k3_mt_reduce_args &a [[buffer(0)]], device const float *part [[buffer(1)]], device float *out [[buffer(2)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, t = gid.y;
    if (c >= a.latent) return;
    float acc = 0.0f;
    for (uint i = 0; i < a.n_used; i++) acc += part[((ulong)t * 16u + i) * a.latent + c];
    out[(ulong)t * a.latent + c] = acc;
}

// ---- step M13: a Q8_0 weight slice dequantized to fp32 (exact: int8 x fp16 scale), rows r0 .. r0 + n_rows - 1 ----
struct k3_q8_deq_args { uint in_dim; uint blocks; uint n_rows; };
kernel void kernel_k3_q8_dequant_f32(constant k3_q8_deq_args &a [[buffer(0)]], device const uchar *w [[buffer(1)]], device float *out [[buffer(2)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, r = gid.y;
    if (i >= a.in_dim || r >= a.n_rows) return;
    device const uchar *blk = w + ((ulong)r * a.blocks + i / 32u) * 34u;
    const float d = (float)(*(device const half *)blk);
    out[(ulong)r * a.in_dim + i] = d * (float)(*(device const char *)(blk + 2u + (i % 32u)));
}

// step M16: the M15 kernel with R0 weight rows per threadgroup (template), RT activation rows per pass. An activation load serves
// R0 weight rows (M15: 2), so the activation traffic through the caches falls by R0/2. Each output's K walk (lane ix/il, simdgroup
// sgitg, blocks ib = sgitg*NQ + ix + k*NSG*NQ), its eight-term chain, the scale multiply and the reduction (simd_sum, threadgroup
// memory across simdgroups, simd_sum) are kernel_mul_mv_q8_0_f32_impl's with the same NSG.
template<short R0, short RT>
static inline void k3_mv_q8_rows_impl(constant ds4_metal_args_mul_mv & args, device const char * src0, device const char * src1,
                                      device char * dst, uint n_rows_x, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    const int nb = args.ne00/QK8_0;
    const int r0 = tgpig.x*R0;
    const int t0 = tgpig.y*RT;
    const int nt = min((int)RT, (int)n_rows_x - t0);
    device const block_q8_0 * ax[R0];
    FOR_UNROLL (short row = 0; row < R0; ++row) {
        const int rr = min(r0 + row, (int)args.ne01 - 1);   /* rows past the end read the last row; never written */
        ax[row] = (device const block_q8_0 *) (src0 + (uint64_t)rr*args.nb01);
    }
    float sumf[R0][RT];
    FOR_UNROLL (short row = 0; row < R0; ++row) {
        FOR_UNROLL (short t = 0; t < RT; ++t) { sumf[row][t] = 0.f; }
    }
    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);
    const int ib0 = sgitg*NQ + ix;
    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        FOR_UNROLL (short t = 0; t < RT; ++t) {
            if (t < nt) {
                device const float * yb = (device const float *) (src1 + (uint64_t)(t0 + t)*args.nb11) + ib*QK8_0 + il*NQ;
                float yl[NQ];
                for (short i = 0; i < NQ; ++i) {
                    yl[i] = yb[i];
                }
                FOR_UNROLL (short row = 0; row < R0; row++) {
                    device const int8_t * qs = ax[row][ib].qs + il*NQ;
                    float sumq = 0.f;
                    FOR_UNROLL (short i = 0; i < NQ; ++i) {
                        sumq += qs[i] * yl[i];
                    }
                    sumf[row][t] += sumq*ax[row][ib].d;
                }
            }
        }
    }
    threadgroup float * sh = (threadgroup float *) shmem;   // [R0][RT][NW]
    FOR_UNROLL (short row = 0; row < R0; ++row) {
        FOR_UNROLL (short t = 0; t < RT; ++t) {
            if (sgitg == 0) { sh[(row*RT + t)*NW + tiisg] = 0.0f; }
            sumf[row][t] = simd_sum(sumf[row][t]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (short row = 0; row < R0; ++row) {
        FOR_UNROLL (short t = 0; t < RT; ++t) {
            if (tiisg == 0) { sh[(row*RT + t)*NW + sgitg] = sumf[row][t]; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (short row = 0; row < R0; ++row) {
        FOR_UNROLL (short t = 0; t < RT; ++t) {
            if (r0 + row < args.ne01 && t < nt) {
                const float tot = simd_sum(sh[(row*RT + t)*NW + tiisg]);
                if (tiisg == 0 && sgitg == 0) { ((device float *) dst)[(uint64_t)(t0 + t)*args.ne0 + r0 + row] = tot; }
            }
        }
    }
}
template<short R0, short RT>
static inline void k3_mv_q8_rows2_impl(constant ds4_metal_args_mul_mv & args, device const char * src0, device const char * src1,
                                       device char * dst, uint n_rows_x, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    const int nb = args.ne00/QK8_0;
    const int r0 = tgpig.x*R0;
    const int t0 = tgpig.y*RT;
    const int nt = min((int)RT, (int)n_rows_x - t0);
    device const block_q8_0 * ax[R0];
    FOR_UNROLL (short row = 0; row < R0; ++row) {
        const int rr = min(r0 + row, (int)args.ne01 - 1);
        ax[row] = (device const block_q8_0 *) (src0 + (uint64_t)rr*args.nb01);
    }
    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);
    const int ib0 = sgitg*NQ + ix;
    device const float * yrow[RT];
    FOR_UNROLL (short t = 0; t < RT; ++t) {
        const int tt = t < nt ? t : 0;   /* rows past the end recompute row t0; never written */
        yrow[t] = (device const float *) (src1 + (uint64_t)(t0 + tt)*args.nb11) + il*NQ;
    }
    float sumf[R0][RT];
    FOR_UNROLL (short row = 0; row < R0; ++row) {
        FOR_UNROLL (short t = 0; t < RT; ++t) { sumf[row][t] = 0.f; }
    }
    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        float yl[RT][NQ];
        FOR_UNROLL (short t = 0; t < RT; ++t) {
            device const float4 * y4 = (device const float4 *) (yrow[t] + ib*QK8_0);
            const float4 ya = y4[0], yb = y4[1];
            yl[t][0] = ya.x; yl[t][1] = ya.y; yl[t][2] = ya.z; yl[t][3] = ya.w;
            yl[t][4] = yb.x; yl[t][5] = yb.y; yl[t][6] = yb.z; yl[t][7] = yb.w;
        }
        FOR_UNROLL (short row = 0; row < R0; ++row) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;
            float qf[NQ];
            FOR_UNROLL (short i = 0; i < NQ; ++i) { qf[i] = qs[i]; }
            const float d = ax[row][ib].d;
            FOR_UNROLL (short t = 0; t < RT; ++t) {
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < NQ; ++i) {
                    sumq += qf[i] * yl[t][i];
                }
                sumf[row][t] += sumq*d;
            }
        }
    }
    threadgroup float * sh = (threadgroup float *) shmem;   // [R0][RT][NW]
    FOR_UNROLL (short row = 0; row < R0; ++row) {
        FOR_UNROLL (short t = 0; t < RT; ++t) {
            if (sgitg == 0) { sh[(row*RT + t)*NW + tiisg] = 0.0f; }
            sumf[row][t] = simd_sum(sumf[row][t]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (short row = 0; row < R0; ++row) {
        FOR_UNROLL (short t = 0; t < RT; ++t) {
            if (tiisg == 0) { sh[(row*RT + t)*NW + sgitg] = sumf[row][t]; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (short row = 0; row < R0; ++row) {
        FOR_UNROLL (short t = 0; t < RT; ++t) {
            if (r0 + row < args.ne01 && t < nt) {
                const float tot = simd_sum(sh[(row*RT + t)*NW + tiisg]);
                if (tiisg == 0 && sgitg == 0) { ((device float *) dst)[(uint64_t)(t0 + t)*args.ne0 + r0 + row] = tot; }
            }
        }
    }
}
#define K3_ROWS2_KERNEL(R0, RT) \
kernel void kernel_k3_mv_q8_rows2_r##R0##_t##RT(constant ds4_metal_args_mul_mv & args, device const char * src0, device const char * src1, \
        device char * dst, constant uint & n_rows_x, threadgroup char * shmem [[threadgroup(0)]], uint3 tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    k3_mv_q8_rows2_impl<R0, RT>(args, src0, src1, dst, n_rows_x, shmem, tgpig, tiisg, sgitg); \
}
K3_ROWS2_KERNEL(2, 4)
K3_ROWS2_KERNEL(2, 8)
K3_ROWS2_KERNEL(4, 4)
K3_ROWS2_KERNEL(4, 8)

#define K3_ROWS_KERNEL(R0, RT) \
kernel void kernel_k3_mv_q8_rows_r##R0##_t##RT(constant ds4_metal_args_mul_mv & args, device const char * src0, device const char * src1, \
        device char * dst, constant uint & n_rows_x, threadgroup char * shmem [[threadgroup(0)]], uint3 tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    k3_mv_q8_rows_impl<R0, RT>(args, src0, src1, dst, n_rows_x, shmem, tgpig, tiisg, sgitg); \
}
K3_ROWS_KERNEL(4, 8)
K3_ROWS_KERNEL(8, 4)
K3_ROWS_KERNEL(8, 8)
K3_ROWS_KERNEL(16, 4)

// step M15: kernel_mul_mv_q8_0_f32_impl for up to K3_RT activation rows per weight load. For every (weight row, activation row) the
// K walk (ib from sgitg*NQ + ix in steps of NSG*NQ), the eight-term qs*yl sum, the multiply by the block scale and the reduction
// (simd_sum, then across simdgroups through threadgroup memory, then simd_sum) are the single-row kernel's.
#define K3_RT 8
kernel void kernel_k3_mul_mv_q8_0_f32_rows(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant uint     & n_rows_x,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NR0 = N_R0_Q8_0;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    const int nb = args.ne00/QK8_0;
    const int r0 = tgpig.x*NR0;
    const int t0 = tgpig.y*K3_RT;
    const int nt = min((int)K3_RT, (int)n_rows_x - t0);
    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q8_0 *) (src0 + (uint64_t)(r0 + row)*args.nb01);
    }
    float sumf[NR0][K3_RT];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        FOR_UNROLL (short t = 0; t < K3_RT; ++t) { sumf[row][t] = 0.f; }
    }
    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);
    const int ib0 = sgitg*NQ + ix;
    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        FOR_UNROLL (short t = 0; t < K3_RT; ++t) {
            if (t < nt) {
                device const float * yb = (device const float *) (src1 + (uint64_t)(t0 + t)*args.nb11) + ib*QK8_0 + il*NQ;
                float yl[NQ];
                for (short i = 0; i < NQ; ++i) {
                    yl[i] = yb[i];
                }
                for (short row = 0; row < NR0; row++) {
                    device const int8_t * qs = ax[row][ib].qs + il*NQ;
                    float sumq = 0.f;
                    FOR_UNROLL (short i = 0; i < NQ; ++i) {
                        sumq += qs[i] * yl[i];
                    }
                    sumf[row][t] += sumq*ax[row][ib].d;
                }
            }
        }
    }
    threadgroup float * sh = (threadgroup float *) shmem;   // [NR0][K3_RT][NW]
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        FOR_UNROLL (short t = 0; t < K3_RT; ++t) {
            if (sgitg == 0) { sh[(row*K3_RT + t)*NW + tiisg] = 0.0f; }
            sumf[row][t] = simd_sum(sumf[row][t]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        FOR_UNROLL (short t = 0; t < K3_RT; ++t) {
            if (tiisg == 0) { sh[(row*K3_RT + t)*NW + sgitg] = sumf[row][t]; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        FOR_UNROLL (short t = 0; t < K3_RT; ++t) {
            if (r0 + row < args.ne01 && t < nt) {
                const float tot = simd_sum(sh[(row*K3_RT + t)*NW + tiisg]);
                if (tiisg == 0 && sgitg == 0) { ((device float *) dst)[(uint64_t)(t0 + t)*args.ne0 + r0 + row] = tot; }
            }
        }
    }
}
