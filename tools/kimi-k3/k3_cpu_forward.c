/* =========================================================================
 * Kimi K3 CPU reference forward (2026-10-31).
 *
 * Sequential single-token decode over the prompt with carried KDA conv/recurrent
 * state and an MLA key/value cache, printing the top logits after every token.
 * It validates the K3 GGUF conversion, the raw-v1 expert source and the K3 math
 * (deltafin is the reference: same greedy continuation expected) before any GPU
 * graph work. Slow by design (scalar MXFP4 dots, single thread).
 * ========================================================================= */

typedef struct {
    float *conv_hist;    /* [3][projection] oldest..newest for q, k, v: 3 x 3 x projection */
    float *state;        /* [heads][dim][dim] recurrent S (k index major, v minor) */
} k3_cpu_kda_state;

typedef struct {
    float *k_nope;       /* [cap][heads][key_mla] */
    float *v;            /* [cap][heads][value_mla] */
    float *k_rope;       /* [cap][rot] */
    uint32_t len;
    uint32_t cap;
} k3_cpu_mla_cache;

static float k3_cpu_situ(float gate, float up) {
    const float b = g_ds4_shape.situ_beta, bl = g_ds4_shape.situ_linear_beta;
    const float bounded_gate = b * tanhf(gate / b) * sigmoid_stable(gate);
    const float bounded_up = bl * tanhf(up / bl);
    return bounded_gate * bounded_up;
}

/* mix(x, anchors, w): values = [anchors..., x]; keys = rmsnorm(values) (no weight); score_i = keys_i . w;
 * p = softmax(scores); out = sum p_i values_i. */
static void k3_cpu_residual_mix(float *out, const float *x, const float *anchors, uint32_t n_anchor,
                                const float *res_norm, const float *res_proj) {
    const uint32_t n = DS4_N_EMBD;
    const uint32_t count = n_anchor + 1u;
    float scores[16];
    float *keys = xmalloc((size_t)n * sizeof(float));
    for (uint32_t i = 0; i < count; i++) {
        const float *v = i < n_anchor ? anchors + (uint64_t)i * n : x;
        rms_norm_no_weight(keys, v, n, DS4_RMS_EPS);
        double s = 0.0;
        for (uint32_t j = 0; j < n; j++) s += (double)keys[j] * res_norm[j] * res_proj[j];
        scores[i] = (float)s;
    }
    float mx = scores[0];
    for (uint32_t i = 1; i < count; i++) if (scores[i] > mx) mx = scores[i];
    double denom = 0.0;
    for (uint32_t i = 0; i < count; i++) { scores[i] = expf(scores[i] - mx); denom += scores[i]; }
    for (uint32_t j = 0; j < n; j++) out[j] = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        const float p = (float)(scores[i] / denom);
        const float *v = i < n_anchor ? anchors + (uint64_t)i * n : x;
        for (uint32_t j = 0; j < n; j++) out[j] += p * v[j];
    }
    free(keys);
}

static void k3_cpu_kda_one(float *out, const ds4_model *model, const ds4_layer_weights *l,
                           const float *normed, k3_cpu_kda_state *st) {
    const uint32_t heads = DS4_N_KDA_HEAD, dim = DS4_N_KDA_HEAD_DIM, proj = heads * dim;
    float *q = xmalloc(proj * sizeof(float)), *k = xmalloc(proj * sizeof(float)), *v = xmalloc(proj * sizeof(float));
    float *f_a = xmalloc(dim * sizeof(float)), *raw_gate = xmalloc(proj * sizeof(float));
    float *beta = xmalloc(heads * sizeof(float)), *ogate = xmalloc(proj * sizeof(float));
    float *o = xmalloc(proj * sizeof(float));
    matvec_q8_0(q, model, l->kda_q, normed);
    matvec_q8_0(k, model, l->kda_k, normed);
    matvec_q8_0(v, model, l->kda_v, normed);
    matvec_q8_0(f_a, model, l->kda_f_a, normed);
    matvec_q8_0(raw_gate, model, l->kda_f_b, f_a);
    matvec_q8_0(beta, model, l->kda_beta, normed);
    matvec_q8_0(ogate, model, l->kda_gate, normed);
    const float *qc = tensor_data(model, l->kda_q_conv), *kc = tensor_data(model, l->kda_k_conv), *vc = tensor_data(model, l->kda_v_conv);
    const float *a_log = tensor_data(model, l->kda_a_log), *dt_bias = tensor_data(model, l->kda_dt_bias), *o_norm = tensor_data(model, l->kda_o_norm);
    /* short causal conv, width 4, history = 3 oldest..newest, then silu */
    float *inputs[3] = {q, k, v};
    const float *convs[3] = {qc, kc, vc};
    for (int s = 0; s < 3; s++) {
        float *hist = st->conv_hist + (uint64_t)s * 3u * proj;
        for (uint32_t c = 0; c < proj; c++) {
            const float *w = convs[s] + (uint64_t)c * 4u;
            float acc = hist[c] * w[0] + hist[proj + c] * w[1] + hist[2u * proj + c] * w[2] + inputs[s][c] * w[3];
            hist[c] = hist[proj + c]; hist[proj + c] = hist[2u * proj + c]; hist[2u * proj + c] = inputs[s][c];
            inputs[s][c] = acc / (1.0f + expf(-acc));   /* silu */
        }
    }
    const float qscale = 1.0f / sqrtf((float)dim);
    for (uint32_t h = 0; h < heads; h++) {
        double qs = 0.0, ks = 0.0;
        for (uint32_t c = 0; c < dim; c++) { qs += (double)q[h * dim + c] * q[h * dim + c]; ks += (double)k[h * dim + c] * k[h * dim + c]; }
        const float qn = (float)(1.0 / fmax(sqrt(qs), 1.0e-6)), kn = (float)(1.0 / fmax(sqrt(ks), 1.0e-6));
        for (uint32_t c = 0; c < dim; c++) { q[h * dim + c] *= qn * qscale; k[h * dim + c] *= kn; }
        const float b = sigmoid_stable(beta[h]);
        float *S = st->state + (uint64_t)h * dim * dim;
        float decay[128];
        for (uint32_t c = 0; c < dim; c++) {
            const float g = raw_gate[h * dim + c] + dt_bias[h * dim + c];
            const float a = g_ds4_shape.kda_a_log_per_channel ? a_log[c] : a_log[h];
            decay[c] = expf(DS4_KDA_GATE_LOWER_BOUND * sigmoid_stable(expf(a) * g));
        }
        for (uint32_t c = 0; c < dim; c++) for (uint32_t j = 0; j < dim; j++) S[c * dim + j] *= decay[c];
        float delta[128];
        for (uint32_t j = 0; j < dim; j++) {
            double hk = 0.0;
            for (uint32_t c = 0; c < dim; c++) hk += (double)k[h * dim + c] * S[c * dim + j];
            delta[j] = (v[h * dim + j] - (float)hk) * b;
        }
        for (uint32_t c = 0; c < dim; c++) for (uint32_t j = 0; j < dim; j++) S[c * dim + j] += k[h * dim + c] * delta[j];
        double os = 0.0;
        for (uint32_t j = 0; j < dim; j++) {
            double acc = 0.0;
            for (uint32_t c = 0; c < dim; c++) acc += (double)q[h * dim + c] * S[c * dim + j];
            o[h * dim + j] = (float)acc; os += acc * acc;
        }
        const float onorm = (float)(1.0 / sqrt(os / dim + DS4_RMS_EPS));
        for (uint32_t j = 0; j < dim; j++) o[h * dim + j] = o[h * dim + j] * onorm * o_norm[j] * sigmoid_stable(ogate[h * dim + j]);
    }
    matvec_q8_0(out, model, l->kda_output, o);
    free(o); free(ogate); free(beta); free(raw_gate); free(f_a); free(v); free(k); free(q);
}

static void k3_cpu_mla_one(float *out, const ds4_model *model, const ds4_layer_weights *l,
                           const float *normed, k3_cpu_mla_cache *cache) {
    const uint32_t heads = DS4_N_HEAD, nope = DS4_N_KEY_MLA, rot = DS4_N_ROT, vdim = DS4_N_VALUE_MLA;
    const uint32_t qdim = nope + rot, kv_lora = DS4_N_KV_LORA;
    float *q_a = xmalloc(DS4_N_LORA_Q * sizeof(float)), *q_an = xmalloc(DS4_N_LORA_Q * sizeof(float));
    float *q = xmalloc((size_t)heads * qdim * sizeof(float));
    float *kv = xmalloc((size_t)(kv_lora + rot) * sizeof(float)), *latent = xmalloc(kv_lora * sizeof(float));
    float *kvb = xmalloc((size_t)heads * (nope + vdim) * sizeof(float));
    float *attn = xmalloc((size_t)heads * vdim * sizeof(float)), *gate = xmalloc((size_t)heads * vdim * sizeof(float));
    matvec_q8_0(q_a, model, l->attn_q_a, normed);
    rms_norm_weight(q_an, q_a, tensor_data(model, l->attn_q_a_norm), DS4_N_LORA_Q, DS4_RMS_EPS);
    matvec_q8_0(q, model, l->attn_q_b, q_an);
    matvec_q8_0(kv, model, l->attn_kv_a_mqa, normed);
    rms_norm_weight(latent, kv, tensor_data(model, l->attn_kv_a_norm), kv_lora, DS4_RMS_EPS);
    matvec_q8_0(kvb, model, l->attn_kv_b, latent);
    if (cache->len >= cache->cap) ds4_die("K3 CPU MLA cache is full");
    const uint32_t p = cache->len++;
    for (uint32_t h = 0; h < heads; h++) {
        memcpy(cache->k_nope + ((uint64_t)p * heads + h) * nope, kvb + (uint64_t)h * (nope + vdim), nope * sizeof(float));
        memcpy(cache->v + ((uint64_t)p * heads + h) * vdim, kvb + (uint64_t)h * (nope + vdim) + nope, vdim * sizeof(float));
    }
    memcpy(cache->k_rope + (uint64_t)p * rot, kv + kv_lora, rot * sizeof(float));
    const float scale = 1.0f / sqrtf((float)qdim);
    float *scores = xmalloc((size_t)cache->len * sizeof(float));
    for (uint32_t h = 0; h < heads; h++) {
        const float *qh = q + (uint64_t)h * qdim;
        float mx = -1e30f;
        for (uint32_t t = 0; t < cache->len; t++) {
            double s = 0.0;
            const float *kn = cache->k_nope + ((uint64_t)t * heads + h) * nope;
            const float *kr = cache->k_rope + (uint64_t)t * rot;
            for (uint32_t i = 0; i < nope; i++) s += (double)qh[i] * kn[i];
            for (uint32_t i = 0; i < rot; i++) s += (double)qh[nope + i] * kr[i];
            scores[t] = (float)s * scale;
            if (scores[t] > mx) mx = scores[t];
        }
        double denom = 0.0;
        for (uint32_t t = 0; t < cache->len; t++) { scores[t] = expf(scores[t] - mx); denom += scores[t]; }
        for (uint32_t i = 0; i < vdim; i++) attn[h * vdim + i] = 0.0f;
        for (uint32_t t = 0; t < cache->len; t++) {
            const float w = (float)(scores[t] / denom);
            const float *vt = cache->v + ((uint64_t)t * heads + h) * vdim;
            for (uint32_t i = 0; i < vdim; i++) attn[h * vdim + i] += w * vt[i];
        }
    }
    free(scores);
    matvec_q8_0(gate, model, l->attn_gate, normed);
    for (uint32_t i = 0; i < heads * vdim; i++) attn[i] *= sigmoid_stable(gate[i]);
    matvec_q8_0(out, model, l->attn_output, attn);
    free(gate); free(attn); free(kvb); free(latent); free(kv); free(q); free(q_an); free(q_a);
}

static void k3_cpu_moe_one(float *out, const ds4_model *model, const ds4_layer_weights *l, const float *normed) {
    const uint32_t n_expert = DS4_N_EXPERT, used = DS4_N_EXPERT_USED, ff = DS4_N_FF_EXP;
    const uint32_t latent_dim = g_ds4_shape.n_routed_latent;
    float *logits = xmalloc(n_expert * sizeof(float));
    matvec_f32(logits, model, l->ffn_gate_inp, normed);
    const float *bias = tensor_data(model, l->ffn_exp_probs_b);
    float *scores = xmalloc(n_expert * sizeof(float)), *choice = xmalloc(n_expert * sizeof(float));
    for (uint32_t e = 0; e < n_expert; e++) { scores[e] = sigmoid_stable(logits[e]); choice[e] = scores[e] + bias[e]; }
    int selected[16]; float weight[16]; double wsum = 0.0;
    for (uint32_t i = 0; i < used; i++) {
        int best = -1;
        for (uint32_t e = 0; e < n_expert; e++) {
            bool taken = false;
            for (uint32_t j = 0; j < i; j++) if (selected[j] == (int)e) { taken = true; break; }
            if (!taken && (best < 0 || choice[e] > choice[best])) best = (int)e;
        }
        selected[i] = best; weight[i] = scores[best]; wsum += scores[best];
    }
    for (uint32_t i = 0; i < used; i++) weight[i] = (float)(weight[i] / (wsum + 1.0e-20));
    float *latent = xmalloc(latent_dim * sizeof(float));
    matvec_q8_0(latent, model, l->moe_latent_down, normed);
    float *routed = xmalloc(latent_dim * sizeof(float));
    memset(routed, 0, latent_dim * sizeof(float));
    const uint64_t gu_bytes = (uint64_t)ff * (latent_dim / 32u) * 17u;
    const uint64_t d_bytes = (uint64_t)latent_dim * (ff / 32u) * 17u;
    uint8_t *gbuf = xmalloc(gu_bytes), *ubuf = xmalloc(gu_bytes), *dbuf = xmalloc(d_bytes);
    float *mid = xmalloc(ff * sizeof(float));
    for (uint32_t i = 0; i < used; i++) {
        const uint32_t e = (uint32_t)selected[i];
        if (!ds4_k3_expert_source_read(l->ffn_gate_exps->abs_offset + (uint64_t)e * gu_bytes, gu_bytes, gbuf) ||
            !ds4_k3_expert_source_read(l->ffn_up_exps->abs_offset + (uint64_t)e * gu_bytes, gu_bytes, ubuf) ||
            !ds4_k3_expert_source_read(l->ffn_down_exps->abs_offset + (uint64_t)e * d_bytes, d_bytes, dbuf)) {
            ds4_die("K3 CPU reference: expert source read failed");
        }
        const uint32_t gu_row = (latent_dim / 32u) * 17u, d_row = (ff / 32u) * 17u;
        for (uint32_t r = 0; r < ff; r++) {
            const float g = ds4_vec_dot_mxfp4_f32((int)latent_dim, (const block_mxfp4 *)(gbuf + (uint64_t)r * gu_row), latent);
            const float u = ds4_vec_dot_mxfp4_f32((int)latent_dim, (const block_mxfp4 *)(ubuf + (uint64_t)r * gu_row), latent);
            mid[r] = k3_cpu_situ(g, u);
        }
        for (uint32_t c = 0; c < latent_dim; c++) {
            routed[c] += weight[i] * ds4_vec_dot_mxfp4_f32((int)ff, (const block_mxfp4 *)(dbuf + (uint64_t)c * d_row), mid);
        }
    }
    float *routed_norm = xmalloc(latent_dim * sizeof(float));
    rms_norm_weight(routed_norm, routed, tensor_data(model, l->moe_latent_norm), latent_dim, DS4_RMS_EPS);
    matvec_q8_0(out, model, l->moe_latent_up, routed_norm);
    /* shared experts (fused 6144 wide) */
    const uint32_t sh = (uint32_t)l->ffn_gate_shexp->dim[1];
    float *sg = xmalloc(sh * sizeof(float)), *su = xmalloc(sh * sizeof(float)), *sd = xmalloc(DS4_N_EMBD * sizeof(float));
    matvec_q8_0(sg, model, l->ffn_gate_shexp, normed);
    matvec_q8_0(su, model, l->ffn_up_shexp, normed);
    for (uint32_t i = 0; i < sh; i++) sg[i] = k3_cpu_situ(sg[i], su[i]);
    matvec_q8_0(sd, model, l->ffn_down_shexp, sg);
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) out[i] += sd[i];
    free(sd); free(su); free(sg); free(routed_norm); free(mid); free(dbuf); free(ubuf); free(gbuf);
    free(routed); free(latent); free(choice); free(scores); free(logits);
}

static void k3_cpu_dense_one(float *out, const ds4_model *model, const ds4_layer_weights *l, const float *normed) {
    const uint32_t ffd = DS4_N_FF_DENSE;
    float *g = xmalloc(ffd * sizeof(float)), *u = xmalloc(ffd * sizeof(float));
    matvec_q8_0(g, model, l->ffn_gate, normed);
    matvec_q8_0(u, model, l->ffn_up, normed);
    for (uint32_t i = 0; i < ffd; i++) g[i] = k3_cpu_situ(g[i], u[i]);
    matvec_q8_0(out, model, l->ffn_down, g);
    free(u); free(g);
}

static void k3_cpu_print_top(const ds4_vocab *vocab, const float *logits, int top) {
    int best[8];
    for (int i = 0; i < top; i++) best[i] = -1;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        for (int j = 0; j < top; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = top - 1; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }
    for (int i = 0; i < top; i++) {
        printf("  %6d  %9.4f  %.*s\n", best[i], logits[best[i]], (int)vocab->token[best[i]].len, vocab->token[best[i]].ptr);
    }
}

/* Sequential CPU decode over the prompt ids (DS4_K3_PROMPT_IDS="a,b,c" overrides the tokenised prompt). */
static int k3_cpu_reference_test(const ds4_model *model, const ds4_vocab *vocab, const ds4_weights *weights,
                                 const ds4_tokens *prompt) {
    int ids[256]; int n_ids = 0;
    const char *env = getenv("DS4_K3_PROMPT_IDS");
    if (env && env[0]) {
        const char *p = env;
        while (*p && n_ids < 256) {
            char *end = NULL; long v = strtol(p, &end, 10);
            if (end == p) break;
            ids[n_ids++] = (int)v; p = end; while (*p == ',' || *p == ' ') p++;
        }
    } else {
        for (int i = 0; i < prompt->len && n_ids < 256; i++) ids[n_ids++] = prompt->v[i];
    }
    if (n_ids == 0) { fprintf(stderr, "ds4: K3 CPU reference needs prompt ids\n"); return 1; }
    const uint32_t n = DS4_N_EMBD, L = DS4_N_LAYER, block = g_ds4_shape.n_res_block;
    const uint32_t heads = DS4_N_KDA_HEAD, dim = DS4_N_KDA_HEAD_DIM, proj = heads * dim;
    k3_cpu_kda_state *kda = xcalloc(L, sizeof(*kda));
    k3_cpu_mla_cache *mla = xcalloc(L, sizeof(*mla));
    for (uint32_t il = 0; il < L; il++) {
        if (ds4_k3_layer_is_kda(il)) {
            kda[il].conv_hist = xcalloc((size_t)3 * 3 * proj, sizeof(float));
            kda[il].state = xcalloc((size_t)heads * dim * dim, sizeof(float));
        } else {
            mla[il].cap = (uint32_t)n_ids;
            mla[il].k_nope = xcalloc((size_t)n_ids * DS4_N_HEAD * DS4_N_KEY_MLA, sizeof(float));
            mla[il].v = xcalloc((size_t)n_ids * DS4_N_HEAD * DS4_N_VALUE_MLA, sizeof(float));
            mla[il].k_rope = xcalloc((size_t)n_ids * DS4_N_ROT, sizeof(float));
        }
    }
    float *cur = xmalloc(n * sizeof(float)), *attn_in = xmalloc(n * sizeof(float)), *normed = xmalloc(n * sizeof(float));
    float *attn_out = xmalloc(n * sizeof(float)), *prefix = xmalloc(n * sizeof(float)), *mixed = xmalloc(n * sizeof(float));
    float *ffn_out = xmalloc(n * sizeof(float)), *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *anchors = xmalloc((size_t)16 * n * sizeof(float));
    for (int t = 0; t < n_ids; t++) {
        const double t0 = now_sec();
        uint32_t n_anchor = 0;
        embed_token_any(model, weights, ids[t], cur);
        for (uint32_t il = 0; il < L; il++) {
            const ds4_layer_weights *l = &weights->layer[il];
            const float *res_attn_norm = tensor_data(model, l->res_attn_norm), *res_attn_proj = tensor_data(model, l->res_attn_proj);
            const float *res_ffn_norm = tensor_data(model, l->res_ffn_norm), *res_ffn_proj = tensor_data(model, l->res_ffn_proj);
            if (n_anchor != 0) k3_cpu_residual_mix(attn_in, cur, anchors, n_anchor, res_attn_norm, res_attn_proj);
            else memcpy(attn_in, cur, n * sizeof(float));
            bool have_prefix = true;
            if (il % block == 0) {
                if (n_anchor >= 16) ds4_die("K3 residual anchors overflow");
                memcpy(anchors + (uint64_t)n_anchor * n, cur, n * sizeof(float)); n_anchor++;
                have_prefix = false;
            }
            rms_norm_weight(normed, attn_in, tensor_data(model, l->attn_norm), n, DS4_RMS_EPS);
            if (ds4_k3_layer_is_kda(il)) k3_cpu_kda_one(attn_out, model, l, normed, &kda[il]);
            else k3_cpu_mla_one(attn_out, model, l, normed, &mla[il]);
            for (uint32_t i = 0; i < n; i++) prefix[i] = have_prefix ? cur[i] + attn_out[i] : attn_out[i];
            k3_cpu_residual_mix(mixed, prefix, anchors, n_anchor, res_ffn_norm, res_ffn_proj);
            rms_norm_weight(normed, mixed, tensor_data(model, l->ffn_norm), n, DS4_RMS_EPS);
            if (il < DS4_N_LEADING_DENSE) k3_cpu_dense_one(ffn_out, model, l, normed);
            else k3_cpu_moe_one(ffn_out, model, l, normed);
            for (uint32_t i = 0; i < n; i++) cur[i] = prefix[i] + ffn_out[i];
            if (getenv("DS4_K3_TRACE_LAYERS")) {
                char name[64]; snprintf(name, sizeof(name), "token %d layer %u hidden", t, il);
                print_vec_stats(name, cur, n);
            }
        }
        k3_cpu_residual_mix(mixed, cur, anchors, n_anchor, tensor_data(model, weights->output_res_norm),
                            tensor_data(model, weights->output_res_proj));
        rms_norm_weight(normed, mixed, tensor_data(model, weights->output_norm), n, DS4_RMS_EPS);
        matvec_q8_0(logits, model, weights->output, normed);
        printf("K3 CPU reference: token %d/%d id %d (%.1f s) top logits:\n", t + 1, n_ids, ids[t], now_sec() - t0);
        k3_cpu_print_top(vocab, logits, 8);
        fflush(stdout);
    }
    for (uint32_t il = 0; il < L; il++) {
        free(kda[il].conv_hist); free(kda[il].state); free(mla[il].k_nope); free(mla[il].v); free(mla[il].k_rope);
    }
    free(kda); free(mla); free(anchors); free(logits); free(ffn_out); free(mixed); free(prefix); free(attn_out);
    free(normed); free(attn_in); free(cur);
    return 0;
}

