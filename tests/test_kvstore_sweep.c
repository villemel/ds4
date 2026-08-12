/* Standalone test for ds4_kvstore_sweep().
 *
 * Crafts minimal .kv checkpoint files with the real header + payload layout
 * and verifies the sweeper keeps only the maximal checkpoint plus one
 * predecessor per lineage, dropping dominated checkpoints and exact
 * duplicates while preserving divergent branches, other conversations, and
 * cross-quant entries.
 *
 * The test links only ds4_kvstore.o.  The ds4_tokens_* helpers are stubbed
 * here so the heavy GPU-backed ds4.o is not required; -ffunction-sections /
 * -dead_strip drops the unreferenced ds4_engine/session symbols.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include "ds4_kvstore.h"
#include "ds4.h"

/* --- stubs for ds4_tokens_* (test-only) --- */
void ds4_tokens_free(ds4_tokens *tv) {
    free(tv->v);
    tv->v = NULL; tv->len = 0; tv->cap = 0;
}
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src) {
    dst->len = 0;
    for (int i = 0; i < src->len; i++) {
        if (dst->len == dst->cap) {
            dst->cap = dst->cap ? dst->cap * 2 : 16;
            dst->v = realloc(dst->v, (size_t)dst->cap * sizeof(int));
        }
        dst->v[dst->len++] = src->v[i];
    }
}
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix) {
    if (prefix->len > tokens->len) return false;
    return memcmp(tokens->v, prefix->v, (size_t)prefix->len * sizeof(int)) == 0;
}
void ds4_tokens_push(ds4_tokens *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 16;
        tv->v = realloc(tv->v, (size_t)tv->cap * sizeof(int));
    }
    tv->v[tv->len++] = token;
}

/* --- helpers --- */
static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); failures++; } \
    else { printf("ok: %s\n", msg); } \
} while (0)

static void le32(uint8_t *b, uint32_t v) {
    b[0]=v&0xff; b[1]=(v>>8)&0xff; b[2]=(v>>16)&0xff; b[3]=(v>>24)&0xff;
}

static void write_checkpoint(const char *path, int quant, int reason,
                             const int *tok, int ntok) {
    FILE *fp = fopen(path, "wb");
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    uint64_t now = 1700000000ull;
    uint32_t payload_bytes = 13u*4u + (uint32_t)ntok*4u;
    ds4_kvstore_fill_header(h, 0, (uint8_t)quant, (uint8_t)reason, 0,
                            (uint32_t)ntok, 0, (uint32_t)1000000, now, now,
                            payload_bytes);
    fwrite(h, 1, sizeof(h), fp);
    uint8_t tb[4]; le32(tb, 0);
    fwrite(tb, 1, 4, fp);
    uint32_t ph[13] = {0};
    ph[0] = DS4_SESSION_PAYLOAD_MAGIC;
    ph[1] = DS4_SESSION_PAYLOAD_VERSION;
    ph[7] = (uint32_t)ntok;
    for (int i = 0; i < 13; i++) { uint8_t b[4]; le32(b, ph[i]); fwrite(b,1,4,fp); }
    for (int i = 0; i < ntok; i++) { uint8_t b[4]; le32(b, (uint32_t)tok[i]); fwrite(b,1,4,fp); }
    fclose(fp);
}

static void sha_name(char *out, int n) { snprintf(out, 41, "%040d", n); }

static int contains(const ds4_kvstore *kc, const char *name) {
    for (int i = 0; i < kc->len; i++)
        if (strstr(kc->entry[i].path, name)) return 1;
    return 0;
}

int main(void) {
    char dir[256];
    snprintf(dir, sizeof(dir), "/tmp/kvsweep_%d", (int)getpid());
    mkdir(dir, 0755);

    char p[300];
    int A[] = {10,20,30};               /* cold, dominated */
    int C[] = {10,20,30,40};            /* continued, predecessor */
    int B[] = {10,20,30,40,50,60};      /* continued, max */
    int D[] = {10,20,30,40,50,60};      /* exact duplicate of B */
    int E[] = {10,20,99};               /* divergent branch */
    int F[] = {5,6};                    /* other conversation */
    int G[] = {10,20,30,40,50,60};      /* same tokens, different quant */

    char nA[41], nB[41], nC[41], nD[41], nE[41], nF[41], nG[41];
    sha_name(nA,1); sha_name(nB,2); sha_name(nC,3); sha_name(nD,4);
    sha_name(nE,5); sha_name(nF,6); sha_name(nG,7);

    snprintf(p,sizeof(p),"%s/%s.kv",dir,nA); write_checkpoint(p,4,1,A,3);
    snprintf(p,sizeof(p),"%s/%s.kv",dir,nB); write_checkpoint(p,4,2,B,6);
    snprintf(p,sizeof(p),"%s/%s.kv",dir,nC); write_checkpoint(p,4,2,C,4);
    snprintf(p,sizeof(p),"%s/%s.kv",dir,nD); write_checkpoint(p,4,3,D,6);
    snprintf(p,sizeof(p),"%s/%s.kv",dir,nE); write_checkpoint(p,4,2,E,3);
    snprintf(p,sizeof(p),"%s/%s.kv",dir,nF); write_checkpoint(p,4,1,F,2);
    snprintf(p,sizeof(p),"%s/%s.kv",dir,nG); write_checkpoint(p,2,2,G,6);

    ds4_kvstore kc;
    memset(&kc, 0, sizeof(kc));
    ds4_kvstore_options opt = { .min_tokens=1, .cold_max_tokens=100000,
                                .continued_interval_tokens=1000,
                                .boundary_trim_tokens=0, .boundary_align_tokens=0 };
    CHECK(ds4_kvstore_open(&kc, dir, 4096, true, opt, "test", NULL, NULL), "open");
    /* open performs a one-time sweep, so redundant files are already gone.
     * Kept: B(max), C(predecessor), E(divergent branch), F(other conv),
     * G(diff quant) = 5. */
    CHECK(kc.len == 5, "5 entries after open+sweep (B,C,E,F,G)");

    kc.last_sweep_ts = 0; /* bypass throttle */
    ds4_kvstore_sweep(&kc);
    CHECK(kc.len == 5, "5 entries after explicit sweep (B,C,E,F,G)");

    CHECK(contains(&kc, nB), "kept max B");
    CHECK(contains(&kc, nC), "kept predecessor C");
    CHECK(!contains(&kc, nA), "dropped dominated cold A");
    CHECK(!contains(&kc, nD), "dropped duplicate D");
    CHECK(contains(&kc, nE), "kept divergent branch E");
    CHECK(contains(&kc, nF), "kept other conversation F");
    CHECK(contains(&kc, nG), "kept different-quant G (not dominated across quant)");

    struct stat st;
    snprintf(p,sizeof(p),"%s/%s.kv",dir,nA); CHECK(stat(p,&st)!=0, "A removed on disk");
    snprintf(p,sizeof(p),"%s/%s.kv",dir,nD); CHECK(stat(p,&st)!=0, "D removed on disk");
    snprintf(p,sizeof(p),"%s/%s.kv",dir,nB); CHECK(stat(p,&st)==0, "B still on disk");
    snprintf(p,sizeof(p),"%s/%s.kv",dir,nC); CHECK(stat(p,&st)==0, "C still on disk");

    ds4_kvstore_close(&kc);
    char cmd[256]; snprintf(cmd,sizeof(cmd),"rm -rf %s",dir); system(cmd);
    printf(failures ? "\n%d FAILURES\n" : "\nALL PASS\n", failures);
    return failures ? 1 : 0;
}
