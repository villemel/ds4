#define _GNU_SOURCE
#include "ds4_kimi_k3.h"
#include <time.h>
#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __APPLE__
/* macOS has no O_DIRECT and no posix_fadvise: a private flag asks for F_NOCACHE after open (uncached reads, no alignment rules) */
#define K3_O_DIRECT_MAC 0x40000000
static int k3_open_mac(const char *path, int flags) {
    const int fd = open(path, flags & ~K3_O_DIRECT_MAC);
    if (fd >= 0 && (flags & K3_O_DIRECT_MAC)) (void)fcntl(fd, F_NOCACHE, 1);
    return fd;
}
#define O_DIRECT K3_O_DIRECT_MAC
#define open(p, f) k3_open_mac((p), (f))
#define posix_fadvise(fd, off, len, adv) 0
#define POSIX_FADV_DONTNEED 0
#endif

#define K3_MAX_COMPONENTS (93u * 3u)
#define K3_BLOCK 32u
#define K3_BLOCK_BYTES 17u

/* raw-v1 plane offsets inside one expert file (w1 = gate, w3 = up: [3072 x 3584]; w2 = down: [3584 x 3072]) */
static const uint64_t K3_W1_PACKED = 0ull;
static const uint64_t K3_W1_SCALE  = 5505024ull;
static const uint64_t K3_W2_PACKED = 5849088ull;
static const uint64_t K3_W2_SCALE  = 11354112ull;
static const uint64_t K3_W3_PACKED = 11698176ull;
static const uint64_t K3_W3_SCALE  = 17203200ull;

typedef struct {
    uint32_t layer;
    uint32_t kind;          /* 0 gate (w1), 1 up (w3), 2 down (w2) */
    uint64_t base;          /* virtual offset of expert 0 */
    uint64_t end;           /* base + n_expert * expert_bytes */
    uint64_t expert_bytes;  /* rows * cols/32 * 17 */
    uint32_t rows;
    uint32_t cols;
    uint32_t n_expert;
    uint64_t src_packed;    /* plane offsets in the raw-v1 file */
    uint64_t src_scale;
} k3_component;

static k3_component g_comp[K3_MAX_COMPONENTS];
static uint32_t g_n_comp;
static char g_primary[4096];
static char g_dir_b[4096];
static char g_dir_c[4096];   /* step 35: tier C (DS4_K3_EXPERT_DIR_C, a further copy on slower storage) for bulk cold reads */
static char g_dir_d[4096];   /* step 67: tier D (DS4_K3_EXPERT_DIR_D), a fourth home homed by the ETA balancer like tier C */
static uint64_t g_file_bytes;
static bool g_active;
static atomic_uint_fast64_t g_stat_reads, g_stat_bytes, g_stat_opens;
static atomic_uint_fast64_t g_stat_reads_tier[4], g_stat_ns_tier[4];   /* per tier: primary (0) / dir_b (1) / dir_c (2) / dir_d (3) */

/* step M1: experts read straight from the downloaded checkpoint (DS4_K3_EXPERT_CKPT_INDEX), for those missing from the directories */
static int g_ckpt_n = 0, *g_ckpt_fd = NULL;
static int16_t *g_ckpt_shard = NULL;   /* (layer << 10 | expert) -> shard or -1 */
static uint64_t *g_ckpt_off = NULL;
static int k3_ckpt_load(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "ds4: kimi-k3 checkpoint index %s: %s\n", path, strerror(errno)); return 0; }
    int n = 0;
    if (fscanf(f, "shards %d\n", &n) != 1 || n <= 0 || n > 4096) { fclose(f); return 0; }
    g_ckpt_fd = malloc(sizeof(int) * (size_t)n);
    g_ckpt_shard = malloc(sizeof(int16_t) * 128u * 1024u);
    g_ckpt_off = calloc(128u * 1024u, sizeof(uint64_t));
    if (!g_ckpt_fd || !g_ckpt_shard || !g_ckpt_off) { fclose(f); return 0; }
    for (uint32_t i = 0; i < 128u * 1024u; i++) g_ckpt_shard[i] = -1;
    char line[4600];
    for (int i = 0; i < n; i++) {
        if (!fgets(line, sizeof(line), f)) { fclose(f); return 0; }
        line[strcspn(line, "\n")] = '\0';
        g_ckpt_fd[i] = open(line, O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (g_ckpt_fd[i] < 0) g_ckpt_fd[i] = open(line, O_RDONLY | O_CLOEXEC);
        if (g_ckpt_fd[i] < 0) { fprintf(stderr, "ds4: kimi-k3 checkpoint shard %s: %s\n", line, strerror(errno)); fclose(f); return 0; }
    }
    unsigned l, e; int sh; unsigned long long off; uint32_t count = 0;
    while (fscanf(f, "%u %u %d %llu\n", &l, &e, &sh, &off) == 4) {
        if (l >= 128u || e >= 1024u || sh < 0 || sh >= n) continue;
        g_ckpt_shard[(l << 10) | e] = (int16_t)sh; g_ckpt_off[(l << 10) | e] = off; count++;
    }
    fclose(f);
    g_ckpt_n = n;
    fprintf(stderr, "ds4: kimi-k3 experts missing from the directories are read from the checkpoint: %u experts in %d shards\n", count, n);
    return 1;
}
static int k3_ckpt_read(uint32_t layer, uint32_t expert, void *out, uint64_t off, uint64_t len) {
    if (!g_ckpt_n || layer >= 128u || expert >= 1024u) return 0;
    const int sh = g_ckpt_shard[(layer << 10) | expert];
    if (sh < 0) return 0;
    uint64_t got = 0;
    while (got < len) {
        uint64_t want = len - got;
        if (want > ((uint64_t)1 << 30)) want = (uint64_t)1 << 30;   /* Darwin's pread refuses more than INT_MAX */
        const ssize_t r = pread(g_ckpt_fd[sh], (uint8_t *)out + got, (size_t)want, (off_t)(g_ckpt_off[(layer << 10) | expert] + off + got));
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; return 0; }
        got += (uint64_t)r;
    }
    return 1;
}
void ds4_k3_expert_source_configure(const char *primary_dir, const char *dir_b, uint64_t file_bytes) {
    {   /* step M1: the machine's own expert directories */
        const char *ep = getenv("DS4_K3_EXPERT_DIR"), *eb = getenv("DS4_K3_EXPERT_DIR_B");
        if (ep) primary_dir = ep;
        if (eb) dir_b = eb[0] ? eb : NULL;
    }
    snprintf(g_primary, sizeof(g_primary), "%s", primary_dir ? primary_dir : "");
    snprintf(g_dir_b, sizeof(g_dir_b), "%s", dir_b ? dir_b : "");
    g_file_bytes = file_bytes;
    g_active = g_primary[0] != '\0';
    {   /* step M1 */
        const char *ci = getenv("DS4_K3_EXPERT_CKPT_INDEX");
        if (ci && ci[0] && k3_ckpt_load(ci)) g_active = true;
    }
    g_dir_c[0] = '\0';
    const char *dc = getenv("DS4_K3_EXPERT_DIR_C");
    if (dc && dc[0]) {
        char probe[4600];
        struct stat st;
        snprintf(probe, sizeof(probe), "%s/L1-E0.bin", dc);
        if (stat(probe, &st) == 0 && (uint64_t)st.st_size == file_bytes) {
            snprintf(g_dir_c, sizeof(g_dir_c), "%s", dc);
            fprintf(stderr, "ds4: kimi-k3 expert tier C (bulk cold reads): %s\n", dc);
        } else {
            fprintf(stderr, "ds4: kimi-k3 expert tier C ignored: %s/L1-E0.bin missing or not %llu bytes\n", dc, (unsigned long long)file_bytes);
        }
    }
    g_dir_d[0] = '\0';
    const char *dd = getenv("DS4_K3_EXPERT_DIR_D");   /* step 67 */
    if (dd && dd[0]) {
        char probe[4600];
        struct stat st;
        snprintf(probe, sizeof(probe), "%s/L1-E0.bin", dd);
        if (stat(probe, &st) == 0 && (uint64_t)st.st_size == file_bytes) {
            snprintf(g_dir_d, sizeof(g_dir_d), "%s", dd);
            fprintf(stderr, "ds4: kimi-k3 expert tier D (fourth home): %s\n", dd);
        } else {
            fprintf(stderr, "ds4: kimi-k3 expert tier D ignored: %s/L1-E0.bin missing or not %llu bytes\n", dd, (unsigned long long)file_bytes);
        }
    }
}
int ds4_k3_expert_source_has_dir_c(void) { return g_dir_c[0] != '\0'; }
int ds4_k3_expert_source_has_dir_d(void) { return g_dir_d[0] != '\0'; }
/* step 42: how each tier answers an O_DIRECT open of one of its files (0 unknown, 1 direct, 2 buffered only, -1 no file) */
int ds4_k3_expert_source_probe_tier(int tier, uint32_t layer, uint32_t expert) {
    const char *dir = tier == 0 ? g_primary : tier == 1 ? g_dir_b : tier == 2 ? g_dir_c : tier == 3 ? g_dir_d : "";
    if (dir[0] == '\0') return 0;
    char path[4600];
    snprintf(path, sizeof(path), "%s/L%u-E%u.bin", dir, layer, expert);
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd >= 0) { close(fd); return 1; }
    if (errno != EINVAL) return -1;
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    close(fd);
    return 2;
}

bool ds4_k3_expert_source_active(void) { return g_active; }

static void k3_add(uint32_t layer, uint32_t kind, uint64_t base, uint32_t rows, uint32_t cols,
                   uint32_t n_expert, uint64_t src_packed, uint64_t src_scale) {
    if (g_n_comp >= K3_MAX_COMPONENTS || cols % K3_BLOCK != 0) {
        fprintf(stderr, "ds4: kimi-k3 expert source cannot register layer %u kind %u\n", layer, kind);
        exit(1);
    }
    k3_component c;
    c.layer = layer; c.kind = kind; c.base = base; c.rows = rows; c.cols = cols; c.n_expert = n_expert;
    c.expert_bytes = (uint64_t)rows * (cols / K3_BLOCK) * K3_BLOCK_BYTES;
    c.end = base + c.expert_bytes * n_expert;
    c.src_packed = src_packed; c.src_scale = src_scale;
    /* keep sorted by base for the lookup */
    uint32_t i = g_n_comp;
    while (i > 0 && g_comp[i - 1].base > base) { g_comp[i] = g_comp[i - 1]; i--; }
    g_comp[i] = c; g_n_comp++;
}

void ds4_k3_expert_source_register_layer(uint32_t layer, uint64_t gate_off, uint64_t up_off, uint64_t down_off,
                                         uint32_t n_expert, uint32_t ff, uint32_t latent) {
    k3_add(layer, 0, gate_off, ff, latent, n_expert, K3_W1_PACKED, K3_W1_SCALE);
    k3_add(layer, 1, up_off,   ff, latent, n_expert, K3_W3_PACKED, K3_W3_SCALE);
    k3_add(layer, 2, down_off, latent, ff, n_expert, K3_W2_PACKED, K3_W2_SCALE);
}

static const k3_component *k3_find(uint64_t offset) {
    uint32_t lo = 0, hi = g_n_comp;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (g_comp[mid].end <= offset) lo = mid + 1;
        else if (g_comp[mid].base > offset) hi = mid;
        else return &g_comp[mid];
    }
    return NULL;
}

bool ds4_k3_expert_source_covers(uint64_t offset, uint64_t bytes) {
    if (!g_active || bytes == 0) return false;
    const k3_component *c = k3_find(offset);
    return c != NULL && offset + bytes <= c->end;
}

static int k3_open_expert(uint32_t layer, uint32_t expert) {
    char path[4600];
    /* deltafin raw-v1 names layers 1..92 for the MoE layers; GGUF blk index == HF layer index */
    snprintf(path, sizeof(path), "%s/L%u-E%u.bin", g_primary, layer, expert);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0 && errno == ENOENT && g_dir_b[0] != '\0') {
        snprintf(path, sizeof(path), "%s/L%u-E%u.bin", g_dir_b, layer, expert);
        fd = open(path, O_RDONLY | O_CLOEXEC);
    }
    if (fd < 0) return -1;
    if (g_file_bytes != 0) {
        struct stat st;
        if (fstat(fd, &st) != 0 || (uint64_t)st.st_size != g_file_bytes) {
            close(fd); errno = EIO; return -1;
        }
    }
    atomic_fetch_add(&g_stat_opens, 1);
    return fd;
}

static int k3_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd, (char *)buf + done, (size_t)(bytes - done), (off_t)(offset + done));
        if (n < 0) { if (errno == EINTR) continue; return 0; }
        if (n == 0) { errno = EIO; return 0; }
        done += (uint64_t)n;
    }
    return 1;
}

/* One raw-v1 row (cols/2 packed bytes, cols/32 scale bytes) -> block_mxfp4 row (cols/32 x 17 bytes).
 * GGML nibble order: byte i of a block holds element i (low nibble) and element i+16 (high nibble);
 * raw-v1 holds element 2i (low) and 2i+1 (high) in byte i. */
static void k3_row_to_blocks(const uint8_t *packed, const uint8_t *scale, uint8_t *out, uint32_t cols) {
    const uint32_t blocks = cols / K3_BLOCK;
    for (uint32_t j = 0; j < blocks; j++) {
        const uint8_t *src = packed + (size_t)j * 16u;
        uint8_t *dst = out + (size_t)j * K3_BLOCK_BYTES;
        dst[0] = scale[j];
        for (uint32_t i = 0; i < 16u; i++) {
            const uint32_t sh = (i & 1u) * 4u;
            const uint8_t lo = (uint8_t)((src[i >> 1] >> sh) & 0x0fu);
            const uint8_t hi = (uint8_t)((src[8u + (i >> 1)] >> sh) & 0x0fu);
            dst[1u + i] = (uint8_t)(lo | (hi << 4));
        }
    }
}

int ds4_k3_expert_source_read(uint64_t offset, uint64_t bytes, void *out) {
    uint8_t *dst = (uint8_t *)out;
    uint64_t remaining = bytes;
    uint64_t pos = offset;
    atomic_fetch_add(&g_stat_reads, 1);
    atomic_fetch_add(&g_stat_bytes, bytes);
    while (remaining != 0) {
        const k3_component *c = k3_find(pos);
        if (!c) { errno = EINVAL; return 0; }
        const uint64_t within = pos - c->base;
        const uint32_t expert = (uint32_t)(within / c->expert_bytes);
        uint64_t eoff = within % c->expert_bytes;
        const uint64_t avail = c->expert_bytes - eoff;
        const uint64_t take = remaining < avail ? remaining : avail;
        const uint32_t row_bytes = (c->cols / K3_BLOCK) * K3_BLOCK_BYTES;
        const uint32_t packed_row = c->cols / 2u;
        const uint32_t scale_row = c->cols / K3_BLOCK;
        const uint32_t row0 = (uint32_t)(eoff / row_bytes);
        const uint32_t row1 = (uint32_t)((eoff + take - 1) / row_bytes);   /* inclusive */
        const uint32_t nrows = row1 - row0 + 1u;
        const int fd = k3_open_expert(c->layer, expert);
        if (fd < 0) return 0;
        uint8_t *packed = malloc((size_t)nrows * packed_row);
        uint8_t *scale = malloc((size_t)nrows * scale_row);
        uint8_t *rowbuf = malloc(row_bytes);
        if (!packed || !scale || !rowbuf) { free(packed); free(scale); free(rowbuf); close(fd); errno = ENOMEM; return 0; }
        int ok = k3_pread_full(fd, packed, (uint64_t)nrows * packed_row, c->src_packed + (uint64_t)row0 * packed_row) &&
                 k3_pread_full(fd, scale, (uint64_t)nrows * scale_row, c->src_scale + (uint64_t)row0 * scale_row);
        close(fd);
        if (!ok) { free(packed); free(scale); free(rowbuf); return 0; }
        uint64_t produced = 0;
        for (uint32_t r = 0; r < nrows && produced < take; r++) {
            k3_row_to_blocks(packed + (size_t)r * packed_row, scale + (size_t)r * scale_row, rowbuf, c->cols);
            const uint64_t row_start = (uint64_t)(row0 + r) * row_bytes;
            const uint64_t from = eoff + produced > row_start ? eoff + produced - row_start : 0;
            uint64_t n = row_bytes - from;
            if (n > take - produced) n = take - produced;
            memcpy(dst + produced, rowbuf + from, (size_t)n);
            produced += n;
        }
        free(packed); free(scale); free(rowbuf);
        dst += take; pos += take; remaining -= take;
    }
    return 1;
}

void ds4_k3_expert_source_stats(uint64_t *reads, uint64_t *bytes, uint64_t *files_opened) {
    if (reads) *reads = atomic_load(&g_stat_reads);
    if (bytes) *bytes = atomic_load(&g_stat_bytes);
    if (files_opened) *files_opened = atomic_load(&g_stat_opens);
}

uint64_t ds4_k3_expert_source_file_bytes(void) { return g_file_bytes; }

/* Which tier holds an expert: a bitmap of the primary directory, built once by listing it (41,500 entries in ~0.1 s),
   so the expert pool can treat cold-tier files differently without a stat per lookup. 1 = primary (hot dir). */
#include <dirent.h>
static uint8_t *g_primary_bits = NULL;
static uint32_t g_primary_layers = 0, g_primary_experts = 0;
static void k3_primary_bitmap_build(void) {
    if (g_primary_bits || g_primary[0] == '\0') return;
    g_primary_layers = 128u; g_primary_experts = 1024u;
    g_primary_bits = calloc((size_t)g_primary_layers * g_primary_experts / 8u, 1u);
    if (!g_primary_bits) return;
    DIR *d = opendir(g_primary);
    if (!d) return;
    struct dirent *de;
    uint32_t n = 0;
    while ((de = readdir(d)) != NULL) {
        unsigned l = 0, e = 0;
        if (sscanf(de->d_name, "L%u-E%u.bin", &l, &e) == 2 && l < g_primary_layers && e < g_primary_experts) {
            const size_t bit = (size_t)l * g_primary_experts + e;
            g_primary_bits[bit >> 3] |= (uint8_t)(1u << (bit & 7u));
            n++;
        }
    }
    closedir(d);
    fprintf(stderr, "ds4: kimi-k3 expert tiers: %u files in the primary directory\n", n);
}
int ds4_k3_expert_source_is_primary(uint32_t layer, uint32_t expert) {
    if (!g_primary_bits) k3_primary_bitmap_build();
    if (!g_primary_bits || layer >= g_primary_layers || expert >= g_primary_experts) return 1;
    const size_t bit = (size_t)layer * g_primary_experts + expert;
    return (g_primary_bits[bit >> 3] >> (bit & 7u)) & 1u;
}
/* step 38: which tiers hold an expert (0 primary, 1 dir_b, 2 dir_c = complete set). dir_b is listed once like the
   primary directory; files copied onto the RAID keep their dir_b copy, so they have two homes. */
static uint8_t *g_dirb_bits = NULL;
static int g_dirb_listed = 0;
/* step 62: tier C listed the same way (a partial second home must not attract chunks for files it lacks) */
static uint8_t *g_dirc_bits = NULL;
static int g_dirc_listed = 0;
static void k3_dirc_bitmap_build(void) {
    if (g_dirc_listed) return;
    g_dirc_listed = 1;
    if (g_dir_c[0] == '\0') return;
    g_dirc_bits = calloc((size_t)128u * 1024u / 8u, 1u);
    if (!g_dirc_bits) return;
    DIR *d = opendir(g_dir_c);
    if (!d) { free(g_dirc_bits); g_dirc_bits = NULL; return; }
    struct dirent *de;
    uint32_t n = 0;
    while ((de = readdir(d)) != NULL) {
        unsigned l = 0, e = 0;
        if (sscanf(de->d_name, "L%u-E%u.bin", &l, &e) == 2 && l < 128u && e < 1024u) {
            const size_t bit = (size_t)l * 1024u + e;
            g_dirc_bits[bit >> 3] |= (uint8_t)(1u << (bit & 7u));
            n++;
        }
    }
    closedir(d);
    fprintf(stderr, "ds4: kimi-k3 expert tiers: %u files in tier C\n", n);
}
/* step 67: tier D listed like tier C */
static uint8_t *g_dird_bits = NULL;
static int g_dird_listed = 0;
static void k3_dird_bitmap_build(void) {
    if (g_dird_listed) return;
    g_dird_listed = 1;
    if (g_dir_d[0] == '\0') return;
    g_dird_bits = calloc((size_t)128u * 1024u / 8u, 1u);
    if (!g_dird_bits) return;
    DIR *d = opendir(g_dir_d);
    if (!d) { free(g_dird_bits); g_dird_bits = NULL; return; }
    struct dirent *de;
    uint32_t n = 0;
    while ((de = readdir(d)) != NULL) {
        unsigned l = 0, e = 0;
        if (sscanf(de->d_name, "L%u-E%u.bin", &l, &e) == 2 && l < 128u && e < 1024u) {
            const size_t bit = (size_t)l * 1024u + e;
            g_dird_bits[bit >> 3] |= (uint8_t)(1u << (bit & 7u));
            n++;
        }
    }
    closedir(d);
    fprintf(stderr, "ds4: kimi-k3 expert tiers: %u files in tier D\n", n);
}
static void k3_dirb_bitmap_build(void) {
    if (g_dirb_listed) return;
    g_dirb_listed = 1;
    if (g_dir_b[0] == '\0') return;
    g_dirb_bits = calloc((size_t)128u * 1024u / 8u, 1u);
    if (!g_dirb_bits) return;
    DIR *d = opendir(g_dir_b);
    if (!d) { free(g_dirb_bits); g_dirb_bits = NULL; return; }
    struct dirent *de;
    uint32_t n = 0;
    while ((de = readdir(d)) != NULL) {
        unsigned l = 0, e = 0;
        if (sscanf(de->d_name, "L%u-E%u.bin", &l, &e) == 2 && l < 128u && e < 1024u) {
            const size_t bit = (size_t)l * 1024u + e;
            g_dirb_bits[bit >> 3] |= (uint8_t)(1u << (bit & 7u));
            n++;
        }
    }
    closedir(d);
    fprintf(stderr, "ds4: kimi-k3 expert tiers: %u files in dir_b\n", n);
}
int ds4_k3_expert_source_has_tier(uint32_t layer, uint32_t expert, int tier) {
    if (tier == 0) return ds4_k3_expert_source_is_primary(layer, expert);
    if (tier == 3) {   /* step 67 */
        if (g_dir_d[0] == '\0') return 0;
        if (!g_dird_listed) k3_dird_bitmap_build();
        if (!g_dird_bits || layer >= 128u || expert >= 1024u) return 1;
        const size_t bit = (size_t)layer * 1024u + expert;
        return (g_dird_bits[bit >> 3] >> (bit & 7u)) & 1u;
    }
    if (tier == 2) {   /* step 62 */
        if (g_dir_c[0] == '\0') return 0;
        if (!g_dirc_listed) k3_dirc_bitmap_build();
        if (!g_dirc_bits || layer >= 128u || expert >= 1024u) return 1;
        const size_t bit = (size_t)layer * 1024u + expert;
        return (g_dirc_bits[bit >> 3] >> (bit & 7u)) & 1u;
    }
    if (tier != 1 || g_dir_b[0] == '\0') return 0;
    if (!g_dirb_listed) k3_dirb_bitmap_build();
    if (!g_dirb_bits || layer >= 128u || expert >= 1024u) return !ds4_k3_expert_source_is_primary(layer, expert);
    const size_t bit = (size_t)layer * 1024u + expert;
    return (g_dirb_bits[bit >> 3] >> (bit & 7u)) & 1u;
}

/* whole-file reads per tier (primary = hot dir, dir_b = cold dir) and the nanoseconds their preads took */
void ds4_k3_expert_source_tier_stats(uint64_t *reads_a, uint64_t *reads_b, uint64_t *ns_a, uint64_t *ns_b) {
    if (reads_a) *reads_a = atomic_load(&g_stat_reads_tier[0]);
    if (reads_b) *reads_b = atomic_load(&g_stat_reads_tier[1]);
    if (ns_a) *ns_a = atomic_load(&g_stat_ns_tier[0]);
    if (ns_b) *ns_b = atomic_load(&g_stat_ns_tier[1]);
}

/* read latency histogram per tier: buckets <1.5, <2.5, <3.5, <4.5, <6, <8, <12, >=12 ms */
static atomic_uint_fast64_t g_lat_hist[4][8];
static const uint64_t k3_lat_edges_ns[7] = {1500000ull, 2500000ull, 3500000ull, 4500000ull, 6000000ull, 8000000ull, 12000000ull};
void ds4_k3_expert_source_lat_hist(int tier, uint64_t *out8) {
    const int t = (tier < 0 || tier > 3) ? 0 : tier;
    for (int i = 0; i < 8; i++) out8[i] = atomic_load(&g_lat_hist[t][i]);
}
void ds4_k3_expert_source_tier_stat(int tier, uint64_t *reads, uint64_t *ns) {
    const int t = (tier < 0 || tier > 3) ? 0 : tier;
    if (reads) *reads = atomic_load(&g_stat_reads_tier[t]);
    if (ns) *ns = atomic_load(&g_stat_ns_tier[t]);
}
/* prefer_c: open the tier C copy first (bulk cold reads while the Samsung backlog is deep); falls back to the tiers */
static int k3_open_expert_flags2(uint32_t layer, uint32_t expert, int extra_flags, int prefer_c, int *tier) {
    char path[4600];
    int fd = -1;
    if (prefer_c && g_dir_c[0] != '\0') {
        snprintf(path, sizeof(path), "%s/L%u-E%u.bin", g_dir_c, layer, expert);
        fd = open(path, O_RDONLY | O_CLOEXEC | extra_flags);
        if (fd >= 0) { if (tier) *tier = 2; atomic_fetch_add(&g_stat_opens, 1); return fd; }
        /* step 41: a filesystem that refuses O_DIRECT (the kernel APFS driver on tier C) must be retried buffered on
           the same tier by the caller, not served by another tier outside its in-flight cap */
        if (errno == EINVAL && (extra_flags & O_DIRECT)) { if (tier) *tier = 2; return -1; }
    }
    snprintf(path, sizeof(path), "%s/L%u-E%u.bin", g_primary, layer, expert);
    fd = open(path, O_RDONLY | O_CLOEXEC | extra_flags);
    if (tier) *tier = 0;
    if (fd < 0 && errno == ENOENT && g_dir_b[0] != '\0') {
        snprintf(path, sizeof(path), "%s/L%u-E%u.bin", g_dir_b, layer, expert);
        fd = open(path, O_RDONLY | O_CLOEXEC | extra_flags);
        if (tier) *tier = 1;
    }
    if (fd >= 0) atomic_fetch_add(&g_stat_opens, 1);
    return fd;
}
static int k3_open_expert_flags(uint32_t layer, uint32_t expert, int extra_flags, int *tier) {
    return k3_open_expert_flags2(layer, expert, extra_flags, 0, tier);
}

int ds4_k3_expert_source_read_file(uint32_t layer, uint32_t expert, void *out, int direct) {
    return ds4_k3_expert_source_read_file_tier(layer, expert, out, direct, 0);
}
/* step 38: open exactly one tier's copy (0 primary, 1 dir_b, 2 dir_c); -1 = the usual chain */
static int k3_open_expert_tier(uint32_t layer, uint32_t expert, int extra_flags, int want, int *tier) {
    if (want < 0) return k3_open_expert_flags2(layer, expert, extra_flags, 0, tier);
    const char *dir = want == 0 ? g_primary : want == 1 ? g_dir_b : want == 3 ? g_dir_d : g_dir_c;
    if (dir[0] == '\0') return k3_open_expert_flags2(layer, expert, extra_flags, want == 2, tier);
    char path[4600];
    snprintf(path, sizeof(path), "%s/L%u-E%u.bin", dir, layer, expert);
    const int fd = open(path, O_RDONLY | O_CLOEXEC | extra_flags);
    if (fd < 0) {
        if (errno == EINVAL && (extra_flags & O_DIRECT)) { if (tier) *tier = want; return -1; }   /* step 41: retry buffered here */
        return k3_open_expert_flags2(layer, expert, extra_flags, want == 2, tier);
    }
    if (tier) *tier = want;
    atomic_fetch_add(&g_stat_opens, 1);
    return fd;
}
/* step 38: read [off, off+len) of an expert file from a chosen tier into base+off (direct I/O when aligned) */
int ds4_k3_expert_source_read_range_tier(uint32_t layer, uint32_t expert, void *base, uint64_t off, uint64_t len,
                                         int direct, int want_tier, int *tier_used) {
    if (!g_active || g_file_bytes == 0 || !base || off >= g_file_bytes || len == 0 || off + len > g_file_bytes) { errno = EINVAL; return 0; }
    uint8_t *out = (uint8_t *)base + off;
    const bool want_direct = direct != 0 && (((uintptr_t)out & 4095u) == 0) && (off % 4096u) == 0 && (len % 4096u) == 0;
    for (int attempt = want_direct ? 0 : 1; attempt < 2; attempt++) {
        int tier = 0;
        const int fd = k3_open_expert_tier(layer, expert, attempt == 0 ? O_DIRECT : 0, want_tier, &tier);
        if (fd < 0) {
            if (attempt == 0 && errno == EINVAL) continue;
            if (errno == ENOENT && g_ckpt_n) {   /* step M1: straight from the checkpoint */
                if (!k3_ckpt_read(layer, expert, out, off, len)) return 0;
                if (tier_used) *tier_used = 3;
                atomic_fetch_add(&g_stat_reads, 1); atomic_fetch_add(&g_stat_bytes, len); atomic_fetch_add(&g_stat_reads_tier[3], 1);
                return 1;
            }
            return 0;
        }
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        const int ok = k3_pread_full(fd, out, len, off);
        const int saved = errno;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        /* a buffered read (a filesystem that refuses O_DIRECT, e.g. the kernel APFS driver on tier C) would otherwise
           leave its pages in the guest page cache beside the pinned arena: drop them, they are never read twice here */
        if (ok && attempt == 1) (void)posix_fadvise(fd, (off_t)off, (off_t)len, POSIX_FADV_DONTNEED);
        close(fd);
        if (ok) {
            if (tier_used) *tier_used = tier;
            atomic_fetch_add(&g_stat_reads, 1);
            atomic_fetch_add(&g_stat_bytes, len);
            atomic_fetch_add(&g_stat_reads_tier[tier], 1);
            const uint64_t ns = (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ull + (uint64_t)(t1.tv_nsec - t0.tv_nsec);
            atomic_fetch_add(&g_stat_ns_tier[tier], ns);
            { int bk = 0; while (bk < 7 && ns > k3_lat_edges_ns[bk]) bk++; atomic_fetch_add(&g_lat_hist[tier][bk], 1); }
            return 1;
        }
        if (attempt == 0 && saved == EINVAL) continue;
        errno = saved;
        return 0;
    }
    return 0;
}
int ds4_k3_expert_source_read_file_tier(uint32_t layer, uint32_t expert, void *out, int direct, int prefer_c) {
    if (!g_active || g_file_bytes == 0 || !out) { errno = EINVAL; return 0; }
    const bool want_direct = direct != 0 && (((uintptr_t)out & 4095u) == 0) && (g_file_bytes % 4096u) == 0;
    for (int attempt = want_direct ? 0 : 1; attempt < 2; attempt++) {
        int tier = 0;
        const int fd = k3_open_expert_flags2(layer, expert, attempt == 0 ? O_DIRECT : 0, prefer_c, &tier);
        if (fd < 0) {
            if (attempt == 0 && errno == EINVAL) continue;   /* filesystem refuses O_DIRECT: buffered */
            if (errno == ENOENT && g_ckpt_n) {   /* step M1: straight from the checkpoint */
                struct timespec c0, c1;
                clock_gettime(CLOCK_MONOTONIC, &c0);
                if (!k3_ckpt_read(layer, expert, out, 0, g_file_bytes)) return 0;
                clock_gettime(CLOCK_MONOTONIC, &c1);
                atomic_fetch_add(&g_stat_reads, 1); atomic_fetch_add(&g_stat_bytes, g_file_bytes); atomic_fetch_add(&g_stat_reads_tier[3], 1);
                atomic_fetch_add(&g_stat_ns_tier[3], (uint64_t)(c1.tv_sec - c0.tv_sec) * 1000000000ull + (uint64_t)(c1.tv_nsec - c0.tv_nsec));
                return 1;
            }
            return 0;
        }
        struct stat st;
        if (fstat(fd, &st) != 0 || (uint64_t)st.st_size != g_file_bytes) { close(fd); errno = EIO; return 0; }
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        const int ok = k3_pread_full(fd, out, g_file_bytes, 0);
        const int saved = errno;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        close(fd);
        if (ok) {
            atomic_fetch_add(&g_stat_reads, 1);
            atomic_fetch_add(&g_stat_bytes, g_file_bytes);
            atomic_fetch_add(&g_stat_reads_tier[tier], 1);
            const uint64_t ns = (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ull + (uint64_t)(t1.tv_nsec - t0.tv_nsec);
            atomic_fetch_add(&g_stat_ns_tier[tier], ns);
            { int bk = 0; while (bk < 7 && ns > k3_lat_edges_ns[bk]) bk++; atomic_fetch_add(&g_lat_hist[tier][bk], 1); }
            return 1;
        }
        if (attempt == 0 && saved == EINVAL) continue;        /* direct read refused: buffered */
        errno = saved;
        return 0;
    }
    return 0;
}
