/* Kimi K3 routed-expert source (2026-10-31).
 *
 * The K3 GGUF declares the routed-expert tensors (blk.N.ffn_{gate,up,down}_exps.weight, MXFP4, 17 bytes per 32) with
 * offsets in a sparse region that holds no data. Every byte in that region is served from the deltafin raw-v1 expert
 * files instead (one 17,547,264-byte file per expert: w1 packed, w1 scale, w2 packed, w2 scale, w3 packed, w3 scale;
 * packed = e2m1 nibbles low-first per element pair, scale = e8m0 per 32), re-laid into block_mxfp4 on the fly. */
#ifndef DS4_KIMI_K3_H
#define DS4_KIMI_K3_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void ds4_k3_expert_source_configure(const char *primary_dir, const char *dir_b, uint64_t file_bytes);
void ds4_k3_expert_source_register_layer(uint32_t layer, uint64_t gate_off, uint64_t up_off, uint64_t down_off,
                                         uint32_t n_expert, uint32_t ff, uint32_t latent);
bool ds4_k3_expert_source_active(void);
bool ds4_k3_expert_source_covers(uint64_t offset, uint64_t bytes);
/* Fill `out` with `bytes` bytes of the virtual expert region starting at `offset`. 1 = ok, 0 = error (errno). */
int  ds4_k3_expert_source_read(uint64_t offset, uint64_t bytes, void *out);
void ds4_k3_expert_source_stats(uint64_t *reads, uint64_t *bytes, uint64_t *files_opened);
/* Whole raw-v1 expert file (layer, expert) into `out`, which must hold file_bytes. With direct != 0 and a
 * 4 KiB-aligned `out` the read is O_DIRECT (falls back to buffered I/O if refused). 1 = ok, 0 = error (errno). */
int  ds4_k3_expert_source_read_file(uint32_t layer, uint32_t expert, void *out, int direct);
uint64_t ds4_k3_expert_source_file_bytes(void);
void ds4_k3_expert_source_lat_hist(int tier, uint64_t *out8);   /* read latency buckets <1.5,<2.5,<3.5,<4.5,<6,<8,<12,>=12 ms */
void ds4_k3_expert_source_tier_stats(uint64_t *reads_a, uint64_t *reads_b, uint64_t *ns_a, uint64_t *ns_b);
int  ds4_k3_expert_source_is_primary(uint32_t layer, uint32_t expert);   /* 1 = served from the primary (hot) directory */
int  ds4_k3_expert_source_read_file_tier(uint32_t layer, uint32_t expert, void *out, int direct, int prefer_c);   /* step 35: prefer_c reads the tier C copy (DS4_K3_EXPERT_DIR_C) */
int  ds4_k3_expert_source_has_dir_c(void);
int  ds4_k3_expert_source_has_dir_d(void);   /* step 67: tier D (DS4_K3_EXPERT_DIR_D), a fourth home */
int  ds4_k3_expert_source_has_tier(uint32_t layer, uint32_t expert, int tier);   /* step 38: 0 primary, 1 dir_b, 2 dir_c */
int  ds4_k3_expert_source_probe_tier(int tier, uint32_t layer, uint32_t expert);   /* step 42: 1 direct I/O, 2 buffered only, -1 no file, 0 tier unset */
int  ds4_k3_expert_source_read_range_tier(uint32_t layer, uint32_t expert, void *base, uint64_t off, uint64_t len,
                                          int direct, int want_tier, int *tier_used);   /* step 38: chunk read from one tier (-1 chain) */
void ds4_k3_expert_source_tier_stat(int tier, uint64_t *reads, uint64_t *ns);   /* 0 primary, 1 dir_b (Samsung), 2 dir_c */
#ifdef __cplusplus
}
#endif
#endif
