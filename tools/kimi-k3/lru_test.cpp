#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <list>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
static const uint64_t K3_NOKEY = ~0ull;
#include "lru_struct.hpp"
// decode simulation: 92 layers x 16 experts per token; per layer a Zipf-ish popularity over 896 experts; 15 % cold tier.
int main(int argc, char **argv) {
    const int chance = argc > 1 ? atoi(argv[1]) : 1;
    char buf[8]; snprintf(buf, sizeof buf, "%d", chance); setenv("DS4_K3_CACHE_COLD_CHANCE", buf, 1);
    std::mt19937 rng(7);
    std::vector<uint8_t> cold(92 * 1024, 0);
    for (auto &c : cold) c = (rng() % 100) < 15;
    // popularity: expert e of layer l has weight 1/(rank+1)^0.8 with a per-layer permutation
    std::vector<std::vector<double>> w(92, std::vector<double>(896));
    for (int l = 0; l < 92; l++) { std::vector<int> perm(896); for (int i = 0; i < 896; i++) perm[i] = i; std::shuffle(perm.begin(), perm.end(), rng);
        for (int i = 0; i < 896; i++) w[l][perm[i]] = 1.0 / pow(i + 1.0, 0.8); }
    std::vector<std::discrete_distribution<int>> dist; for (int l = 0; l < 92; l++) dist.emplace_back(w[l].begin(), w[l].end());
    k3_lru dc, hc; dc.init(64); hc.init(2202); dc.grow(1494);
    long hits_dev = 0, hits_host = 0, disk = 0, tok_hits = 0;
    for (int t = 0; t < 40; t++) {
        long hd = 0, hh = 0, dk = 0;
        for (uint32_t l = 0; l < 92; l++) {
            int sel[16]; int n = 0;
            while (n < 16) { int e = dist[l](rng); bool dup = false; for (int i = 0; i < n; i++) if (sel[i] == e) dup = true; if (!dup) sel[n++] = e; }
            for (int i = 0; i < 16; i++) {
                const uint64_t k = ((uint64_t)l << 20) | (uint32_t)sel[i];
                int d = dc.lookup(k, true);
                if (d >= 0) { hd++; continue; }
                int h = hc.lookup(k, true);
                const bool read = h < 0;
                const bool ct = cold[l * 1024 + sel[i]];
                if (read) { h = hc.alloc(k, false); if (h < 0) { printf("host full\n"); return 1; } }
                hc.pinned[(uint32_t)h] = 1;
                uint64_t ev = K3_NOKEY;
                d = dc.alloc(k, true, &ev, false);   /* device: no second chance */
                if (d < 0) { printf("dev full\n"); return 1; }
                if (ev != K3_NOKEY && hc.lookup(ev, false) < 0) {
                    const bool c2 = cold[(ev >> 20) * 1024 + (ev & 0xfffff)];
                    (void)hc.alloc(ev, true, nullptr, c2);
                }
                if (read) dk++; else hh++;
                hc.pinned[(uint32_t)h] = 0; hc.drop((uint32_t)h);
            }
        }
        hits_dev += hd; hits_host += hh; disk += dk;
        if (t >= 20) tok_hits += hd;
        if (t % 10 == 0 || t == 39) printf("token %d: device %ld host %ld disk %ld\n", t, hd, hh, dk);
    }
    // sanity: list length and map consistency
    size_t len = dc.order.size(); printf("chance=%d: order size %zu (cap %u), map %zu; avg device hits/token (tokens 20-39) %.0f\n", chance, len, dc.cap, dc.map.size(), tok_hits / 20.0);
    return 0;
}
