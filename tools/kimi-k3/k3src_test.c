/* Expert-source test: serve layer 1 expert 7's gate/up/down components through the K3 source and dump them. */
#include "ds4_kimi_k3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    const uint64_t expert_bytes = 3072ull * (3584 / 32) * 17;  /* 5,849,088 */
    const uint64_t tensor_bytes = expert_bytes * 896;
    ds4_k3_expert_source_configure("<path>", "<path>", 17547264);
    ds4_k3_expert_source_register_layer(1, 0, tensor_bytes, 2 * tensor_bytes, 896, 3072, 3584);
    const uint32_t expert = argc > 1 ? (uint32_t)atoi(argv[1]) : 7u;
    const char *names[3] = {"gate", "up", "down"};
    for (int k = 0; k < 3; k++) {
        uint8_t *buf = malloc(expert_bytes);
        const uint64_t off = (uint64_t)k * tensor_bytes + expert * expert_bytes;
        if (!ds4_k3_expert_source_covers(off, expert_bytes) || !ds4_k3_expert_source_read(off, expert_bytes, buf)) {
            perror("read"); return 1;
        }
        char path[256]; snprintf(path, sizeof(path), "/tmp/k3src-%s.bin", names[k]);
        FILE *f = fopen(path, "wb"); fwrite(buf, 1, expert_bytes, f); fclose(f); free(buf);
    }
    /* partial read crossing rows: 1000 bytes from the middle of the up component */
    uint8_t *part = malloc(1000);
    const uint64_t poff = tensor_bytes + expert * expert_bytes + 1904 * 5 + 700;
    if (!ds4_k3_expert_source_read(poff, 1000, part)) { perror("partial"); return 1; }
    FILE *f = fopen("/tmp/k3src-partial.bin", "wb"); fwrite(part, 1, 1000, f); fclose(f);
    uint64_t r, b, o; ds4_k3_expert_source_stats(&r, &b, &o);
    printf("reads=%llu bytes=%llu opens=%llu\n", (unsigned long long)r, (unsigned long long)b, (unsigned long long)o);
    return 0;
}
