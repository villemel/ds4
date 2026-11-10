// readbench2: T file-threads; each reads a random whole 17.5 MB expert file with O_DIRECT, split into K parallel
// slice reads (K sub-threads per file). Reports GB/s and the latency per file: does splitting a read cut the
// per-file latency on the thin LV / RAID paths?
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
static char **files; static int n_files; static int seconds = 5, K = 1; static size_t fsize = 17547264;
static volatile int stop_flag = 0;
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
typedef struct { int fd; char *buf; size_t off, len; } slice;
static void *slice_worker(void *a){ slice *s = a; size_t done = 0; while (done < s->len) { ssize_t r = pread(s->fd, s->buf + done, s->len - done, (off_t)(s->off + done)); if (r <= 0) { perror("pread"); break; } done += (size_t)r; } return NULL; }
static void *worker(void *arg){
    long id = (long)arg; unsigned seed = 777u + (unsigned)id * 7919u;
    char *buf = NULL; if (posix_memalign((void**)&buf, 4096, fsize)) return NULL;
    long long bytes = 0; long reads = 0; double lat = 0;
    pthread_t th[64]; slice sl[64];
    size_t slice_len = ((fsize / (size_t)K) + 4095) & ~(size_t)4095;
    while (!stop_flag) {
        const char *f = files[rand_r(&seed) % n_files];
        double t0 = now();
        int fd = open(f, O_RDONLY | O_DIRECT); if (fd < 0) { perror(f); break; }
        for (int k = 0; k < K; k++) { sl[k].fd = fd; sl[k].buf = buf; sl[k].off = (size_t)k * slice_len; sl[k].len = sl[k].off >= fsize ? 0 : (sl[k].off + slice_len > fsize ? fsize - sl[k].off : slice_len); if (K > 1) pthread_create(&th[k], NULL, slice_worker, &sl[k]); else slice_worker(&sl[k]); }
        if (K > 1) for (int k = 0; k < K; k++) pthread_join(th[k], NULL);
        close(fd); bytes += (long long)fsize; reads++; lat += now() - t0;
    }
    free(buf);
    double *res = malloc(3 * sizeof(double)); res[0] = (double)bytes; res[1] = (double)reads; res[2] = lat; return res;
}
int main(int argc, char **argv){
    if (argc < 5) { fprintf(stderr, "usage: readbench2 THREADS SLICES SECONDS LISTFILE\n"); return 2; }
    int threads = atoi(argv[1]); K = atoi(argv[2]); if (K < 1) K = 1; if (K > 64) K = 64; seconds = atoi(argv[3]);
    FILE *fp = fopen(argv[4], "r"); if (!fp) { perror(argv[4]); return 1; }
    char line[4096]; int cap = 1024; files = malloc(cap * sizeof(char*));
    while (fgets(line, sizeof line, fp)) { line[strcspn(line, "\n")] = 0; if (!line[0]) continue; if (n_files == cap) { cap *= 2; files = realloc(files, cap * sizeof(char*)); } files[n_files++] = strdup(line); }
    fclose(fp);
    pthread_t *th = malloc(threads * sizeof(pthread_t));
    double t0 = now();
    for (long i = 0; i < threads; i++) pthread_create(&th[i], NULL, worker, (void*)i);
    while (now() - t0 < seconds) usleep(20000);
    stop_flag = 1;
    double bytes = 0, reads = 0, lat = 0;
    for (int i = 0; i < threads; i++) { double *r = NULL; pthread_join(th[i], (void**)&r); if (r) { bytes += r[0]; reads += r[1]; lat += r[2]; free(r); } }
    double el = now() - t0;
    printf("threads=%d slices=%d reads=%.0f %.2f GB/s, %.1f ms per file\n", threads, K, reads, bytes / el / 1e9, reads ? lat * 1e3 / reads : 0.0);
    return 0;
}
