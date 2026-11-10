// readbench: N threads, each repeatedly opens a random file from a list and preads it whole with O_DIRECT
// into an aligned buffer (the K3 expert pipeline's access pattern). Prints GB/s over the run.
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
static char **files; static int n_files; static int seconds = 6; static size_t fsize = 17547264;
static volatile int stop_flag = 0;
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
static void *worker(void *arg){
    long id = (long)arg; unsigned seed = 12345u + (unsigned)id * 7919u;
    void *buf = NULL; if (posix_memalign(&buf, 4096, fsize)) return NULL;
    long long bytes = 0; long reads = 0;
    while (!stop_flag) {
        const char *f = files[rand_r(&seed) % n_files];
        int fd = open(f, O_RDONLY | O_DIRECT); if (fd < 0) { perror(f); break; }
        size_t off = 0; while (off < fsize) { ssize_t r = pread(fd, (char*)buf + off, fsize - off, (off_t)off); if (r <= 0) { perror("pread"); break; } off += (size_t)r; }
        close(fd); bytes += (long long)off; reads++;
    }
    free(buf);
    long long *res = malloc(2 * sizeof(long long)); res[0] = bytes; res[1] = reads; return res;
}
int main(int argc, char **argv){
    if (argc < 4) { fprintf(stderr, "usage: readbench THREADS SECONDS LISTFILE [LISTFILE2 THREADS2]\n"); return 2; }
    int threads = atoi(argv[1]); seconds = atoi(argv[2]);
    // list A
    FILE *fp = fopen(argv[3], "r"); if (!fp) { perror(argv[3]); return 1; }
    char line[4096]; int cap = 1024; files = malloc(cap * sizeof(char*));
    while (fgets(line, sizeof line, fp)) { line[strcspn(line, "\n")] = 0; if (!line[0]) continue; if (n_files == cap) { cap *= 2; files = realloc(files, cap * sizeof(char*)); } files[n_files++] = strdup(line); }
    fclose(fp);
    // optional list B served by its own threads (two-device mix): handled by running a second process instead; keep simple
    pthread_t *th = malloc(threads * sizeof(pthread_t));
    double t0 = now();
    for (long i = 0; i < threads; i++) pthread_create(&th[i], NULL, worker, (void*)i);
    while (now() - t0 < seconds) usleep(20000);
    stop_flag = 1;
    long long bytes = 0, reads = 0;
    for (int i = 0; i < threads; i++) { long long *r = NULL; pthread_join(th[i], (void**)&r); if (r) { bytes += r[0]; reads += r[1]; free(r); } }
    double el = now() - t0;
    printf("threads=%d files=%d reads=%lld %.2f GB/s (%.1f ms per read per thread)\n", threads, n_files, reads, bytes / el / 1e9, reads ? el * 1e3 * threads / reads : 0.0);
    return 0;
}
