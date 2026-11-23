/*
 * Assignment 4 - distributed dense matrix multiplication with MPI (Open MPI), in C.
 * Storage: one flat double array per matrix, row-major (element (i, j) at i*n + j), as in Assignments 1-3.
 *
 * Algorithm: 1-D row-block decomposition (the distributed counterpart of the OpenMP static schedule of A3).
 *   rank r (0..P-1) owns rows [d_r, d_r + m_r) of A and C, with m_r = n/P (+1 for the first n%P ranks).
 *   1. scatter  rank 0 sends to every rank its band of rows of A            (MPI_Scatterv)
 *   2. bcast    rank 0 sends the whole B to every rank                       (MPI_Bcast)
 *   3. compute  every rank computes its band of C with the A3 row kernel     (row_ikj, no MPI inside)
 *   4. gather   rank 0 collects the bands of C into the full C              (MPI_Gatherv)
 * Rank 0 keeps its own band in place (MPI_IN_PLACE): with P = 1 nothing is copied and only the
 * framework overhead remains. Same row body and same order of the k terms as A3 seq/par, so the
 * result must be bitwise identical to seq for every P.
 *
 * Timing (rank 0, MPI_Wtime): MPI_Barrier before the start and after every phase, so the four
 * phases do not overlap and add up to the total. Total boundary: A and B in the memory of rank 0
 * -> complete C in the memory of rank 0. Outside: mpirun launch, MPI_Init, reading the input files,
 * allocation of all buffers (reported separately as init_s and load_s).
 *
 * Build: mpicc -O2 -fno-tree-vectorize -std=c11 -Wall -o build/matmul4 src/matmul4.c -lm
 * Usage (under mpirun -np P):
 *   matmul4 test                          correctness tests (CSV on stdout of rank 0)
 *   matmul4 bench n warmup reps datadir   one row per measured repetition (see HEADER)
 *   matmul4 init                          MPI_Init + MPI_Finalize only (launch overhead)
 * Environment: BUDGET_S = time budget in seconds for ONE timed multiplication (rank 0 exits with 3).
 */
#define _POSIX_C_SOURCE 200809L
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>

#define HEADER "method,n,P,rep,total_s,scatter_s,bcast_s,compute_s,gather_s,comp_max_s,comp_min_s," \
               "init_s,load_s,checksum,rss_root_kb,rss_worker_max_kb,rss_sum_kb"

/* ---------------- kernel (unchanged from Assignment 3) ---------------- */

static inline void row_ikj(const double *a, const double *b, double *c, int n, int i) {
    double *ci = c + (size_t) i * n;
    const double *ai = a + (size_t) i * n;
    memset(ci, 0, (size_t) n * sizeof(double));
    for (int k = 0; k < n; k++) {
        const double aik = ai[k];
        const double *bk = b + (size_t) k * n;
        for (int j = 0; j < n; j++) ci[j] += aik * bk[j];
    }
}

static void mm_seq(const double *a, const double *b, double *c, int n) {       /* reference only */
    for (int i = 0; i < n; i++) row_ikj(a, b, c, n, i);
}

static void mm_naive(const double *a, const double *b, double *c, int n) {     /* reference only */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += a[(size_t) i * n + k] * b[(size_t) k * n + j];
            c[(size_t) i * n + j] = s;
        }
}

/* ---------------- distribution ---------------- */

typedef struct {
    int n, P, rank, rows, *cnt, *dsp;   /* cnt/dsp in elements (rows * n), used by rank 0 */
    double *aloc, *cloc, *b;            /* local band of A, local band of C, full B on every rank */
} Dist;

static void dist_init(Dist *d, int n, double *b_root) {
    MPI_Comm_size(MPI_COMM_WORLD, &d->P);
    MPI_Comm_rank(MPI_COMM_WORLD, &d->rank);
    d->n = n;
    d->cnt = malloc(d->P * sizeof(int));
    d->dsp = malloc(d->P * sizeof(int));
    int off = 0;
    for (int r = 0; r < d->P; r++) {
        int m = n / d->P + (r < n % d->P ? 1 : 0);
        d->cnt[r] = m * n;
        d->dsp[r] = off * n;
        off += m;
    }
    d->rows = d->cnt[d->rank] / (n ? n : 1);
    if (d->rank == 0) { d->aloc = d->cloc = NULL; d->b = b_root; }
    else {
        d->aloc = malloc((size_t) d->cnt[d->rank] * sizeof(double) + 8);
        d->cloc = malloc((size_t) d->cnt[d->rank] * sizeof(double) + 8);
        d->b = malloc((size_t) n * n * sizeof(double));
    }
}

static void dist_free(Dist *d) {
    if (d->rank) { free(d->aloc); free(d->cloc); free(d->b); }
    free(d->cnt); free(d->dsp);
}

/* One distributed multiplication C = A * B. a, c significant on rank 0 only. ph[0..4]: phase times. */
static void mm_dist(Dist *d, const double *a, double *c, double ph[5], double *comp_local) {
    const int root = d->rank == 0, n = d->n;
    double t0, t1, t2, t3, t4, c0, c1;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    if (root) MPI_Scatterv(a, d->cnt, d->dsp, MPI_DOUBLE, MPI_IN_PLACE, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else      MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, d->aloc, d->cnt[d->rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    MPI_Bcast(d->b, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    const double *ab = root ? a : d->aloc;          /* rank 0 band starts at row 0 */
    double *cb = root ? c : d->cloc;
    c0 = MPI_Wtime();
    for (int i = 0; i < d->rows; i++) row_ikj(ab, d->b, cb, n, i);
    c1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    t3 = MPI_Wtime();
    if (root) MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, c, d->cnt, d->dsp, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else      MPI_Gatherv(d->cloc, d->cnt[d->rank], MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    t4 = MPI_Wtime();
    ph[0] = t4 - t0; ph[1] = t1 - t0; ph[2] = t2 - t1; ph[3] = t3 - t2; ph[4] = t4 - t3;
    *comp_local = c1 - c0;
}

/* ---------------- helpers ---------------- */

static uint64_t sm_state;                  /* splitmix64, test inputs only (same generator as A2-A3) */
static uint64_t sm_next(void) {
    uint64_t z = (sm_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void gen(double *x, long len, uint64_t seed) {
    sm_state = seed;
    for (long i = 0; i < len; i++) x[i] = 2.0 * ((sm_next() >> 11) * 0x1.0p-53) - 1.0;
}

static double *load(const char *path, int n) {
    double *m = malloc((size_t) n * n * sizeof(double));
    FILE *f = fopen(path, "rb");
    if (!m || !f || fread(m, sizeof(double), (size_t) n * n, f) != (size_t) n * n) {
        fprintf(stderr, "cannot read %s\n", path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fclose(f);
    return m;
}

static long peak_rss_kb(void) {            /* VmHWM = peak resident set size of this process */
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    long kb = -1;
    while (f && fgets(line, sizeof line, f))
        if (strncmp(line, "VmHWM:", 6) == 0) { kb = strtol(line + 6, NULL, 10); break; }
    if (f) fclose(f);
    return kb;
}

static double checksum(const double *c, long len) {
    double s = 0.0;
    for (long i = 0; i < len; i++) s += c[i];
    return s;
}

static double mono(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void on_alarm(int sig) { (void) sig; static const char m[] = "TIMEOUT\n"; if (write(2, m, sizeof m - 1) < 0) _exit(4); _exit(3); }

static double maxdiff(const double *x, const double *ref, long len, double atol, double rtol, int *ok) {
    double worst = 0.0;
    *ok = 1;
    for (long i = 0; i < len; i++) {
        double e = fabs(x[i] - ref[i]);
        if (e > atol + rtol * fabs(ref[i])) *ok = 0;
        if (e > worst) worst = e;
    }
    return worst;
}

/* ---------------- correctness tests (all ranks take part, rank 0 checks and prints) ---------------- */

static int tests(void) {
    const double ATOL = 1e-12, RTOL = 1e-12;
    int rank, P, all = 1, ok;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    double ph[5], cl;
    if (rank == 0) printf("test,n,P,max_abs_diff,pass\n");
    int sizes[] = {2, 1, 7, 67, 130, 257};      /* n = 1 < P; sizes not divisible by P */
    for (int s = 0; s < 6; s++) {
        int n = sizes[s];
        long len = (long) n * n;
        double *a = NULL, *b = NULL, *id = NULL, *z = NULL, *ref = NULL, *cs = NULL, *c = NULL;
        if (rank == 0) {
            a = malloc(len * 8); b = malloc(len * 8); id = calloc(len, 8); z = calloc(len, 8);
            ref = malloc(len * 8); cs = malloc(len * 8); c = malloc(len * 8);
            if (n == 2) { double a2[] = {1, 2, 3, 4}, b2[] = {5, 6, 7, 8}, e2[] = {19, 22, 43, 50};
                          memcpy(a, a2, 32); memcpy(b, b2, 32); memcpy(ref, e2, 32); }
            else { gen(a, len, 11 + n); gen(b, len, 99 + n); mm_naive(a, b, ref, n); }
            for (int i = 0; i < n; i++) id[(size_t) i * n + i] = 1.0;
            mm_seq(a, b, cs, n);
        }
        Dist d;
        /* A * B */
        dist_init(&d, n, b);
        mm_dist(&d, a, c, ph, &cl);
        if (rank == 0) {
            double e = maxdiff(c, ref, len, n == 2 ? 0 : ATOL, n == 2 ? 0 : RTOL, &ok);
            printf("%s,%d,%d,%g,%d\n", n == 2 ? "known_2x2" : "dist_vs_naive_tol", n, P, e, ok); all &= ok;
            e = maxdiff(c, cs, len, 0, 0, &ok);
            printf("dist_equals_seq_bitwise,%d,%d,%g,%d\n", n, P, e, ok); all &= ok;
        }
        dist_free(&d);
        /* A * I = A */
        dist_init(&d, n, id);
        mm_dist(&d, a, c, ph, &cl);
        if (rank == 0) { double e = maxdiff(c, a, len, 0, 0, &ok); printf("A*I=A,%d,%d,%g,%d\n", n, P, e, ok); all &= ok; }
        dist_free(&d);
        /* A * 0 = 0, with garbage in C and in the local buffers */
        dist_init(&d, n, z);
        if (rank == 0) for (long x = 0; x < len; x++) c[x] = 7.0;
        else for (long x = 0; x < d.cnt[rank]; x++) d.cloc[x] = 7.0;
        mm_dist(&d, a, c, ph, &cl);
        if (rank == 0) { double e = maxdiff(c, z, len, 0, 0, &ok); printf("A*0=0,%d,%d,%g,%d\n", n, P, e, ok); all &= ok; }
        dist_free(&d);
        free(a); free(b); free(id); free(z); free(ref); free(cs); free(c);
    }
    if (rank == 0) fprintf(stderr, all ? "ALL TESTS PASSED (P=%d)\n" : "SOME TESTS FAILED (P=%d)\n", P);
    return all ? 0 : 1;
}

/* ---------------- main ---------------- */

int main(int argc, char **argv) {
    double i0 = mono();
    MPI_Init(&argc, &argv);
    double init_s = mono() - i0;
    int rank, P, rc = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    if (argc >= 2 && !strcmp(argv[1], "init")) { MPI_Finalize(); return 0; }
    if (argc >= 2 && !strcmp(argv[1], "test")) { rc = tests(); MPI_Finalize(); return rc; }
    if (argc < 6 || strcmp(argv[1], "bench")) {
        if (rank == 0) fprintf(stderr, "usage: matmul4 bench n warmup reps datadir | test | init\n");
        MPI_Finalize();
        return 1;
    }
    int n = atoi(argv[2]), warmup = atoi(argv[3]), reps = atoi(argv[4]);
    unsigned budget = getenv("BUDGET_S") ? (unsigned) atoi(getenv("BUDGET_S")) : 0;
    if (rank == 0 && budget) signal(SIGALRM, on_alarm);

    double *a = NULL, *b = NULL, *c = NULL, load_s = 0;
    if (rank == 0) {                                       /* inputs live on rank 0 only */
        char pa[1024], pb[1024];
        snprintf(pa, sizeof pa, "%s/A_%d.bin", argv[5], n);
        snprintf(pb, sizeof pb, "%s/B_%d.bin", argv[5], n);
        double l0 = mono();
        a = load(pa, n); b = load(pb, n);
        load_s = mono() - l0;
        c = calloc((size_t) n * n, sizeof(double));
        printf(HEADER "\n");
    }
    Dist d;
    dist_init(&d, n, b);
    double ph[5], cl, cmax, cmin;
    long rss_all[64];
    for (int r = -warmup; r < reps; r++) {
        if (rank == 0 && budget) alarm(budget);
        mm_dist(&d, a, c, ph, &cl);
        if (rank == 0 && budget) alarm(0);
        MPI_Reduce(&cl, &cmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&cl, &cmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        long rss = peak_rss_kb();
        MPI_Gather(&rss, 1, MPI_LONG, rss_all, 1, MPI_LONG, 0, MPI_COMM_WORLD);
        if (rank == 0 && r >= 0) {
            long wmax = 0, sum = 0;
            for (int q = 0; q < P && q < 64; q++) { sum += rss_all[q]; if (q && rss_all[q] > wmax) wmax = rss_all[q]; }
            printf("mpi,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.6f,%.6f,%.12e,%ld,%ld,%ld\n",
                   n, P, r, ph[0], ph[1], ph[2], ph[3], ph[4], cmax, cmin, init_s, load_s,
                   checksum(c, (long) n * n), rss_all[0], wmax, sum);
            fflush(stdout);
        }
    }
    dist_free(&d);
    free(a); free(b); free(c);
    MPI_Finalize();
    return 0;
}
