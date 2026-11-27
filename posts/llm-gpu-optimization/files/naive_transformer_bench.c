/* naive_transformer_bench.c — deliberately naive transformer forward pass,
 * as a single-core CPU benchmark.  This is "v1" from our optimization
 * ladder, scaled to GPT-2-small class and given multi-head attention:
 * row-major dot-product matvecs, runtime layernorm, scalar libm expf,
 * per-token everything, no fusion, no algebra.  The point is a clean,
 * deterministic rung-1 baseline to measure machines and compilers against.
 *
 * Build ladder (compare them — the deltas are the experiment):
 *   cc -O2                                   -o bench naive_transformer_bench.c -lm
 *   cc -O3 -march=native                     -o bench naive_transformer_bench.c -lm
 *   cc -O3 -march=native -ffast-math         -o bench naive_transformer_bench.c -lm
 *   clang -O3 -march=native -ffast-math -mprefer-vector-width=512 \
 *                                            -o bench naive_transformer_bench.c -lm
 * Debug build (catches UB before it deletes your program):
 *   clang -O1 -g -fsanitize=address,undefined -o bench_dbg ...
 *
 * Knobs (all -D):
 *   DIM (768)  LAYERS (12)  HEADS (12)  SEQ (128)  FFMULT (4)
 *   ITERS (3)  REPS (3)
 * Regime sweep: weights = 12*DIM^2*4 bytes/layer.  Defaults ~340 MB
 * (DRAM-resident).  -DDIM=256 -DLAYERS=4 ≈ 12 MB (L3).  -DDIM=128
 * -DLAYERS=2 ≈ 1.6 MB (L2).  Watch GF/s collapse as the working set
 * falls out of each cache level — that is the roofline, live.
 *
 * Pin it:  taskset -c <isolated core> ./bench
 * The two printed logits are deterministic for a given DIM/LAYERS/SEQ:
 * use them to confirm different builds compute the same function.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef DIM
#define DIM 768
#endif
#ifndef LAYERS
#define LAYERS 12
#endif
#ifndef HEADS
#define HEADS 12
#endif
#ifndef SEQ
#define SEQ 128
#endif
#ifndef FFMULT
#define FFMULT 4
#endif
#ifndef ITERS
#define ITERS 3
#endif
#ifndef REPS
#define REPS 3
#endif

#define D  DIM
#define FF (FFMULT*DIM)
#define HD (DIM/HEADS)
#define NOUT 16

typedef struct {
    float *wq, *wk, *wv, *wo;      /* [D][D], row-major [out][in] */
    float *w1, *b1;                /* [FF][D], [FF] */
    float *w2, *b2;                /* [D][FF], [D]  */
    float *g1, *bt1, *g2, *bt2;    /* layernorm affine, [D] each   */
} Layer;
static Layer Ls[LAYERS];
static float *w_out, *gf, *bf;                 /* head: [NOUT][D] + ln_f */
static float (*x)[D], (*xn)[D], (*q)[D], (*k)[D], (*v)[D];
static float (*att)[SEQ], (*ao)[D], (*h)[FF];

/* ---------------- naive kernels (do not optimize these) -------------- */
static void matvec(int od, int id, const float *W,
                   const float *xin, float *y)
{
    for (int i = 0; i < od; i++) {
        float acc = 0.0f;
        for (int j = 0; j < id; j++)
            acc += W[(long)i*id + j] * xin[j];
        y[i] = acc;
    }
}
static void layernorm(const float *xi, const float *g, const float *b,
                      float *y)
{
    float m = 0.0f, va = 0.0f;
    for (int i = 0; i < D; i++) m += xi[i];
    m /= D;
    for (int i = 0; i < D; i++) { float d = xi[i]-m; va += d*d; }
    va /= D;
    float r = 1.0f / sqrtf(va + 1e-5f);
    for (int i = 0; i < D; i++) y[i] = (xi[i]-m)*r*g[i] + b[i];
}
static void softmax(float *a, int n)
{
    float mx = a[0];
    for (int i = 1; i < n; i++) if (a[i] > mx) mx = a[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { a[i] = expf(a[i]-mx); s += a[i]; }
    float inv = 1.0f/s;
    for (int i = 0; i < n; i++) a[i] *= inv;
}

static void layer_fwd(const Layer *L)
{
    const float scale = 1.0f / sqrtf((float)HD);
    for (int t = 0; t < SEQ; t++) {
        layernorm(x[t], L->g1, L->bt1, xn[t]);
        matvec(D, D, L->wq, xn[t], q[t]);
        matvec(D, D, L->wk, xn[t], k[t]);
        matvec(D, D, L->wv, xn[t], v[t]);
    }
    for (int hh = 0; hh < HEADS; hh++) {
        int o = hh * HD;
        for (int t = 0; t < SEQ; t++) {
            for (int s = 0; s <= t; s++) {
                float acc = 0.0f;
                for (int j = 0; j < HD; j++)
                    acc += q[t][o+j] * k[s][o+j];
                att[t][s] = acc * scale;
            }
            softmax(att[t], t+1);
            for (int j = 0; j < HD; j++) {
                float acc = 0.0f;
                for (int s = 0; s <= t; s++)
                    acc += att[t][s] * v[s][o+j];
                ao[t][o+j] = acc;
            }
        }
    }
    for (int t = 0; t < SEQ; t++) {
        float tmp[D];
        matvec(D, D, L->wo, ao[t], tmp);
        for (int j = 0; j < D; j++) x[t][j] += tmp[j];
    }
    for (int t = 0; t < SEQ; t++) {
        layernorm(x[t], L->g2, L->bt2, xn[t]);
        matvec(FF, D, L->w1, xn[t], h[t]);
        for (int i = 0; i < FF; i++) {
            float z = h[t][i] + L->b1[i];
            h[t][i] = z > 0.0f ? z : 0.0f;
        }
        float tmp[D];
        matvec(D, FF, L->w2, h[t], tmp);
        for (int j = 0; j < D; j++) x[t][j] += tmp[j] + L->b2[j];
    }
}

static void forward(const float *in, float *logits)
{
    memcpy(x, in, (long)SEQ*D*sizeof(float));
    for (int l = 0; l < LAYERS; l++) layer_fwd(&Ls[l]);
    float xf[D];
    layernorm(x[SEQ-1], gf, bf, xf);
    matvec(NOUT, D, w_out, xf, logits);
}

/* ---------------- harness ---------------- */
static unsigned long long rng = 0x243F6A8885A308D3ull;
static float fr(float s)
{
    rng = rng*6364136223846793005ull + 1442695040888963407ull;
    return s * (((rng>>33)&0xFFFFFF)/8388608.0f - 1.0f);
}
static float *al(long n)
{
    float *p;
    if (posix_memalign((void**)&p, 64, n*sizeof(float))) { perror("alloc"); exit(1); }
    return p;
}
static void fillv(float *p, long n, float s){ for (long i=0;i<n;i++) p[i]=fr(s); }
static double now(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

int main(void)
{
    float ws = 0.6f / sqrtf((float)D);       /* keep activations tame */
    for (int l = 0; l < LAYERS; l++) {
        Layer *L = &Ls[l];
        L->wq=al((long)D*D);  L->wk=al((long)D*D);
        L->wv=al((long)D*D);  L->wo=al((long)D*D);
        L->w1=al((long)FF*D); L->b1=al(FF);
        L->w2=al((long)D*FF); L->b2=al(D);
        L->g1=al(D); L->bt1=al(D); L->g2=al(D); L->bt2=al(D);
        fillv(L->wq,(long)D*D,ws);  fillv(L->wk,(long)D*D,ws);
        fillv(L->wv,(long)D*D,ws);  fillv(L->wo,(long)D*D,ws);
        fillv(L->w1,(long)FF*D,ws); fillv(L->b1,FF,0.01f);
        fillv(L->w2,(long)D*FF,ws*0.5f); fillv(L->b2,D,0.01f);
        for (int i = 0; i < D; i++) {
            L->g1[i]=1.0f+fr(0.1f); L->bt1[i]=fr(0.05f);
            L->g2[i]=1.0f+fr(0.1f); L->bt2[i]=fr(0.05f);
        }
    }
    w_out=al((long)NOUT*D); gf=al(D); bf=al(D);
    fillv(w_out,(long)NOUT*D,ws);
    for (int i = 0; i < D; i++){ gf[i]=1.0f+fr(0.1f); bf[i]=fr(0.05f); }

    x =(void*)al((long)SEQ*D);  xn=(void*)al((long)SEQ*D);
    q =(void*)al((long)SEQ*D);  k =(void*)al((long)SEQ*D);
    v =(void*)al((long)SEQ*D);  ao=(void*)al((long)SEQ*D);
    att=(void*)al((long)SEQ*SEQ);
    h =(void*)al((long)SEQ*FF);

    float *in = al((long)SEQ*D);
    fillv(in, (long)SEQ*D, 0.5f);
    float lg[NOUT];

    double params = (double)LAYERS*(4.0*D*D + 2.0*D*FF) + NOUT*(double)D;
    double macs_p = (double)LAYERS*SEQ*(4.0*D*D + 2.0*D*FF);         /* proj */
    double macs_a = (double)LAYERS*HEADS*(double)SEQ*(SEQ+1)*HD;     /* attn */
    double macs   = macs_p + macs_a;
    double wbytes = (double)LAYERS*SEQ*(4.0*D*D + 2.0*D*FF)*4.0;     /* naive:
                        weights re-streamed once per token per layer  */

    forward(in, lg);                                   /* warm + verify */
    printf("model: D=%d L=%d H=%d SEQ=%d FF=%d | params %.1fM (%.0f MB fp32)\n",
           D, LAYERS, HEADS, SEQ, FF, params/1e6, params*4/1e6);
    printf("work: %.2f GMAC/pass (%.1f%% attention) | naive weight stream %.2f GB/pass\n",
           macs/1e9, 100.0*macs_a/macs, wbytes/1e9);
    printf("logits[0..1] = %.6f %.6f   (determinism check)\n", lg[0], lg[1]);

    volatile float sink = 0;
    double best = 1e30;
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        for (int i = 0; i < ITERS; i++) {
            in[0] += sink * 1e-30f;
            forward(in, lg);
            sink = lg[0];
        }
        double t = (now()-t0)/ITERS;
        if (t < best) best = t;
        printf("rep %d: %.1f ms/pass  %.2f GFLOP/s  %.2f GB/s(weights)\n",
               r, t*1e3, 2*macs/t/1e9, wbytes/t/1e9);
    }
    printf("best: %.1f ms/pass  %.2f GFLOP/s  |  %.1f us/token\n",
           best*1e3, 2*macs/best/1e9, best*1e6/SEQ);
    return 0;
}
