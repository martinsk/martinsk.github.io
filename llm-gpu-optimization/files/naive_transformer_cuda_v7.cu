/* naive_transformer_cuda_v7.cu — RUNG 7: parallel attention on the cuBLAS base.
 * Not a rung: a pricing instrument.  Answers "what does NVIDIA's full
 * tactic library (double buffering, float4, autotuned tiles) get on
 * exactly our shapes?" so we know the ceiling before hand-building
 * rung 7.  Build:
 *   nvcc -O3 -arch=sm_86 -DPROFILE=1 -o bench_oracle \
 *        naive_transformer_cuda_oracle.cu -lcublas          (fp32)
 *   add -DTF32 for tensor-core TF32 (logits gate loosens to ~1e-3:
 *   TF32 rounds inputs to 10-bit mantissas -- measure the drift!)
 *
 * Changes vs v3 (both driven by the measured phase table: GEMMs = 87.7%):
 *  1. Register tiling: each thread computes a 4x4 micro-tile from
 *     register-cached operands: 16 FMA per 8 smem loads (v3: 1 per 2).
 *     Block tile 32 tokens x 64 outputs, 128 threads, smem padded
 *     against bank conflicts.
 *  2. qkv fused into ONE GEMM (od=3D): same weights, same PRNG fill
 *     order (fill q,k,v separately, then interleave columns), so the
 *     logits gate still applies: 0.095916 0.033000 for defaults.
 *
 * Build:  nvcc -O3 -arch=sm_86 [-DPROFILE=1] -o bench_cu4 naive_transformer_cuda_v4.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
#define ITERS 50
#endif
#ifndef REPS
#define REPS 3
#endif

#define D  DIM
#define FF (FFMULT*DIM)
#define HD (DIM/HEADS)
#define NOUT 16

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", \
            cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

/* ------------- register-tiled GEMM + SPLIT-K --------------------------
 * Rung 6.  Tactic B (more, smaller blocks) was FALSIFIED on ada: w2 went
 * 3.66 -> 4.03 ms.  Diagnosis: w2 and w1 have identical FLOPs (604M) but
 * run 150 vs 558 GF/s -- the narrow shapes expose only 48 blocks x 4
 * warps = 2.4 warps/SM, far below latency-hiding needs, and no tile size
 * in the (od x T) plane can create more parallelism: od*T is fixed.
 * The missing parallelism lives on the k axis: SPLIT-K.  Narrow GEMMs
 * split the reduction across gridDim.z slices writing partial sums,
 * followed by a fixed-order reduce (deterministic -- no atomics).      */
#define BM 32
#define BN 64
#define BK 16
#define TM 4
#define TN 4
__global__ void k_gemm_rt(int od, int id, const float *__restrict__ WT,
                          const float *__restrict__ X, float *__restrict__ Y,
                          int T, float *__restrict__ P)
{
    __shared__ float Xs[BM][BK + 1];
    __shared__ float Ws[BK][BN];
    int bx = blockIdx.x * BN, by = blockIdx.y * BM;
    int S = gridDim.z, z = blockIdx.z;
    int kb = (id + S - 1) / S;
    int ks = z * kb, ke = ks + kb < id ? ks + kb : id;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * 16 + tx;
    float acc[TM][TN] = {};
    for (int k0 = ks; k0 < ke; k0 += BK) {
        for (int e = tid; e < BM*BK; e += 128) {
            int r = e / BK, c = e % BK;
            Xs[r][c] = (by + r < T && k0 + c < ke)
                       ? X[(long)(by + r)*id + k0 + c] : 0.f;
        }
        for (int e = tid; e < BK*BN; e += 128) {
            int r = e / BN, c = e % BN;
            Ws[r][c] = (k0 + r < ke && bx + c < od)
                       ? WT[(long)(k0 + r)*od + bx + c] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float xr[TM], wr[TN];
            #pragma unroll
            for (int m = 0; m < TM; m++) xr[m] = Xs[ty*TM + m][kk];
            #pragma unroll
            for (int n = 0; n < TN; n++) wr[n] = Ws[kk][tx*TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    acc[m][n] += xr[m] * wr[n];
        }
        __syncthreads();
    }
    float *out = (S == 1) ? Y : P + (long)z * T * od;
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int t = by + ty*TM + m;
        if (t >= T) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int i = bx + tx*TN + n;
            if (i < od) out[(long)t*od + i] = acc[m][n];
        }
    }
}
/* fixed-order reduction of split-K partials: deterministic */
__global__ void k_reduce(float *__restrict__ Y, const float *__restrict__ P,
                         int S, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int z = 0; z < S; z++) acc += P[(long)z*n + i];
    Y[i] = acc;
}

#define MAX_SPLIT 8
static float *d_part;                    /* [MAX_SPLIT][SEQ][max od<1024] */

static cublasHandle_t CBH;
static void mm(int od, int id, const float *WT, const float *X, float *Y,
               int T)
{
    /* row-major Y[T][od] = X[T][id] * WT[id][od]  ==  column-major
     * Y^T = WT^T * X^T, which cuBLAS sees as a plain (od x T) GEMM: */
    const float a = 1.f, b = 0.f;
#ifdef TF32
    cublasComputeType_t ct = CUBLAS_COMPUTE_32F_FAST_TF32;
#else
    cublasComputeType_t ct = CUBLAS_COMPUTE_32F;
#endif
    cublasGemmEx(CBH, CUBLAS_OP_N, CUBLAS_OP_N, od, T, id,
                 &a, WT, CUDA_R_32F, od, X, CUDA_R_32F, id,
                 &b, Y,  CUDA_R_32F, od, ct, CUBLAS_GEMM_DEFAULT);
}

/* ------------- other kernels (unchanged from v3) ------------- */
__global__ void k_layernorm(const float *x, const float *g, const float *b,
                            float *y)
{
    int t = blockIdx.x;
    const float *xi = x + (long)t * D;
    float *yo = y + (long)t * D;
    __shared__ float ss[256], sq[256];
    float s = 0.f, q = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = xi[i]; s += v; q += v * v;
    }
    ss[threadIdx.x] = s; sq[threadIdx.x] = q;
    __syncthreads();
    for (int w = blockDim.x >> 1; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            ss[threadIdx.x] += ss[threadIdx.x + w];
            sq[threadIdx.x] += sq[threadIdx.x + w];
        }
        __syncthreads();
    }
    float mean = ss[0] / D;
    float var  = sq[0] / D - mean * mean;
    float rstd = rsqrtf(var + 1e-5f);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        yo[i] = (xi[i] - mean) * rstd * g[i] + b[i];
}

/* RUNG 7 attention: everything the rung-1 kernel did wrong, fixed.
 *  - score dots: one WARP per source position s, 32 lanes across the
 *    head dim -> coalesced k-reads + shuffle reduction (was: one thread
 *    per s walking 64 strided floats)
 *  - softmax: parallel block max/sum via warp shuffles (was: literally
 *    single-threaded)
 *  - q slice staged in shared memory once
 *  - value accumulation was already coalesced; normalization folded in.
 * All reductions are fixed-order -> deterministic; grouping differs
 * from serial so logits may drift ~1e-6.                              */
__global__ void k_attn(const float *__restrict__ qkv, float *__restrict__ ao)
{
    int hh = blockIdx.x, t = blockIdx.y, o = hh * HD;
    const long R = 3L * D;
    __shared__ float sc[SEQ], qs[HD], rmax[4], rsum[4];
    int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    float scale = rsqrtf((float)HD);

    for (int j = tid; j < HD; j += blockDim.x)
        qs[j] = qkv[(long)t*R + o + j];
    __syncthreads();

    for (int s = w; s <= t; s += 4) {                 /* warp per s */
        float acc = 0.f;
        for (int j = lane; j < HD; j += 32)           /* coalesced   */
            acc += qs[j] * qkv[(long)s*R + D + o + j];
        for (int off = 16; off; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) sc[s] = acc * scale;
    }
    __syncthreads();

    float mx = -3.4e38f;
    for (int s = tid; s <= t; s += blockDim.x) mx = fmaxf(mx, sc[s]);
    for (int off = 16; off; off >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
    if (lane == 0) rmax[w] = mx;
    __syncthreads();
    if (tid == 0) {
        float m = rmax[0];
        for (int i = 1; i < 4; i++) m = fmaxf(m, rmax[i]);
        rmax[0] = m;
    }
    __syncthreads();
    mx = rmax[0];

    float sum = 0.f;
    for (int s = tid; s <= t; s += blockDim.x) {
        float e = expf(sc[s] - mx);
        sc[s] = e; sum += e;
    }
    for (int off = 16; off; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) rsum[w] = sum;
    __syncthreads();
    if (tid == 0) rsum[0] = rsum[0] + rsum[1] + rsum[2] + rsum[3];
    __syncthreads();
    float inv = 1.f / rsum[0];

    for (int j = tid; j < HD; j += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s <= t; s++)
            acc += sc[s] * qkv[(long)s*R + 2*D + o + j];
        ao[(long)t*D + o + j] = acc * inv;
    }
}

__global__ void k_bias_relu(float *h, const float *b, long n, int width)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float z = h[i] + b[i % width];
    h[i] = z > 0.f ? z : 0.f;
}
__global__ void k_residual(float *x, const float *t, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += t[i];
}
__global__ void k_residual_bias(float *x, const float *t, const float *b,
                                long n, int width)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += t[i] + b[i % width];
}

/* ------------- host ------------- */
struct Layer {
    float *wqkv;                       /* fused, transposed: [D][3D] */
    float *wo, *w1, *b1, *w2, *b2, *g1, *bt1, *g2, *bt2;
};
static Layer Ls[LAYERS];
static float *d_wout, *d_gf, *d_bf;
static float *d_x, *d_xn, *d_qkv, *d_ao, *d_h, *d_tmp, *d_in, *d_lg;

static unsigned long long rng = 0x243F6A8885A308D3ull;
static float fr(float s)
{
    rng = rng*6364136223846793005ull + 1442695040888963407ull;
    return s * (((rng>>33)&0xFFFFFF)/8388608.0f - 1.0f);
}
static float *host_fill(long n, float s)
{
    float *p = (float*)malloc(n*sizeof(float));
    for (long i = 0; i < n; i++) p[i] = fr(s);
    return p;
}
static float *to_dev(float *h, long n)
{
    float *d; CK(cudaMalloc(&d, n*sizeof(float)));
    CK(cudaMemcpy(d, h, n*sizeof(float), cudaMemcpyHostToDevice));
    free(h);
    return d;
}
static float *to_dev_T(float *h, int od, int id)
{
    float *t = (float*)malloc((long)od*id*sizeof(float));
    for (int i = 0; i < od; i++)
        for (int j = 0; j < id; j++)
            t[(long)j*od + i] = h[(long)i*id + j];
    free(h);
    return to_dev(t, (long)od*id);
}
/* fill q,k,v in the ORIGINAL PRNG order, then pack transposed side by
 * side into one [D][3D] matrix: same numbers, fused layout. */
static float *to_dev_qkvT(float s)
{
    float *q = host_fill((long)D*D, s);
    float *k = host_fill((long)D*D, s);
    float *v = host_fill((long)D*D, s);
    float *t = (float*)malloc(3L*D*D*sizeof(float));
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) {
            t[(long)j*3*D +       i] = q[(long)i*D + j];
            t[(long)j*3*D + D   + i] = k[(long)i*D + j];
            t[(long)j*3*D + 2*D + i] = v[(long)i*D + j];
        }
    free(q); free(k); free(v);
    return to_dev(t, 3L*D*D);
}

#ifdef PROFILE
static double pnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }
static double ph[8];
#define P0 double _pt = (cudaDeviceSynchronize(), pnow());
#define P(n) do { cudaDeviceSynchronize(); double _q = pnow(); \
                  ph[n] += _q - _pt; _pt = _q; } while (0)
#else
#define P0
#define P(n)
#endif

static void forward(void)
{
    long nD = (long)SEQ*D, nF = (long)SEQ*FF;
    int TB = 256;
    CK(cudaMemcpy(d_x, d_in, nD*sizeof(float), cudaMemcpyDeviceToDevice));
    P0;
    for (int l = 0; l < LAYERS; l++) {
        Layer *L = &Ls[l];
        k_layernorm<<<SEQ,256>>>(d_x, L->g1, L->bt1, d_xn);
        P(0);
        mm(3*D, D, L->wqkv, d_xn, d_qkv, SEQ);
        P(1);
        dim3 g(HEADS, SEQ);
        k_attn<<<g,128>>>(d_qkv, d_ao);
        P(2);
        mm(D, D, L->wo, d_ao, d_tmp, SEQ);
        k_residual<<<(nD+TB-1)/TB,TB>>>(d_x, d_tmp, nD);
        P(3);
        k_layernorm<<<SEQ,256>>>(d_x, L->g2, L->bt2, d_xn);
        P(0);
        mm(FF, D, L->w1, d_xn, d_h, SEQ);
        P(4);
        k_bias_relu<<<(nF+TB-1)/TB,TB>>>(d_h, L->b1, nF, FF);
        P(5);
        mm(D, FF, L->w2, d_h, d_tmp, SEQ);
        P(6);
        k_residual_bias<<<(nD+TB-1)/TB,TB>>>(d_x, d_tmp, L->b2, nD, D);
        P(3);
    }
    k_layernorm<<<1,256>>>(d_x + (long)(SEQ-1)*D, d_gf, d_bf, d_xn);
    mm(NOUT, D, d_wout, d_xn, d_lg, 1);
    P(7);
}

int main(void)
{
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    float ws = 0.6f / sqrtf((float)D);
    for (int l = 0; l < LAYERS; l++) {              /* SAME PRNG ORDER */
        Layer *L = &Ls[l];
        L->wqkv = to_dev_qkvT(ws);
        L->wo   = to_dev_T(host_fill((long)D*D,  ws), D, D);
        L->w1   = to_dev_T(host_fill((long)FF*D, ws), FF, D);
        L->b1   = to_dev(host_fill(FF, 0.01f), FF);
        L->w2   = to_dev_T(host_fill((long)D*FF, ws*0.5f), D, FF);
        L->b2   = to_dev(host_fill(D, 0.01f), D);
        float *g1 = (float*)malloc(D*4), *bt1 = (float*)malloc(D*4);
        float *g2 = (float*)malloc(D*4), *bt2 = (float*)malloc(D*4);
        for (int i = 0; i < D; i++) {
            g1[i] = 1.0f + fr(0.1f); bt1[i] = fr(0.05f);
            g2[i] = 1.0f + fr(0.1f); bt2[i] = fr(0.05f);
        }
        L->g1  = to_dev(g1, D); L->bt1 = to_dev(bt1, D);
        L->g2  = to_dev(g2, D); L->bt2 = to_dev(bt2, D);
    }
    d_wout = to_dev_T(host_fill((long)NOUT*D, ws), NOUT, D);
    {
        float *g = (float*)malloc(D*4), *b = (float*)malloc(D*4);
        for (int i = 0; i < D; i++) { g[i] = 1.0f+fr(0.1f); b[i] = fr(0.05f); }
        d_gf = to_dev(g, D); d_bf = to_dev(b, D);
    }
    d_in = to_dev(host_fill((long)SEQ*D, 0.5f), (long)SEQ*D);

    CK(cudaMalloc(&d_x,   (long)SEQ*D*4));
    CK(cudaMalloc(&d_xn,  (long)SEQ*D*4));
    CK(cudaMalloc(&d_qkv, 3L*SEQ*D*4));
    CK(cudaMalloc(&d_ao,  (long)SEQ*D*4));
    CK(cudaMalloc(&d_h,   (long)SEQ*FF*4));
    CK(cudaMalloc(&d_tmp, (long)SEQ*D*4));
    CK(cudaMalloc(&d_lg,  NOUT*4));
    CK(cudaMalloc(&d_part, (long)MAX_SPLIT*SEQ*1024*4));

    double params = (double)LAYERS*(4.0*D*D + 2.0*D*FF) + NOUT*(double)D;
    double macs_p = (double)LAYERS*SEQ*(4.0*D*D + 2.0*D*FF);
    double macs_a = (double)LAYERS*HEADS*(double)SEQ*(SEQ+1)*HD;
    double macs   = macs_p + macs_a;

    cublasCreate(&CBH);
#ifdef TF32
    printf("math mode: TF32 tensor cores (gate ~1e-3)\n");
#else
    printf("math mode: fp32 cuBLAS (gate ~1e-4)\n");
#endif
    forward(); forward(); CK(cudaDeviceSynchronize());   /* 2x: autotune warm */
    float lg[NOUT];
    CK(cudaMemcpy(lg, d_lg, NOUT*4, cudaMemcpyDeviceToHost));
    printf("device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("model: D=%d L=%d H=%d SEQ=%d FF=%d | params %.1fM (%.0f MB fp32)\n",
           D, LAYERS, HEADS, SEQ, FF, params/1e6, params*4/1e6);
    printf("work: %.2f GMAC/pass | logits[0..1] = %.6f %.6f  "
           "(ref: 0.095916 0.033000)\n", macs/1e9, lg[0], lg[1]);

#ifdef PROFILE
    {
        memset(ph, 0, sizeof ph);
        int N = 40;
        double t0 = pnow();
        for (int i = 0; i < N; i++) forward();
        cudaDeviceSynchronize();
        double wall = (pnow()-t0)/N;
        const char *nm[8] = {"layernorm","qkv-gemm","attention","wo+resid",
                             "w1-gemm","bias-relu","w2-gemm","head"};
        double s = 0;
        printf("\n--- phase breakdown (avg of %d passes, sync-timed) ---\n", N);
        for (int p = 0; p < 8; p++) {
            printf("  %-10s %7.3f ms  %5.1f%%\n", nm[p], ph[p]/N*1e3,
                   100.0*ph[p]/N/wall);
            s += ph[p]/N;
        }
        printf("  %-10s %7.3f ms  %5.1f%%\n", "other", (wall-s)*1e3,
               100.0*(wall-s)/wall);
        printf("  %-10s %7.3f ms\n", "wall/pass", wall*1e3);
    }
#endif
    for (int r = 0; r < REPS; r++) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        for (int i = 0; i < ITERS; i++) forward();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        double t = ms/1e3/ITERS;
        printf("rep %d: %.3f ms/pass  %.1f GFLOP/s  (%.1f us/token)\n",
               r, t*1e3, 2*macs/t/1e9, t*1e6/SEQ);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    return 0;
}
