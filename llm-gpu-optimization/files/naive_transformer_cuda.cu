/* naive_transformer_cuda.cu — the CUDA rung-1 counterpart of
 * naive_transformer_bench.c.  SAME model, SAME PRNG, SAME weight fill
 * order: for the default config the logits must match your CPU run
 * (0.095916, 0.033000) to reduction-order tolerance (~1e-4 relative;
 * GPU sums in a different order).
 *
 * Deliberately naive: no cuBLAS, no tensor cores, no shared-memory
 * tiling.  Each projection is one thread per output element doing a
 * bare dot product; attention is one block per (head, token) with a
 * single-thread softmax.  This is the honest GPU baseline the
 * optimization ladder starts from.
 *
 * Build (RTX 3080 Ti = sm_86):
 *   nvcc -O3 -arch=sm_86 -o bench_cu naive_transformer_cuda.cu
 * Run:
 *   ./bench_cu
 * For stable reps, lock clocks first:
 *   sudo nvidia-smi -lgc 1695 && sudo nvidia-smi -lmc 9501   (3080 Ti)
 *
 * Same knobs as the C version: -DDIM -DLAYERS -DHEADS -DSEQ -DFFMULT
 * -DITERS (default 20) -DREPS (default 3).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

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
#define ITERS 20
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

/* ---------------- kernels (naive on purpose) ---------------- */

/* one block per token; two-accumulator single-pass stats */
__global__ void k_layernorm(const float *x, const float *g, const float *b,
                            float *y)
{
    int t = blockIdx.x;
    const float *xi = x + (long)t * D;
    float *yo = y + (long)t * D;
    __shared__ float ss[256], sq[256];
    float s = 0.f, q = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = xi[i];
        s += v; q += v * v;
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

/* Y[t][i] = sum_j W[i][j] X[t][j] ; one thread per (t,i), bare dot */
__global__ void k_matmul(int od, int id, const float *__restrict__ W,
                         const float *__restrict__ X, float *__restrict__ Y,
                         int T)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)T * od) return;
    int t = idx / od, i = idx % od;
    const float *xi = X + (long)t * id;
    const float *w  = W + (long)i * id;
    float acc = 0.f;
    for (int j = 0; j < id; j++) acc += w[j] * xi[j];
    Y[idx] = acc;
}

/* one block per (head, token): scores, softmax (single thread!), AV */
__global__ void k_attn(const float *__restrict__ q,
                       const float *__restrict__ k,
                       const float *__restrict__ v, float *__restrict__ ao)
{
    int hh = blockIdx.x, t = blockIdx.y, o = hh * HD;
    __shared__ float sc[SEQ];
    float scale = rsqrtf((float)HD);
    for (int s = threadIdx.x; s <= t; s += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < HD; j++)
            acc += q[(long)t*D + o + j] * k[(long)s*D + o + j];
        sc[s] = acc * scale;
    }
    __syncthreads();
    if (threadIdx.x == 0) {                       /* naive softmax */
        float mx = sc[0];
        for (int s = 1; s <= t; s++) mx = fmaxf(mx, sc[s]);
        float sum = 0.f;
        for (int s = 0; s <= t; s++) { sc[s] = expf(sc[s]-mx); sum += sc[s]; }
        float inv = 1.f / sum;
        for (int s = 0; s <= t; s++) sc[s] *= inv;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < HD; j += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s <= t; s++)
            acc += sc[s] * v[(long)s*D + o + j];
        ao[(long)t*D + o + j] = acc;
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

/* ---------------- host side ---------------- */
struct Layer {
    float *wq, *wk, *wv, *wo, *w1, *b1, *w2, *b2, *g1, *bt1, *g2, *bt2;
};
static Layer Ls[LAYERS];
static float *d_wout, *d_gf, *d_bf;
static float *d_x, *d_xn, *d_q, *d_k, *d_v, *d_ao, *d_h, *d_tmp, *d_in, *d_lg;

/* PRNG + fill: BYTE-IDENTICAL to the CPU benchmark */
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

static void forward(void)
{
    long nD = (long)SEQ*D, nF = (long)SEQ*FF;
    int TB = 256;
    CK(cudaMemcpy(d_x, d_in, nD*sizeof(float), cudaMemcpyDeviceToDevice));
    for (int l = 0; l < LAYERS; l++) {
        Layer *L = &Ls[l];
        k_layernorm<<<SEQ,256>>>(d_x, L->g1, L->bt1, d_xn);
        k_matmul<<<(nD+TB-1)/TB,TB>>>(D, D, L->wq, d_xn, d_q, SEQ);
        k_matmul<<<(nD+TB-1)/TB,TB>>>(D, D, L->wk, d_xn, d_k, SEQ);
        k_matmul<<<(nD+TB-1)/TB,TB>>>(D, D, L->wv, d_xn, d_v, SEQ);
        dim3 g(HEADS, SEQ);
        k_attn<<<g,128>>>(d_q, d_k, d_v, d_ao);
        k_matmul<<<(nD+TB-1)/TB,TB>>>(D, D, L->wo, d_ao, d_tmp, SEQ);
        k_residual<<<(nD+TB-1)/TB,TB>>>(d_x, d_tmp, nD);
        k_layernorm<<<SEQ,256>>>(d_x, L->g2, L->bt2, d_xn);
        k_matmul<<<(nF+TB-1)/TB,TB>>>(FF, D, L->w1, d_xn, d_h, SEQ);
        k_bias_relu<<<(nF+TB-1)/TB,TB>>>(d_h, L->b1, nF, FF);
        k_matmul<<<(nD+TB-1)/TB,TB>>>(D, FF, L->w2, d_h, d_tmp, SEQ);
        k_residual_bias<<<(nD+TB-1)/TB,TB>>>(d_x, d_tmp, L->b2, nD, D);
    }
    k_layernorm<<<1,256>>>(d_x + (long)(SEQ-1)*D, d_gf, d_bf, d_xn);
    k_matmul<<<1,64>>>(NOUT, D, d_wout, d_xn, d_lg, 1);
}

int main(void)
{
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    float ws = 0.6f / sqrtf((float)D);
    for (int l = 0; l < LAYERS; l++) {                /* SAME ORDER as CPU */
        Layer *L = &Ls[l];
        L->wq = to_dev(host_fill((long)D*D,  ws), (long)D*D);
        L->wk = to_dev(host_fill((long)D*D,  ws), (long)D*D);
        L->wv = to_dev(host_fill((long)D*D,  ws), (long)D*D);
        L->wo = to_dev(host_fill((long)D*D,  ws), (long)D*D);
        L->w1 = to_dev(host_fill((long)FF*D, ws), (long)FF*D);
        L->b1 = to_dev(host_fill(FF, 0.01f), FF);
        L->w2 = to_dev(host_fill((long)D*FF, ws*0.5f), (long)D*FF);
        L->b2 = to_dev(host_fill(D, 0.01f), D);
        float *g1 = (float*)malloc(D*4), *bt1 = (float*)malloc(D*4);
        float *g2 = (float*)malloc(D*4), *bt2 = (float*)malloc(D*4);
        for (int i = 0; i < D; i++) {
            g1[i] = 1.0f + fr(0.1f); bt1[i] = fr(0.05f);
            g2[i] = 1.0f + fr(0.1f); bt2[i] = fr(0.05f);
        }
        L->g1  = to_dev(g1, D); L->bt1 = to_dev(bt1, D);
        L->g2  = to_dev(g2, D); L->bt2 = to_dev(bt2, D);
    }
    d_wout = to_dev(host_fill((long)NOUT*D, ws), (long)NOUT*D);
    {
        float *g = (float*)malloc(D*4), *b = (float*)malloc(D*4);
        for (int i = 0; i < D; i++) { g[i] = 1.0f+fr(0.1f); b[i] = fr(0.05f); }
        d_gf = to_dev(g, D); d_bf = to_dev(b, D);
    }
    d_in = to_dev(host_fill((long)SEQ*D, 0.5f), (long)SEQ*D);

    CK(cudaMalloc(&d_x,  (long)SEQ*D*4)); CK(cudaMalloc(&d_xn, (long)SEQ*D*4));
    CK(cudaMalloc(&d_q,  (long)SEQ*D*4)); CK(cudaMalloc(&d_k,  (long)SEQ*D*4));
    CK(cudaMalloc(&d_v,  (long)SEQ*D*4)); CK(cudaMalloc(&d_ao, (long)SEQ*D*4));
    CK(cudaMalloc(&d_h,  (long)SEQ*FF*4)); CK(cudaMalloc(&d_tmp,(long)SEQ*D*4));
    CK(cudaMalloc(&d_lg, NOUT*4));

    double params = (double)LAYERS*(4.0*D*D + 2.0*D*FF) + NOUT*(double)D;
    double macs_p = (double)LAYERS*SEQ*(4.0*D*D + 2.0*D*FF);
    double macs_a = (double)LAYERS*HEADS*(double)SEQ*(SEQ+1)*HD;
    double macs   = macs_p + macs_a;

    forward(); CK(cudaDeviceSynchronize());          /* warm + verify */
    float lg[NOUT];
    CK(cudaMemcpy(lg, d_lg, NOUT*4, cudaMemcpyDeviceToHost));
    printf("device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("model: D=%d L=%d H=%d SEQ=%d FF=%d | params %.1fM (%.0f MB fp32)\n",
           D, LAYERS, HEADS, SEQ, FF, params/1e6, params*4/1e6);
    printf("work: %.2f GMAC/pass | logits[0..1] = %.6f %.6f  "
           "(CPU ref for defaults: 0.095916 0.033000; expect ~1e-4 agreement)\n",
           macs/1e9, lg[0], lg[1]);

    for (int r = 0; r < REPS; r++) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        for (int i = 0; i < ITERS; i++) forward();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        double t = ms/1e3/ITERS;
        printf("rep %d: %.3f ms/pass  %.1f GFLOP/s  (%.1f us/token, "
               "weight-set/pass = %.1f GB/s)\n",
               r, t*1e3, 2*macs/t/1e9, t*1e6/SEQ, params*4/t/1e9);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    return 0;
}
