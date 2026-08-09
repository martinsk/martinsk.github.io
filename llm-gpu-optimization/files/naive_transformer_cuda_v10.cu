/* naive_transformer_cuda_v10.cu — RUNG 10: the CPU campaign's algebra, ported.
 * On the v9 tensor-core base: (1) last-token pruning — the final layer
 * computes attention/MLP for ONE row, not 128 (exact: dead code);
 * (2) exact layernorm folding — W(g*x+b) = (W diag g)x + Wb, gammas into
 * weights, betas into fused biases, attention scale into q-columns;
 * (3) fused epilogues — bias/relu/residual live inside the GEMM store
 * or the split-K reduce; the separate elementwise kernels are deleted.
 * Gate: TF32 + fold regrouping => expect ~1e-3-5e-3 vs the fp32 ref
 * logits; run-to-run deterministic.
 * v8's skeleton with the register micro-tiles replaced by WMMA tf32
 * fragments.  The question: can hand-written tensor-core code beat
 * cuBLAS-TF32 (GEMM phases 2.117 ms) the way v8 beat cuBLAS-fp32?
 * Guard-hoisted interior GEMM + float4 + cp.async double buffering
 * (the three SASS-diagnosed fixes) + split-K + parallel attention.
 * Zero library calls.  The question it answers: can hand-written CUDA
 * match cuBLAS fp32 (GEMM phases 2.58 ms) on these shapes?
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

/* ------------- RUNG 8 GEMM: guard-hoisted + float4 + cp.async ---------
 * The SASS of rung 6 showed 256 FFMA vs 175 integer instructions (bounds
 * checks + addressing) = 53% issue density, and no copy/compute overlap.
 * Fixes, all plain CUDA C++:
 *   - INTERIOR kernel: all our real shapes are tile-multiples, so the
 *     hot path carries zero bounds predicates (legacy guarded kernel
 *     kept for the 16-wide head).
 *   - float4 staging: global tiles move as 16-byte LDGSTS/LDS.128.
 *   - cp.async double buffering: tile k+1 streams into the second smem
 *     buffer while tile k computes; one barrier per tile, not two.
 * X is staged TRANSPOSED in smem so both operands read as float4 rows.
 * Split-K retained unchanged.                                          */
#include <cuda_pipeline.h>
#define BM 32
#define BN 64
#define BK 16
#define TM 4
#define TN 4

#include <mma.h>
using namespace nvcuda;

/* RUNG 10 GEMM: v9's tensor-core kernel + FUSED EPILOGUES.
 * template EPI bits: 1 = add bias (od-periodic), 2 = relu.
 * EPI=0 keeps the direct wmma store (split-K partials path);
 * EPI!=0 stages the 32x64 result tile in smem (reusing the double
 * buffers: 2*16*64 floats = exactly 32*64) and writes out with
 * bias/relu applied — deleting the separate elementwise kernels
 * and their global-memory round trips.                              */
template<int EPI>
__global__ void k_gemm_tc(int od, int id, const float *__restrict__ WT,
                          const float *__restrict__ X,
                          float *__restrict__ Y, int T,
                          float *__restrict__ P,
                          const float *__restrict__ bias)
{
    __shared__ float Xs[2][BM][BK + 4];
    __shared__ float Ws[2][BK][BN];
    int bx = blockIdx.x * BN, by = blockIdx.y * BM;
    int S = gridDim.z, z = blockIdx.z;
    int kb = id / S, ks = z * kb, nt = kb / BK;
    int tid = threadIdx.y * 16 + threadIdx.x;
    int warp = tid >> 5, mw = warp & 1, nw = warp >> 1;
    int xr = tid >> 2,  xc  = (tid & 3)  << 2;
    int wr0 = tid >> 4, wc0 = (tid & 15) << 2;

    auto stage = [&](int buf, int kt) {
        int k0 = ks + kt * BK;
        __pipeline_memcpy_async(&Ws[buf][wr0][wc0],
            &WT[(long)(k0 + wr0)*od + bx + wc0], 16);
        __pipeline_memcpy_async(&Ws[buf][wr0 + 8][wc0],
            &WT[(long)(k0 + wr0 + 8)*od + bx + wc0], 16);
        __pipeline_memcpy_async(&Xs[buf][xr][xc],
            &X[(long)(by + xr)*id + k0 + xc], 16);
        __pipeline_commit();
    };

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2];
    wmma::fill_fragment(acc[0], 0.f);
    wmma::fill_fragment(acc[1], 0.f);
    stage(0, 0);
    for (int kt = 0; kt < nt; kt++) {
        int buf = kt & 1;
        __pipeline_wait_prior(0);
        __syncthreads();
        if (kt + 1 < nt) stage(buf ^ 1, kt + 1);
        #pragma unroll
        for (int k8 = 0; k8 < BK; k8 += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> a;
            wmma::load_matrix_sync(a, &Xs[buf][mw*16][k8], BK + 4);
            #pragma unroll
            for (int i = 0; i < a.num_elements; i++)
                a.x[i] = wmma::__float_to_tf32(a.x[i]);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32, wmma::row_major> b;
                wmma::load_matrix_sync(b, &Ws[buf][k8][nw*32 + j*16], BN);
                #pragma unroll
                for (int i = 0; i < b.num_elements; i++)
                    b.x[i] = wmma::__float_to_tf32(b.x[i]);
                wmma::mma_sync(acc[j], a, b, acc[j]);
            }
        }
        __syncthreads();
    }
    float *out = (S == 1) ? Y : P + (long)z * T * od;
    if (EPI == 0) {
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::store_matrix_sync(
                &out[(long)(by + mw*16)*od + bx + nw*32 + j*16],
                acc[j], od, wmma::mem_row_major);
    } else {
        float *Ct = &Ws[0][0][0];              /* 2048 floats = 32x64 */
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::store_matrix_sync(&Ct[(mw*16)*BN + nw*32 + j*16],
                                    acc[j], BN, wmma::mem_row_major);
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int e = tid + i*128, r = e >> 4, c4 = (e & 15) << 2;
            float4 v = *(float4*)&Ct[r*BN + c4];
            if (EPI & 1) {
                float4 bb = *(const float4*)&bias[bx + c4];
                v.x += bb.x; v.y += bb.y; v.z += bb.z; v.w += bb.w;
            }
            if (EPI & 2) {
                v.x = fmaxf(v.x, 0.f); v.y = fmaxf(v.y, 0.f);
                v.z = fmaxf(v.z, 0.f); v.w = fmaxf(v.w, 0.f);
            }
            *(float4*)&out[(long)(by + r)*od + bx + c4] = v;
        }
    }
}

/* guarded fallback (non-tile-multiple shapes, e.g. last-layer nt=1),
 * with runtime bias/relu epilogue */
__global__ void k_gemm_guard(int od, int id, const float *__restrict__ WT,
                             const float *__restrict__ X,
                             float *__restrict__ Y, int T,
                             float *__restrict__ P,
                             const float *__restrict__ bias, int relu)
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
            if (i >= od) continue;
            float v = acc[m][n];
            if (S == 1) {
                if (bias) v += bias[i];
                if (relu) v = fmaxf(v, 0.f);
            }
            out[(long)t*od + i] = v;
        }
    }
}

/* fixed-order split-K reduction with fused epilogue:
 * dst = (accum ? dst : 0) + sum_z P[z] + bias    (deterministic) */
__global__ void k_reduce(float *__restrict__ dst,
                         const float *__restrict__ P, int S, long n,
                         int od, const float *__restrict__ bias, int accum)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int z = 0; z < S; z++) acc += P[(long)z*n + i];
    if (bias) acc += bias[i % od];
    dst[i] = accum ? dst[i] + acc : acc;
}

#define MAX_SPLIT 8
static float *d_part;                    /* [MAX_SPLIT][SEQ][max od<1024] */

static void mm(int od, int id, const float *WT, const float *X,
               float *dst, int T, const float *bias, int relu, int accum)
{
    int S = 1;
    if (od < 1024) S = (id >= 2048) ? 8 : 4;
    dim3 g((od + BN - 1)/BN, (T + BM - 1)/BM, S), b(16, 8);
    int interior = (od % BN == 0) && (T % BM == 0) &&
                   (id % S == 0) && ((id / S) % BK == 0);
    if (S == 1) {
        if (interior) {
            int epi = (bias ? 1 : 0) | (relu ? 2 : 0);
            if      (epi == 0) k_gemm_tc<0><<<g,b>>>(od,id,WT,X,dst,T,d_part,bias);
            else if (epi == 1) k_gemm_tc<1><<<g,b>>>(od,id,WT,X,dst,T,d_part,bias);
            else               k_gemm_tc<3><<<g,b>>>(od,id,WT,X,dst,T,d_part,bias);
        } else
            k_gemm_guard<<<g,b>>>(od,id,WT,X,dst,T,d_part,bias,relu);
        /* accum unused on S==1 paths by construction */
    } else {
        if (interior)
            k_gemm_tc<0><<<g,b>>>(od,id,WT,X,dst,T,d_part,NULL);
        else
            k_gemm_guard<<<g,b>>>(od,id,WT,X,dst,T,d_part,NULL,0);
        long n = (long)T * od;
        k_reduce<<<(int)((n + 255)/256), 256>>>(dst, d_part, S, n,
                                                od, bias, accum);
    }
}

/* ------------- other kernels (unchanged from v3) ------------- */
/* affine-free normalize: gamma/beta are folded into the adjacent
 * weight matrices and biases (exact rewrite, done on the host). */
__global__ void k_norm(const float *__restrict__ x, float *__restrict__ y)
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
        yo[i] = (xi[i] - mean) * rstd;
}

/* t0: first query row to compute (last-token pruning).  The 1/sqrt(HD)
 * score scale is folded into the q-columns of wqkv offline. */
__global__ void k_attn(const float *__restrict__ qkv, float *__restrict__ ao,
                       int t0)
{
    int hh = blockIdx.x, t = blockIdx.y + t0, o = hh * HD;
    const long R = 3L * D;
    __shared__ float sc[SEQ], qs[HD], rmax[4], rsum[4];
    int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;

    for (int j = tid; j < HD; j += blockDim.x)
        qs[j] = qkv[(long)t*R + o + j];
    __syncthreads();

    for (int s = w; s <= t; s += 4) {                 /* warp per s */
        float acc = 0.f;
        for (int j = lane; j < HD; j += 32)           /* coalesced   */
            acc += qs[j] * qkv[(long)s*R + D + o + j];
        for (int off = 16; off; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) sc[s] = acc;
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


/* ------------- host ------------- */
struct Layer {
    float *wqkv, *bqkv;        /* folded: (W diag(g1)) and W*beta1,
                                  q-part also carries the 1/sqrt(HD) */
    float *wo, *w1, *b1, *w2, *b2;   /* b1 = folded b1 + W1*beta2 */
};
static Layer Ls[LAYERS];
static float *d_wout, *d_bout;
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
    long nD = (long)SEQ*D;
    CK(cudaMemcpy(d_x, d_in, nD*sizeof(float), cudaMemcpyDeviceToDevice));
    P0;
    for (int l = 0; l < LAYERS; l++) {
        Layer *L = &Ls[l];
        int t0 = (l == LAYERS-1) ? SEQ-1 : 0, nt = SEQ - t0;
        k_norm<<<SEQ,256>>>(d_x, d_xn);
        P(0);
        mm(3*D, D, L->wqkv, d_xn, d_qkv, SEQ, L->bqkv, 0, 0);
        P(1);
        dim3 g(HEADS, nt);
        k_attn<<<g,128>>>(d_qkv, d_ao, t0);
        P(2);
        mm(D, D, L->wo, d_ao + (long)t0*D, d_x + (long)t0*D, nt,
           NULL, 0, /*accum*/1);
        P(3);
        k_norm<<<nt,256>>>(d_x + (long)t0*D, d_xn);
        P(0);
        mm(FF, D, L->w1, d_xn, d_h, nt, L->b1, /*relu*/1, 0);
        P(4);
        mm(D, FF, L->w2, d_h, d_x + (long)t0*D, nt, L->b2, 0, /*accum*/1);
        P(6);
    }
    k_norm<<<1,256>>>(d_x + (long)(SEQ-1)*D, d_xn);
    mm(NOUT, D, d_wout, d_xn, d_lg, 1, d_bout, 0, 0);
    P(7);
}

int main(void)
{
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    float ws = 0.6f / sqrtf((float)D);
    float ascale = 1.0f / sqrtf((float)HD);
    for (int l = 0; l < LAYERS; l++) {
        Layer *L = &Ls[l];
        /* -- fill host buffers in the ORIGINAL PRNG order (the gate) -- */
        float *hq  = host_fill((long)D*D,  ws);
        float *hk  = host_fill((long)D*D,  ws);
        float *hv  = host_fill((long)D*D,  ws);
        float *hwo = host_fill((long)D*D,  ws);
        float *hw1 = host_fill((long)FF*D, ws);
        float *hb1 = host_fill(FF, 0.01f);
        float *hw2 = host_fill((long)D*FF, ws*0.5f);
        float *hb2 = host_fill(D, 0.01f);
        float *g1  = (float*)malloc(D*4), *bt1 = (float*)malloc(D*4);
        float *g2  = (float*)malloc(D*4), *bt2 = (float*)malloc(D*4);
        for (int i = 0; i < D; i++) {
            g1[i] = 1.0f + fr(0.1f); bt1[i] = fr(0.05f);
            g2[i] = 1.0f + fr(0.1f); bt2[i] = fr(0.05f);
        }
        /* -- fold layernorm-1 into qkv (EXACT: W(g*x+b) = (Wg)x + Wb),
              plus the attention scale into the q third -- */
        float *wq3 = (float*)malloc(3L*D*D*4);
        float *bq3 = (float*)malloc(3L*D*4);
        for (int i = 0; i < D; i++) {
            double sq = 0, sk = 0, sv = 0;
            for (int j = 0; j < D; j++) {
                wq3[(long)j*3*D +       i] = hq[(long)i*D+j]*g1[j]*ascale;
                wq3[(long)j*3*D + D   + i] = hk[(long)i*D+j]*g1[j];
                wq3[(long)j*3*D + 2*D + i] = hv[(long)i*D+j]*g1[j];
                sq += (double)hq[(long)i*D+j]*bt1[j];
                sk += (double)hk[(long)i*D+j]*bt1[j];
                sv += (double)hv[(long)i*D+j]*bt1[j];
            }
            bq3[i]       = (float)(sq * ascale);
            bq3[D + i]   = (float)sk;
            bq3[2*D + i] = (float)sv;
        }
        L->wqkv = to_dev(wq3, 3L*D*D);
        L->bqkv = to_dev(bq3, 3L*D);
        L->wo   = to_dev_T(hwo, D, D);
        /* -- fold layernorm-2 into w1 -- */
        float *w1t = (float*)malloc((long)FF*D*4);
        float *b1f = (float*)malloc((long)FF*4);
        for (int f = 0; f < FF; f++) {
            double s = hb1[f];
            for (int j = 0; j < D; j++) {
                w1t[(long)j*FF + f] = hw1[(long)f*D+j]*g2[j];
                s += (double)hw1[(long)f*D+j]*bt2[j];
            }
            b1f[f] = (float)s;
        }
        L->w1 = to_dev(w1t, (long)FF*D);
        L->b1 = to_dev(b1f, FF);
        L->w2 = to_dev_T(hw2, D, FF);
        L->b2 = to_dev(hb2, D);
        free(hq); free(hk); free(hv); free(hw1); free(hb1);
        free(g1); free(bt1); free(g2); free(bt2);
    }
    {   /* head: fill wout then gf/bf in original order, fold, upload */
        float *hwout = host_fill((long)NOUT*D, ws);
        float *g = (float*)malloc(D*4), *b = (float*)malloc(D*4);
        for (int i = 0; i < D; i++) { g[i] = 1.0f+fr(0.1f); b[i] = fr(0.05f); }
        float *wt = (float*)malloc((long)NOUT*D*4);
        float *bo = (float*)malloc(NOUT*4);
        for (int o = 0; o < NOUT; o++) {
            double s = 0;
            for (int j = 0; j < D; j++) {
                wt[(long)j*NOUT + o] = hwout[(long)o*D+j]*g[j];
                s += (double)hwout[(long)o*D+j]*b[j];
            }
            bo[o] = (float)s;
        }
        d_wout = to_dev(wt, (long)NOUT*D);
        d_bout = to_dev(bo, NOUT);
        free(hwout); free(g); free(b);
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

    forward(); CK(cudaDeviceSynchronize());
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
