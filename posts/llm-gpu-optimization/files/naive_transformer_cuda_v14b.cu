/* naive_transformer_cuda_v14b.cu — v14b: THE PERSISTENT MEGAKERNEL.
 * The entire 12-layer forward pass is ONE cooperative kernel: persistent
 * blocks, every phase a device function with a block-stride work loop,
 * kernel boundaries replaced by grid.sync() (~111/pass), one union'd
 * shared-memory arena re-dressed per phase.  The price, stated up
 * front: cuBLASLt cannot live inside a kernel, so the wide GEMMs come
 * back in-house (our WMMA vs the library's tuned kernels, a known
 * ~0.4 ms regression the boundary savings must out-earn).  Zero
 * library calls; ONE launch per pass; no graphs needed.
 * Build: nvcc -O3 -arch=sm_86 -o bench_cu14b \
 *        naive_transformer_cuda_v14b.cu
 * Gate: TF32 class, ~1e-3 vs 0.095916 0.033000. */
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
using namespace nvcuda;
namespace cg = cooperative_groups;

#define D 768
#define LAYERS 12
#define HEADS 12
#define SEQ 128
#define FF (4*D)
#define HD (D/HEADS)
#define NOUT 16
#define BM 32
#define BN 64
#define BK 16
#ifndef ITERS
#define ITERS 40
#endif
#ifndef REPS
#define REPS 3
#endif
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); \
    exit(1); } } while (0)

typedef struct {
    float *wqkv, *bqkv;
    float *wo, *w1, *b1, *w2, *b2;
} Layer;

struct Params {
    Layer L[LAYERS];
    float *wout, *bout;
    float *x, *xn, *qkv, *ao, *h, *part, *lg, *in;
};
__constant__ Params CP;

union Arena {
    struct { float Xs[2][BM][BK+4]; float Ws[2][BK][BN]; } g;
    struct { float Qs[16][HD]; float Ks[32][HD+4]; float Vs[32][HD+4];
             float Ss[16][33]; float rm[16], rl[16]; } a;
    struct { float ss[128], sq[128]; } n;
};

__device__ void dev_gemm_tc(Arena &U, int od, int id,
                            const float *__restrict__ WT,
                            const float *__restrict__ X,
                            float *__restrict__ Y, int T, int S,
                            const float *__restrict__ bias, int relu)
{
    int gx = od / BN, gy = T / BM;
    int tid = threadIdx.x, warp = tid >> 5, mw = warp & 1, nw = warp >> 1;
    int xr = tid >> 2,  xc  = (tid & 3)  << 2;
    int wr0 = tid >> 4, wc0 = (tid & 15) << 2;
    int kb = id / S;
    for (int tile = blockIdx.x; tile < gx*gy*S; tile += gridDim.x) {
        int bx = (tile % gx) * BN;
        int by = ((tile / gx) % gy) * BM;
        int z  = tile / (gx*gy);
        int ks = z * kb, nt = kb / BK;
        __syncthreads();
        auto stage = [&](int buf, int kt) {
            int k0 = ks + kt * BK;
            __pipeline_memcpy_async(&U.g.Ws[buf][wr0][wc0],
                &WT[(long)(k0 + wr0)*od + bx + wc0], 16);
            __pipeline_memcpy_async(&U.g.Ws[buf][wr0 + 8][wc0],
                &WT[(long)(k0 + wr0 + 8)*od + bx + wc0], 16);
            __pipeline_memcpy_async(&U.g.Xs[buf][xr][xc],
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
                wmma::load_matrix_sync(a, &U.g.Xs[buf][mw*16][k8], BK + 4);
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 8,
                                   wmma::precision::tf32, wmma::row_major> b;
                    wmma::load_matrix_sync(b, &U.g.Ws[buf][k8][nw*32+j*16], BN);
                    wmma::mma_sync(acc[j], a, b, acc[j]);
                }
            }
            __syncthreads();
        }
        float *out = (S == 1) ? Y : CP.part + (long)z * T * od;
        if (S > 1 || (!bias && !relu)) {
            #pragma unroll
            for (int j = 0; j < 2; j++)
                wmma::store_matrix_sync(
                    &out[(long)(by + mw*16)*od + bx + nw*32 + j*16],
                    acc[j], od, wmma::mem_row_major);
        } else {
            float *Ct = &U.g.Ws[0][0][0];
            __syncthreads();
            #pragma unroll
            for (int j = 0; j < 2; j++)
                wmma::store_matrix_sync(&Ct[(mw*16)*BN + nw*32 + j*16],
                                        acc[j], BN, wmma::mem_row_major);
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int e = tid + i*128, r = e >> 4, c4 = (e & 15) << 2;
                float4 v = *(float4*)&Ct[r*BN + c4];
                if (bias) {
                    float4 bb = *(const float4*)&bias[bx + c4];
                    v.x += bb.x; v.y += bb.y; v.z += bb.z; v.w += bb.w;
                }
                if (relu) {
                    v.x = fmaxf(v.x,0.f); v.y = fmaxf(v.y,0.f);
                    v.z = fmaxf(v.z,0.f); v.w = fmaxf(v.w,0.f);
                }
                *(float4*)&Y[(long)(by + r)*od + bx + c4] = v;
            }
        }
    }
}

__device__ void dev_reduce(long n, int S, int od, float *__restrict__ dst,
                           const float *__restrict__ bias, int accum)
{
    for (long i = (long)blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x*blockDim.x) {
        float acc = 0.f;
        for (int z = 0; z < S; z++) acc += CP.part[(long)z*n + i];
        if (bias) acc += bias[i % od];
        dst[i] = accum ? dst[i] + acc : acc;
    }
}

__device__ void dev_gemv(int od, int id, const float *__restrict__ WT,
                         const float *__restrict__ x, float *__restrict__ dst,
                         const float *__restrict__ bias, int relu, int accum)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < od;
         i += gridDim.x*blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < id; j++) acc += WT[(long)j*od + i] * x[j];
        if (bias) acc += bias[i];
        if (relu) acc = fmaxf(acc, 0.f);
        dst[i] = accum ? dst[i] + acc : acc;
    }
}

__device__ void dev_norm(Arena &U, const float *__restrict__ x,
                         float *__restrict__ y, int n)
{
    for (int t = blockIdx.x; t < n; t += gridDim.x) {
        __syncthreads();
        const float *xi = x + (long)t * D;
        float *yo = y + (long)t * D;
        float s = 0.f, q = 0.f;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            float v = xi[i]; s += v; q += v * v;
        }
        U.n.ss[threadIdx.x] = s; U.n.sq[threadIdx.x] = q;
        __syncthreads();
        for (int w = blockDim.x >> 1; w > 0; w >>= 1) {
            if (threadIdx.x < w) {
                U.n.ss[threadIdx.x] += U.n.ss[threadIdx.x + w];
                U.n.sq[threadIdx.x] += U.n.sq[threadIdx.x + w];
            }
            __syncthreads();
        }
        float mean = U.n.ss[0] / D;
        float var  = U.n.sq[0] / D - mean * mean;
        float rstd = rsqrtf(var + 1e-5f);
        for (int i = threadIdx.x; i < D; i += blockDim.x)
            yo[i] = (xi[i] - mean) * rstd;
    }
}

__device__ void dev_attn(Arena &U, int t0, int nt)
{
    const float *qkv = CP.qkv; float *ao = CP.ao;
    const long R = 3L * D;
    int nq = (nt + 15) / 16;
    int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    for (int item = blockIdx.x; item < HEADS*nq; item += gridDim.x) {
        int hh = item % HEADS, o = hh * HD;
        int qb = t0 + (item / HEADS) * 16;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int e = tid + i*128, r = e >> 4, c = (e & 15) << 2;
            int tq = qb + r;
            float4 v = (tq < SEQ) ? *(const float4*)&qkv[(long)tq*R + o + c]
                                  : make_float4(0.f,0.f,0.f,0.f);
            *(float4*)&U.a.Qs[r][c] = v;
        }
        if (tid < 16) { U.a.rm[tid] = -3.4e38f; U.a.rl[tid] = 0.f; }
        float O[4][2] = {};
        __syncthreads();
        int tmax = qb + 15 < SEQ - 1 ? qb + 15 : SEQ - 1;
        for (int s0 = 0; s0 <= tmax; s0 += 32) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int e = tid + i*128, r = e >> 4, c = (e & 15) << 2;
                int s = s0 + r;
                float4 kv = (s < SEQ) ? *(const float4*)&qkv[(long)s*R + D + o + c]
                                      : make_float4(0.f,0.f,0.f,0.f);
                *(float4*)&U.a.Ks[r][c] = kv;
                float4 vv = (s < SEQ) ? *(const float4*)&qkv[(long)s*R + 2*D + o + c]
                                      : make_float4(0.f,0.f,0.f,0.f);
                *(float4*)&U.a.Vs[r][c] = vv;
            }
            __syncthreads();
            float f[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int r = 4*w + i, tq = qb + r, s = s0 + lane;
                float acc = 0.f;
                #pragma unroll
                for (int j = 0; j < HD; j++)
                    acc += U.a.Qs[r][j] * U.a.Ks[lane][j];
                if (s > tq || tq >= SEQ) acc = -3.4e38f;
                float mt = acc;
                for (int off = 16; off; off >>= 1)
                    mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, off));
                float mold = U.a.rm[r], mnew = fmaxf(mold, mt);
                float p = (acc > -1e37f) ? expf(acc - mnew) : 0.f;
                float ps = p;
                for (int off = 16; off; off >>= 1)
                    ps += __shfl_xor_sync(0xffffffffu, ps, off);
                f[i] = expf(mold - mnew);
                if (lane == 0) { U.a.rm[r] = mnew;
                                 U.a.rl[r] = U.a.rl[r]*f[i] + ps; }
                U.a.Ss[r][lane] = p;
            }
            __syncwarp();
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int r = 4*w + i;
                float o0 = 0.f, o1 = 0.f;
                #pragma unroll
                for (int c = 0; c < 32; c++) {
                    float p = U.a.Ss[r][c];
                    o0 += p * U.a.Vs[c][lane];
                    o1 += p * U.a.Vs[c][lane + 32];
                }
                O[i][0] = O[i][0]*f[i] + o0;
                O[i][1] = O[i][1]*f[i] + o1;
            }
            __syncthreads();
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = 4*w + i, tq = qb + r;
            if (tq < SEQ) {
                float inv = 1.f / U.a.rl[r];
                ao[(long)tq*D + o + lane]      = O[i][0] * inv;
                ao[(long)tq*D + o + lane + 32] = O[i][1] * inv;
            }
        }
    }
}

__global__ void mega(void)
{
    __shared__ Arena U;
    cg::grid_group g = cg::this_grid();
    float *x = CP.x, *xn = CP.xn;

    for (long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
         i < (long)SEQ*D; i += (long)gridDim.x*blockDim.x)
        x[i] = CP.in[i];
    g.sync();

    for (int l = 0; l < LAYERS; l++) {
        const Layer *L = &CP.L[l];
        int t0 = (l == LAYERS-1) ? SEQ-1 : 0, nt = SEQ - t0;
        dev_norm(U, x, xn, SEQ);                        g.sync();
        dev_gemm_tc(U, 3*D, D, L->wqkv, xn, CP.qkv, SEQ, 1, L->bqkv, 0);
        g.sync();
        dev_attn(U, t0, nt);                            g.sync();
        if (nt == 1) {
            dev_gemv(D, D, L->wo, CP.ao + (long)t0*D, x + (long)t0*D,
                     NULL, 0, 1);                       g.sync();
        } else {
            dev_gemm_tc(U, D, D, L->wo, CP.ao, x, SEQ, 4, NULL, 0);
            g.sync();
            dev_reduce((long)SEQ*D, 4, D, x, NULL, 1);  g.sync();
        }
        dev_norm(U, x + (long)t0*D, xn, nt);            g.sync();
        if (nt == 1) {
            dev_gemv(FF, D, L->w1, xn, CP.h, L->b1, 1, 0);   g.sync();
            dev_gemv(D, FF, L->w2, CP.h, x + (long)t0*D, L->b2, 0, 1);
            g.sync();
        } else {
            dev_gemm_tc(U, FF, D, L->w1, xn, CP.h, SEQ, 1, L->b1, 1);
            g.sync();
            dev_gemm_tc(U, D, FF, L->w2, CP.h, x, SEQ, 8, NULL, 0);
            g.sync();
            dev_reduce((long)SEQ*D, 8, D, x, L->b2, 1); g.sync();
        }
    }
    dev_norm(U, x + (long)(SEQ-1)*D, xn, 1);            g.sync();
    dev_gemv(NOUT, D, CP.wout, xn, CP.lg, CP.bout, 0, 0);
}

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


static Layer Ls[LAYERS];
static float *d_wout, *d_bout;
static float *d_x, *d_xn, *d_qkv, *d_ao, *d_h, *d_part, *d_lg, *d_in;
#define MAX_SPLIT 8

int main(void)
{
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    if (!prop.cooperativeLaunch) {
        fprintf(stderr, "no cooperative launch support\n"); return 1;
    }
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
    CK(cudaMalloc(&d_part,(long)MAX_SPLIT*SEQ*D*4));
    CK(cudaMalloc(&d_lg,  NOUT*4));

    Params P;
    memcpy(P.L, Ls, sizeof Ls);
    P.wout = d_wout; P.bout = d_bout;
    P.x = d_x; P.xn = d_xn; P.qkv = d_qkv; P.ao = d_ao;
    P.h = d_h; P.part = d_part; P.lg = d_lg; P.in = d_in;
    CK(cudaMemcpyToSymbol(CP, &P, sizeof P));

    int nb = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mega, 128, 0));
    dim3 grid(nb * prop.multiProcessorCount), block(128);
    printf("device: %s (%d SMs) | mega: %d blocks/SM -> grid %d, "
           "1 launch/pass, ~111 grid.syncs/pass\n",
           prop.name, prop.multiProcessorCount, nb, grid.x);

    void *args[] = { NULL };
    CK(cudaLaunchCooperativeKernel((void*)mega, grid, block, args, 0, 0));
    CK(cudaLaunchCooperativeKernel((void*)mega, grid, block, args, 0, 0));
    CK(cudaDeviceSynchronize());
    float lg[NOUT];
    CK(cudaMemcpy(lg, d_lg, NOUT*4, cudaMemcpyDeviceToHost));
    double params = (double)LAYERS*(4.0*D*D + 2.0*D*FF) + NOUT*(double)D;
    double macs = (double)LAYERS*SEQ*(4.0*D*D + 2.0*D*FF)
                + (double)LAYERS*HEADS*(double)SEQ*(SEQ+1)*HD;
    printf("model: D=%d L=%d H=%d SEQ=%d FF=%d | params %.1fM (%.0f MB)\n",
           D, LAYERS, HEADS, SEQ, FF, params/1e6, params*4/1e6);
    printf("logits[0..1] = %.6f %.6f  (ref: 0.095916 0.033000)\n",
           lg[0], lg[1]);
    for (int r = 0; r < REPS; r++) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        for (int i = 0; i < ITERS; i++)
            CK(cudaLaunchCooperativeKernel((void*)mega, grid, block,
                                           args, 0, 0));
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
