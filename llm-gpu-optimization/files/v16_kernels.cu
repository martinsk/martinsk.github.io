/* v16_kernels.cu — the SEARCH SUBSTRATE.
 * One binary containing every Port-1 candidate as a template
 * instantiation (compile once, sweep at runtime), the proven attention/
 * norm/gemv phases, cuBLASLt as vendor tactic, and an in-context PAIRED
 * A/B measurement mode: two tactic tables alternate within one process,
 * honoring the standalone-kernels-lie and same-window lessons.
 * Usage:  ./v16 a0 s0 a1 s1 a2 s2 a3 s3  b0 t0 b1 t1 b2 t2 b3 t3
 *   (impl,split) per site [qkv,wo,w1,w2] for table A then table B;
 *   impl: -1 = cuBLASLt, 0..NCFG-1 = template config.  Prints per-site
 *   phase ms for both tables, paired, plus gate logits per table.
 * Build:  nvcc -O3 -arch=sm_86 -o v16 v16_kernels.cu -lcublasLt -lcublas
 */
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
using namespace nvcuda;

#define D 768
#define LAYERS 12
#define HEADS 12
#define SEQ 128
#define FF (4*D)
#define HD (D/HEADS)
#define NOUT 16
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); \
    exit(1); } } while (0)
#define LTCK(x) do { cublasStatus_t s_=(x); if (s_!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cublasLt err %d @%d\n",(int)s_,__LINE__); exit(1);} } while(0)

/* ---------------- generic templated tensor-core GEMM ----------------
 * Warp owns a 16x32 slab (two 16x16 fp32 accumulators); NW warps tile
 * the TBM x TBN block; STG-deep cp.async pipeline; always writes
 * split-K partials (epilogues live in the reduce: uniform + simple). */
template<int TBM, int TBN, int TBK, int STG>
__global__ void k_gemm(int od, int id, const float *__restrict__ WT,
                       const float *__restrict__ X,
                       float *__restrict__ P, int T)
{
    constexpr int NW  = (TBM/16)*(TBN/32);
    constexpr int THR = 32*NW;
    __shared__ float Xs[STG][TBM][TBK + 4];
    __shared__ float Ws[STG][TBK][TBN];
    int bx = blockIdx.x * TBN, by = blockIdx.y * TBM;
    int S = gridDim.z, z = blockIdx.z;
    int kb = id / S, ks = z * kb, nt = kb / TBK;
    int tid = threadIdx.x, warp = tid >> 5;
    int mw = warp % (TBM/16), nw = warp / (TBM/16);
    auto stage = [&](int buf, int kt) {
        int k0 = ks + kt * TBK;
        for (int e = tid; e < TBK*TBN/4; e += THR) {
            int r = e / (TBN/4), c = (e % (TBN/4)) * 4;
            __pipeline_memcpy_async(&Ws[buf][r][c],
                &WT[(long)(k0 + r)*od + bx + c], 16);
        }
        for (int e = tid; e < TBM*TBK/4; e += THR) {
            int r = e / (TBK/4), c = (e % (TBK/4)) * 4;
            __pipeline_memcpy_async(&Xs[buf][r][c],
                &X[(long)(by + r)*id + k0 + c], 16);
        }
        __pipeline_commit();
    };
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2];
    wmma::fill_fragment(acc[0], 0.f);
    wmma::fill_fragment(acc[1], 0.f);
    int pre = STG - 1 < nt ? STG - 1 : nt;
    for (int p = 0; p < pre; p++) stage(p, p);
    for (int kt = 0; kt < nt; kt++) {
        int buf = kt % STG;
        int wp = nt - kt - 1; if (wp > STG - 2) wp = STG - 2; if (wp < 0) wp = 0;
        __pipeline_wait_prior(wp);
        __syncthreads();
        if (kt + STG - 1 < nt) stage((kt + STG - 1) % STG, kt + STG - 1);
        #pragma unroll
        for (int k8 = 0; k8 < TBK; k8 += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> a;
            wmma::load_matrix_sync(a, &Xs[buf][mw*16][k8], TBK + 4);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32, wmma::row_major> b;
                wmma::load_matrix_sync(b, &Ws[buf][k8][nw*32 + j*16], TBN);
                wmma::mma_sync(acc[j], a, b, acc[j]);
            }
        }
        __syncthreads();
    }
    float *out = P + (long)z * T * od;
    #pragma unroll
    for (int j = 0; j < 2; j++)
        wmma::store_matrix_sync(
            &out[(long)(by + mw*16)*od + bx + nw*32 + j*16],
            acc[j], od, wmma::mem_row_major);
}

/* epilogue reduce: dst = (accum?dst:0) + sum partials + bias, opt relu */
__global__ void k_reduce(float *__restrict__ dst,
                         const float *__restrict__ P, int S, long n,
                         int od, const float *__restrict__ bias,
                         int accum, int relu)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int z = 0; z < S; z++) acc += P[(long)z*n + i];
    if (bias) acc += bias[i % od];
    if (relu) acc = fmaxf(acc, 0.f);
    dst[i] = accum ? dst[i] + acc : acc;
}

typedef void (*gemmfn)(int,int,const float*,const float*,float*,int);
struct Cfg { int bm,bn,bk,stg,thr; gemmfn fn; };
static const Cfg CFGS[] = {
  {32,64,16,2,128,(gemmfn)k_gemm<32,64,16,2>},
  {32,64,16,3,128,(gemmfn)k_gemm<32,64,16,3>},
  {32,64,32,2,128,(gemmfn)k_gemm<32,64,32,2>},
  {32,64,32,3,128,(gemmfn)k_gemm<32,64,32,3>},
  {32,128,16,2,256,(gemmfn)k_gemm<32,128,16,2>},
  {32,128,16,3,256,(gemmfn)k_gemm<32,128,16,3>},
  {32,128,32,2,256,(gemmfn)k_gemm<32,128,32,2>},
  {64,64,16,2,256,(gemmfn)k_gemm<64,64,16,2>},
  {64,64,16,3,256,(gemmfn)k_gemm<64,64,16,3>},
  {64,64,32,2,256,(gemmfn)k_gemm<64,64,32,2>},
  {64,128,16,2,512,(gemmfn)k_gemm<64,128,16,2>},
  {64,128,16,3,512,(gemmfn)k_gemm<64,128,16,3>},
};
#define NCFG (int)(sizeof(CFGS)/sizeof(CFGS[0]))


static cublasLtHandle_t LTH;
static cudaStream_t ST;
static void *d_ltws;
#define LTWS (32u<<20)
typedef struct { int od,id,T,ep; cublasLtMatmulDesc_t op;
                 cublasLtMatrixLayout_t La,Lb,Lc; } LtPlan;
static LtPlan ltp[16]; static int nltp = 0;
static LtPlan *lt_plan(int od, int id, int T, int ep)
{
    for (int i = 0; i < nltp; i++)
        if (ltp[i].od==od && ltp[i].id==id && ltp[i].T==T && ltp[i].ep==ep)
            return &ltp[i];
    LtPlan *p = &ltp[nltp++];
    p->od=od; p->id=id; p->T=T; p->ep=ep;
    LTCK(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F_FAST_TF32,
                                  CUDA_R_32F));
    cublasLtEpilogue_t e = ep==2 ? CUBLASLT_EPILOGUE_RELU_BIAS
                         : ep==1 ? CUBLASLT_EPILOGUE_BIAS
                                 : CUBLASLT_EPILOGUE_DEFAULT;
    LTCK(cublasLtMatmulDescSetAttribute(p->op,
        CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
    LTCK(cublasLtMatrixLayoutCreate(&p->La, CUDA_R_32F, od, id, od));
    LTCK(cublasLtMatrixLayoutCreate(&p->Lb, CUDA_R_32F, id, T,  id));
    LTCK(cublasLtMatrixLayoutCreate(&p->Lc, CUDA_R_32F, od, T,  od));
    return p;
}

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

/* v13b: FLASH ATTENTION (hand-built).  One block = (head, 16-row query
 * tile).  K and V stream through shared memory in 32-row tiles, each
 * serving all 16 query rows (~16x less K/V traffic).  Softmax is ONLINE:
 * running max m and sum l per row; when a tile raises the max, previous
 * contributions rescale by exp(m_old - m_new).  Scores never touch
 * global memory.  Fixed tile/shuffle order -> deterministic.  Causal
 * mask in the score; pruned last layer = one tile, 15 rows masked.
 * K/V rows padded +4: bank-conflict-free AND float4-alignable. */
__global__ void k_attn(const float *__restrict__ qkv,
                       float *__restrict__ ao, int t0)
{
    const long R = 3L * D;
    int hh = blockIdx.x, o = hh * HD;
    int qb = t0 + blockIdx.y * 16;
    __shared__ float Qs[16][HD];
    __shared__ float Ks[32][HD + 4], Vs[32][HD + 4];
    __shared__ float Ss[16][33];
    __shared__ float rm[16], rl[16];
    int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int e = tid + i*128, r = e >> 4, c = (e & 15) << 2;
        int tq = qb + r;
        float4 v = (tq < SEQ) ? *(const float4*)&qkv[(long)tq*R + o + c]
                              : make_float4(0.f, 0.f, 0.f, 0.f);
        *(float4*)&Qs[r][c] = v;
    }
    if (tid < 16) { rm[tid] = -3.4e38f; rl[tid] = 0.f; }
    float O[4][2] = {};
    __syncthreads();

    int tmax = qb + 15 < SEQ - 1 ? qb + 15 : SEQ - 1;
    for (int s0 = 0; s0 <= tmax; s0 += 32) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int e = tid + i*128, r = e >> 4, c = (e & 15) << 2;
            int s = s0 + r;
            float4 kv = (s < SEQ) ? *(const float4*)&qkv[(long)s*R + D + o + c]
                                  : make_float4(0.f, 0.f, 0.f, 0.f);
            *(float4*)&Ks[r][c] = kv;
            float4 vv = (s < SEQ) ? *(const float4*)&qkv[(long)s*R + 2*D + o + c]
                                  : make_float4(0.f, 0.f, 0.f, 0.f);
            *(float4*)&Vs[r][c] = vv;
        }
        __syncthreads();

        float f[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = 4*w + i, tq = qb + r, s = s0 + lane;
            float acc = 0.f;
            #pragma unroll
            for (int j = 0; j < HD; j++) acc += Qs[r][j] * Ks[lane][j];
            if (s > tq || tq >= SEQ) acc = -3.4e38f;
            float mt = acc;
            for (int off = 16; off; off >>= 1)
                mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, off));
            float mold = rm[r], mnew = fmaxf(mold, mt);
            float p = (acc > -1e37f) ? expf(acc - mnew) : 0.f;
            float ps = p;
            for (int off = 16; off; off >>= 1)
                ps += __shfl_xor_sync(0xffffffffu, ps, off);
            f[i] = expf(mold - mnew);
            if (lane == 0) { rm[r] = mnew; rl[r] = rl[r]*f[i] + ps; }
            Ss[r][lane] = p;
        }
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = 4*w + i;
            float o0 = 0.f, o1 = 0.f;
            #pragma unroll
            for (int c = 0; c < 32; c++) {
                float p = Ss[r][c];
                o0 += p * Vs[c][lane];
                o1 += p * Vs[c][lane + 32];
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
            float inv = 1.f / rl[r];
            ao[(long)tq*D + o + lane]      = O[i][0] * inv;
            ao[(long)tq*D + o + lane + 32] = O[i][1] * inv;
        }
    }
}
/* deterministic coalesced GEMV for T==1 sites (pruned last layer, head):
 * thread i owns output column i; WT[j*od+i] reads are coalesced across
 * the warp by construction; serial j-loop -> fixed summation order. */
__global__ void k_gemv(int od, int id, const float *__restrict__ WT,
                       const float *__restrict__ x, float *__restrict__ dst,
                       const float *__restrict__ bias, int relu, int accum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= od) return;
    float acc = 0.f;
    for (int j = 0; j < id; j++) acc += WT[(long)j*od + i] * x[j];
    if (bias) acc += bias[i];
    if (relu) acc = fmaxf(acc, 0.f);
    dst[i] = accum ? dst[i] + acc : acc;
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

typedef struct { float *wqkv,*bqkv,*wo,*w1,*b1,*w2,*b2; } Layer;
static Layer Ls[LAYERS];
static float *d_wout,*d_bout;
static float *d_x,*d_xn,*d_qkv,*d_ao,*d_h,*d_lg,*d_in,*d_part;

/* tactic: per site [qkv,wo,w1,w2]: impl (-1=Lt, 0..NCFG-1) + split */
static int TAC[4], SPL[4];
static double ph[4];               /* per-site ms accumulators */
static double pnow(void){ struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static void site_mm(int site, int od, int id, const float *WT,
                    const float *X, float *dst, int T,
                    const float *bias, int relu, int accum)
{
    CK(cudaDeviceSynchronize()); double t0 = pnow();
    int impl = TAC[site], S = SPL[site];
    if (impl < 0) {                    /* vendor: cuBLASLt, fused epi */
        int ep = bias ? (relu ? 2 : 1) : 0;
        LtPlan *p = lt_plan(od, id, T, ep);
        if (bias) LTCK(cublasLtMatmulDescSetAttribute(p->op,
            CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        const float a = 1.f, b0 = 0.f;
        if (accum) {                   /* Lt beta path for residual */
            const float b1v = 1.f;
            LTCK(cublasLtMatmul(LTH,p->op,&a,WT,p->La,X,p->Lb,
                                &b1v,dst,p->Lc,dst,p->Lc,NULL,d_ltws,LTWS,ST));
        } else
            LTCK(cublasLtMatmul(LTH,p->op,&a,WT,p->La,X,p->Lb,
                                &b0,dst,p->Lc,dst,p->Lc,NULL,d_ltws,LTWS,ST));
    } else {
        const Cfg *c = &CFGS[impl];
        dim3 g(od/c->bn, T/c->bm, S), b(c->thr);
        c->fn<<<g,b,0,ST>>>(od, id, WT, X, d_part, T);
        long n = (long)T * od;
        k_reduce<<<(int)((n+255)/256),256,0,ST>>>(dst, d_part, S, n,
                                                  od, bias, accum, relu);
    }
    CK(cudaDeviceSynchronize()); ph[site] += pnow() - t0;
}

static void forward(void)
{
    CK(cudaMemcpyAsync(d_x, d_in, (long)SEQ*D*4,
                       cudaMemcpyDeviceToDevice, ST));
    for (int l = 0; l < LAYERS; l++) {
        Layer *L = &Ls[l];
        int t0 = (l == LAYERS-1) ? SEQ-1 : 0, nt = SEQ - t0;
        k_norm<<<SEQ,256,0,ST>>>(d_x, d_xn);
        site_mm(0, 3*D, D, L->wqkv, d_xn, d_qkv, SEQ, L->bqkv, 0, 0);
        dim3 g(HEADS, (nt + 15)/16);
        k_attn<<<g,128,0,ST>>>(d_qkv, d_ao, t0);
        if (nt == 1) {
            k_gemv<<<(D+255)/256,256,0,ST>>>(D, D, L->wo,
                d_ao + (long)t0*D, d_x + (long)t0*D, NULL, 0, 1);
            k_norm<<<1,256,0,ST>>>(d_x + (long)t0*D, d_xn);
            k_gemv<<<(FF+255)/256,256,0,ST>>>(FF, D, L->w1, d_xn, d_h,
                                              L->b1, 1, 0);
            k_gemv<<<(D+255)/256,256,0,ST>>>(D, FF, L->w2, d_h,
                d_x + (long)t0*D, L->b2, 0, 1);
        } else {
            site_mm(1, D, D, L->wo, d_ao, d_x, SEQ, NULL, 0, 1);
            k_norm<<<SEQ,256,0,ST>>>(d_x, d_xn);
            site_mm(2, FF, D, L->w1, d_xn, d_h, SEQ, L->b1, 1, 0);
            site_mm(3, D, FF, L->w2, d_h, d_x, SEQ, L->b2, 0, 1);
        }
    }
    k_norm<<<1,256,0,ST>>>(d_x + (long)(SEQ-1)*D, d_xn);
    k_gemv<<<(NOUT+255)/256,256,0,ST>>>(NOUT, D, d_wout, d_xn, d_lg,
                                        d_bout, 0, 0);
    CK(cudaDeviceSynchronize());
}

static void run_table(const int *tac, const int *spl, int iters,
                      double *site_ms, float *lg01)
{
    memcpy(TAC, tac, 4*sizeof(int));
    memcpy(SPL, spl, 4*sizeof(int));
    memset(ph, 0, sizeof ph);
    forward();                                     /* warm + gate */
    float lg[NOUT];
    CK(cudaMemcpy(lg, d_lg, NOUT*4, cudaMemcpyDeviceToHost));
    lg01[0] = lg[0]; lg01[1] = lg[1];
    memset(ph, 0, sizeof ph);
    for (int i = 0; i < iters; i++) forward();
    for (int s = 0; s < 4; s++) site_ms[s] = ph[s]/iters*1e3;
}

int main(int argc, char **argv)
{
    if (argc != 17) { fprintf(stderr,
        "usage: %s a0 s0 a1 s1 a2 s2 a3 s3  b0 t0 b1 t1 b2 t2 b3 t3\n",
        argv[0]); return 2; }
    int A[4], AS[4], B[4], BS_[4];
    for (int i = 0; i < 4; i++) {
        A[i]  = atoi(argv[1+2*i]);  AS[i]  = atoi(argv[2+2*i]);
        B[i]  = atoi(argv[9+2*i]);  BS_[i] = atoi(argv[10+2*i]);
    }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
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
    CK(cudaMalloc(&d_lg,  NOUT*4));
    CK(cudaMalloc(&d_part,(long)8*SEQ*3072*4));
    CK(cudaStreamCreate(&ST));
    LTCK(cublasLtCreate(&LTH));
    CK(cudaMalloc(&d_ltws, LTWS));

    double a_ms[4], b_ms[4]; float alg[2], blg[2];
    /* paired: A,B,A,B interleaved epochs in one process/window */
    double aa[4]={0,0,0,0}, bb[4]={0,0,0,0};
    for (int ep = 0; ep < 4; ep++) {
        run_table(A, AS, 10, a_ms, alg);
        run_table(B, BS_, 10, b_ms, blg);
        for (int s = 0; s < 4; s++) { aa[s]+=a_ms[s]/4; bb[s]+=b_ms[s]/4; }
    }
    printf("A: %.4f %.4f %.4f %.4f  logits %.6f %.6f\n",
           aa[0],aa[1],aa[2],aa[3], alg[0],alg[1]);
    printf("B: %.4f %.4f %.4f %.4f  logits %.6f %.6f\n",
           bb[0],bb[1],bb[2],bb[3], blg[0],blg[1]);
    return 0;
}
