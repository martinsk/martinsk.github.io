/* v6.c — all tricks combined, vs v3 reference from identical weights.
 *
 * v6 = v3 algebra
 *    + R9 (new): Wov pushed through the attention sum:
 *         sum_s a_s (Wg xh_s) = Wg (sum_s a_s xh_s)
 *      -> value projection per TOKEN deleted; one Wg GEMV per computed
 *         ROW.  Layer 2 (pruned, 1 row) drops from 20D^2 to 2D^2 MACs.
 *      -> u projection computed only for rows that need it (v3 wasted
 *         u for layer-2 rows 0..8 inside the fused uv).
 *    + explicit-partial reductions (4-way) in norm and score dots:
 *      parallelism in the source, not in -ffast-math's permission.
 *    + 64-strip axpy matvec, always_inline, compiled with
 *      -mprefer-vector-width=512.
 *
 * Both paths share Wu, Wg, biases -> logits must agree to fp tolerance.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SEQ 10
#define D   128
#define FF  512
#define NOUT 10
#define NOUTP 16
#define LAYERS 2
#ifndef ITERS
#define ITERS 2000
#endif

typedef struct {
    float wu[D][D],  bu[D];      /* A2^T axpy layout, query bias r   */
    float wg[D][D],  bg[D];      /* Wov*diag(g) axpy layout, bias bv */
    float w1[D][FF], b1[FF];
    float w2[FF][D], b2[D];
} L6;
static L6 Ls[LAYERS];
static float wout[D][NOUTP], bout[NOUTP];

static float x_[SEQ][D], xh[SEQ][D];
static float uv3[SEQ][2*D];          /* v3 path scratch */
static float hb[FF];

__attribute__((always_inline))
static inline void mvb(int id, int od, const float *restrict W,
                       const float *restrict bias,
                       const float *restrict xin, float *restrict y)
{
    for (int o = 0; o < od; o += 64) {
        int w = od - o; if (w > 64) w = 64;
        float a2[64];
        for (int i = 0; i < w; i++) a2[i] = bias[o + i];
        for (int j = 0; j < id; j++) {
            float a = xin[j];
            const float *restrict wr = W + (long)j * od + o;
            for (int i = 0; i < w; i++) a2[i] += a * wr[i];
        }
        for (int i = 0; i < w; i++) y[o + i] = a2[i];
    }
}

/* norm with explicit 4-way partial sums: needs no reassociation license */
static inline void norm(const float *restrict xi, float *restrict out)
{
    float s0=0,s1=0,s2=0,s3=0, q0=0,q1=0,q2=0,q3=0;
    for (int i = 0; i < D; i += 4) {
        s0+=xi[i];   q0+=xi[i]*xi[i];
        s1+=xi[i+1]; q1+=xi[i+1]*xi[i+1];
        s2+=xi[i+2]; q2+=xi[i+2]*xi[i+2];
        s3+=xi[i+3]; q3+=xi[i+3]*xi[i+3];
    }
    float mean = (s0+s1+s2+s3) * (1.0f/D);
    float var  = (q0+q1+q2+q3) * (1.0f/D) - mean*mean;
    float rstd = 1.0f/sqrtf(var + 1e-5f);
    for (int i = 0; i < D; i++) out[i] = (xi[i]-mean)*rstd;
}

static inline float dot4(const float *restrict a, const float *restrict b)
{
    float d0=0,d1=0,d2=0,d3=0;
    for (int i = 0; i < D; i += 4) {
        d0+=a[i]*b[i]; d1+=a[i+1]*b[i+1];
        d2+=a[i+2]*b[i+2]; d3+=a[i+3]*b[i+3];
    }
    return (d0+d1)+(d2+d3);
}

static inline float fexp(float xx)
{
    float t=xx*1.4426950408889634f;
    float fl=floorf(t), f=t-fl;
    float p=1.0f+f*(0.69314718f+f*(0.24022651f+f*(0.05550411f+
            f*(0.00898934f+f*0.00187757f))));
    union{float f;int i;}u; u.i=((int)fl+127)<<23;
    return p*u.f;
}

/* ---------------- v3 reference path (uv fused wide projection) ------- */
static float wuv[LAYERS][D][2*D], buv[LAYERS][2*D];
static void v3_forward(const float *restrict in, float *restrict lg)
{
    memcpy(x_, in, sizeof x_);
    for (int l = 0; l < LAYERS; l++) {
        const L6 *L = &Ls[l];
        int t0 = (l == LAYERS-1) ? SEQ-1 : 0;
        for (int t = 0; t < SEQ; t++) {
            norm(x_[t], xh[t]);
            mvb(D, 2*D, &wuv[l][0][0], buv[l], xh[t], uv3[t]);
        }
        for (int t = t0; t < SEQ; t++) {
            const float *restrict u = uv3[t];
            float w[SEQ], mx = -3.4e38f;
            for (int s = 0; s <= t; s++) {
                float a = dot4(u, xh[s]);
                w[s] = a; mx = a > mx ? a : mx;
            }
            float sum = 0;
            for (int s = 0; s <= t; s++) { w[s]=fexp(w[s]-mx); sum+=w[s]; }
            float tmp[D];
            for (int j = 0; j < D; j++) tmp[j] = 0;
            for (int s = 0; s <= t; s++) {
                float a = w[s];
                const float *restrict vv = uv3[s] + D;
                for (int j = 0; j < D; j++) tmp[j] += a * vv[j];
            }
            float inv = 1.0f/sum;
            for (int j = 0; j < D; j++) x_[t][j] += tmp[j]*inv;
        }
        for (int t = t0; t < SEQ; t++) {
            float xb[D], tb[D];
            norm(x_[t], xb);
            mvb(D, FF, &L->w1[0][0], L->b1, xb, hb);
            for (int i = 0; i < FF; i++) hb[i] = hb[i] > 0 ? hb[i] : 0;
            mvb(FF, D, &L->w2[0][0], L->b2, hb, tb);
            for (int j = 0; j < D; j++) x_[t][j] += tb[j];
        }
    }
    float xf[D], o[NOUTP];
    norm(x_[SEQ-1], xf);
    mvb(D, NOUTP, &wout[0][0], bout, xf, o);
    memcpy(lg, o, NOUT*4);
}

/* ---------------- v6 path: Wg after the sum ---------------- */
static void v6_forward(const float *restrict in, float *restrict lg)
{
    memcpy(x_, in, sizeof x_);
    for (int l = 0; l < LAYERS; l++) {
        const L6 *restrict L = &Ls[l];
        int t0 = (l == LAYERS-1) ? SEQ-1 : 0;
        for (int t = 0; t < SEQ; t++)
            norm(x_[t], xh[t]);              /* keys, values, all in one */
        for (int t = t0; t < SEQ; t++) {
            float u[D];
            mvb(D, D, &L->wu[0][0], L->bu, xh[t], u);   /* only needed rows */
            float w[SEQ], mx = -3.4e38f;
            for (int s = 0; s <= t; s++) {
                float a = dot4(u, xh[s]);
                w[s] = a; mx = a > mx ? a : mx;
            }
            float sum = 0;
            for (int s = 0; s <= t; s++) { w[s]=fexp(w[s]-mx); sum+=w[s]; }
            float tmp[D];
            for (int j = 0; j < D; j++) tmp[j] = 0;
            for (int s = 0; s <= t; s++) {   /* sum RAW xh — no v cache  */
                float a = w[s];
                const float *restrict ks = xh[s];
                for (int j = 0; j < D; j++) tmp[j] += a * ks[j];
            }
            float inv = 1.0f/sum;
            for (int j = 0; j < D; j++) tmp[j] *= inv;
            float ao[D];
            mvb(D, D, &L->wg[0][0], L->bg, tmp, ao);    /* ONE Wg per row */
            for (int j = 0; j < D; j++) x_[t][j] += ao[j];
        }
        for (int t = t0; t < SEQ; t++) {
            float xb[D], tb[D];
            norm(x_[t], xb);
            mvb(D, FF, &L->w1[0][0], L->b1, xb, hb);
            for (int i = 0; i < FF; i++) hb[i] = hb[i] > 0 ? hb[i] : 0;
            mvb(FF, D, &L->w2[0][0], L->b2, hb, tb);
            for (int j = 0; j < D; j++) x_[t][j] += tb[j];
        }
    }
    float xf[D], o[NOUTP];
    norm(x_[SEQ-1], xf);
    mvb(D, NOUTP, &wout[0][0], bout, xf, o);
    memcpy(lg, o, NOUT*4);
}

/* ---------------- harness ---------------- */
static unsigned long long rng = 0x243F6A8885A308D3ull;
static float fr(float s){ rng=rng*6364136223846793005ull+1442695040888963407ull;
    return s*(((rng>>33)&0xFFFFFF)/8388608.0f-1.0f); }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+1e-9*t.tv_nsec; }

int main(void)
{
    for (int l = 0; l < LAYERS; l++) {
        L6 *L = &Ls[l];
        fill_block: ;
        float *p = (float*)L;
        for (long i = 0; i < (long)sizeof(L6)/4; i++) p[i] = fr(0.03f);
        /* build v3's fused wide projection from the SAME weights */
        for (int j = 0; j < D; j++)
            for (int o = 0; o < D; o++) {
                wuv[l][j][o]     = L->wu[j][o];
                wuv[l][j][D + o] = L->wg[j][o];   /* v-proj in axpy form */
            }
        for (int o = 0; o < D; o++) { buv[l][o]=L->bu[o]; buv[l][D+o]=L->bg[o]; }
    }
    for (long i = 0; i < D*NOUTP; i++) (&wout[0][0])[i] = fr(0.05f);
    for (int i = 0; i < NOUTP; i++) bout[i] = fr(0.02f);
    static float in[SEQ][D];
    for (long i = 0; i < SEQ*D; i++) (&in[0][0])[i] = fr(0.5f);

    float l3[NOUT], l6[NOUT];
    v3_forward(&in[0][0], l3);
    v6_forward(&in[0][0], l6);
    double dd = 0, mag = 1e-9;
    for (int i = 0; i < NOUT; i++) {
        dd = fmax(dd, fabs(l3[i]-l6[i]));
        mag = fmax(mag, fabs(l3[i]));
    }

    volatile float s = 0; double t3 = 1e9, t6 = 1e9;
    for (int r = 0; r < 3; r++) {
        double t0 = now();
        for (int i = 0; i < ITERS; i++){ in[0][0]+=s*1e-30f;
            v3_forward(&in[0][0], l3); s = l3[0]; }
        t3 = fmin(t3, (now()-t0)/ITERS);
        t0 = now();
        for (int i = 0; i < ITERS; i++){ in[0][0]+=s*1e-30f;
            v6_forward(&in[0][0], l6); s = l6[0]; }
        t6 = fmin(t6, (now()-t0)/ITERS);
    }
    /* MACs: v3: L1 10*2D^2 + 10*2*D*FF ; L2 10*2D^2 + 2*D*FF (+small)
       v6: L1 10*D^2 (u) + 10*D^2 (Wg) + 10*2*D*FF ; L2 2*D^2 + 2*D*FF */
    double m3 = (10*2.0*D*D + 10*2.0*D*FF) + (10*2.0*D*D + 2.0*D*FF);
    double m6 = (10*2.0*D*D + 10*2.0*D*FF) + (2.0*D*D + 2.0*D*FF);
    printf("relerr %.2e\n", dd/mag);
    printf("v3  %7.1f us  %5.1f GF/s  (%.0f MACs)\n", t3*1e6, 2*m3/t3/1e9, m3);
    printf("v6  %7.1f us  %5.1f GF/s  (%.0f MACs)   v6/v3 %.2fx\n",
           t6*1e6, 2*m6/t6/1e9, m6, t3/t6);
    return 0;
}
