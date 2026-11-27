/* fused.c — v3 (per-token matvec) vs v4 (token-fused GEMM projections).
 *
 * v4 idea: the uv-projection and the MLP are per-token independent, so
 * instead of   for t: stream_all_weights(matvec(t))
 * do           for weight_tile: for t: consume tile
 * Each weight byte is loaded from DRAM once per pass instead of SEQ
 * times.  Register microkernel: 2 tokens x 32-wide output strip
 * (8 ymm accumulators), input dim blocked at JB=512 rows so the active
 * weight tile (JB x 32 x 4B = 64KB) stays in L2 across the token loop.
 * The attention phase (token-coupled, tiny) is untouched; the last
 * layer's MLP is a single token (pruning), so fusion is a no-op there.
 *
 * Same PLayer weight arrays are shared by v3 and v4 -> logits must match
 * to fp tolerance.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef DIM
#define DIM 1024
#endif
#define D   DIM
#define FF  (4*DIM)
#define SEQ 10
#define NOUT 10
#define NOUTP 16
#define LAYERS 2
#ifndef FUSE_UV
#define FUSE_UV 1
#endif
#ifndef FUSE_W1
#define FUSE_W1 1
#endif
#ifndef FUSE_W2
#define FUSE_W2 1
#endif
#ifndef ITERS
#define ITERS 40
#endif

static inline float fexp(float xx)
{
    float t=xx*1.4426950408889634f;
    float fl=floorf(t), f=t-fl;
    float p=1.0f+f*(0.69314718f+f*(0.24022651f+f*(0.05550411f+
            f*(0.00898934f+f*0.00187757f))));
    union{float f;int i;}u; u.i=((int)fl+127)<<23;
    return p*u.f;
}
static unsigned long long rng=0x243F6A8885A308D3ull;
static float frand(float s)
{
    rng=rng*6364136223846793005ull+1442695040888963407ull;
    return s*(((rng>>33)&0xFFFFFF)/8388608.0f-1.0f);
}
static void fill(float *p,long n,float s){for(long i=0;i<n;i++)p[i]=frand(s);}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+1e-9*t.tv_nsec;}
static float *alloc(long n){float*p;posix_memalign((void**)&p,64,n*4);return p;}

typedef struct { float *wuv,*buv,*w1,*b1,*w2,*b2; } PL;
static PL Lv[LAYERS];
static float *wout,*bout;
static float (*x_)[D], (*xh)[D], (*uv)[2*D];
static float *h1buf;                 /* [SEQ][FF] for v4, row 0 for v3 */

/* ---------- shared primitives ---------- */
static inline void norm(const float*restrict xin,float*restrict out)
{
    float s=0,ss=0;
    for(int i=0;i<D;i++){s+=xin[i];ss+=xin[i]*xin[i];}
    float mean=s/D, var=ss/D-mean*mean;
    float rstd=1.0f/sqrtf(var+1e-5f);
    for(int i=0;i<D;i++) out[i]=(xin[i]-mean)*rstd;
}
__attribute__((always_inline))
static inline void mvb(int id,int od,const float*restrict W,
                       const float*restrict bias,
                       const float*restrict xin,float*restrict y)
{
    for(int o=0;o<od;o+=64){
        int w=od-o; if(w>64)w=64;
        float a2[64];
        for(int i=0;i<w;i++)a2[i]=bias[o+i];
        for(int j=0;j<id;j++){
            float a=xin[j];
            const float*restrict wr=W+(long)j*od+o;
            for(int i=0;i<w;i++)a2[i]+=a*wr[i];
        }
        for(int i=0;i<w;i++)y[o+i]=a2[i];
    }
}

/* ---------- v4 GEMM-fused projection ----------
 * Y[t] = bias + W^T X[t] for t in [0,nt), W in axpy layout [id][od].
 * Loop order: input block -> output strip -> token pair -> input row.
 * Weight tile (JB x 32) is L2-resident across all token pairs. */
#define JB 512
__attribute__((always_inline))
static inline void gemm_tp(int id,int od,const float*restrict W,
                           const float*restrict bias,
                           const float*restrict X,long xs,
                           float*restrict Y,long ys,int nt)
{
    for(int t=0;t<nt;t++)
        for(int o=0;o<od;o++) Y[(long)t*ys+o]=bias[o];
    for(int j0=0;j0<id;j0+=JB){
        int j1=j0+JB<id?j0+JB:id;
        for(int o=0;o<od;o+=32){
            int t=0;
            for(;t+1<nt;t+=2){
                float a0[32],a1[32];
                float*restrict y0=Y+(long)t*ys+o;
                float*restrict y1=Y+(long)(t+1)*ys+o;
                for(int i=0;i<32;i++){a0[i]=y0[i];a1[i]=y1[i];}
                const float*restrict x0=X+(long)t*xs;
                const float*restrict x1=X+(long)(t+1)*xs;
                for(int j=j0;j<j1;j++){
                    float b0=x0[j], b1=x1[j];
                    const float*restrict wr=W+(long)j*od+o;
                    for(int i=0;i<32;i++){
                        a0[i]+=b0*wr[i];
                        a1[i]+=b1*wr[i];
                    }
                }
                for(int i=0;i<32;i++){y0[i]=a0[i];y1[i]=a1[i];}
            }
            if(t<nt){
                float a0[32];
                float*restrict y0=Y+(long)t*ys+o;
                for(int i=0;i<32;i++)a0[i]=y0[i];
                const float*restrict x0=X+(long)t*xs;
                for(int j=j0;j<j1;j++){
                    float b0=x0[j];
                    const float*restrict wr=W+(long)j*od+o;
                    for(int i=0;i<32;i++)a0[i]+=b0*wr[i];
                }
                for(int i=0;i<32;i++)y0[i]=a0[i];
            }
        }
    }
}

/* ---------- attention (shared by both) ---------- */
static void attention(int t0)
{
    for(int t=t0;t<SEQ;t++){
        const float*restrict u=uv[t];
        float w[SEQ],mx=-3.4e38f;
        for(int s=0;s<=t;s++){
            float acc=0;
            for(int j=0;j<D;j++)acc+=u[j]*xh[s][j];
            w[s]=acc; mx=acc>mx?acc:mx;
        }
        float sum=0;
        for(int s=0;s<=t;s++){w[s]=fexp(w[s]-mx);sum+=w[s];}
        static float tb[16384];
        for(int j=0;j<D;j++)tb[j]=0;
        for(int s=0;s<=t;s++){
            float a=w[s];
            const float*restrict vv=uv[s]+D;
            for(int j=0;j<D;j++)tb[j]+=a*vv[j];
        }
        float inv=1.0f/sum;
        for(int j=0;j<D;j++)x_[t][j]+=tb[j]*inv;
    }
}

/* ---------- v3 forward ---------- */
static void v3_forward(const float*restrict in,float*restrict logits)
{
    memcpy(x_,in,(long)SEQ*D*4);
    for(int l=0;l<LAYERS;l++){
        const PL*L=&Lv[l]; int last=(l==LAYERS-1), t0=last?SEQ-1:0;
        for(int t=0;t<SEQ;t++){ norm(x_[t],xh[t]);
            mvb(D,2*D,L->wuv,L->buv,xh[t],uv[t]); }
        attention(t0);
        for(int t=t0;t<SEQ;t++){
            static float xb[16384],tb[16384];
            norm(x_[t],xb);
            mvb(D,FF,L->w1,L->b1,xb,h1buf);
            for(int i=0;i<FF;i++)h1buf[i]=h1buf[i]>0?h1buf[i]:0;
            mvb(FF,D,L->w2,L->b2,h1buf,tb);
            for(int j=0;j<D;j++)x_[t][j]+=tb[j];
        }
    }
    static float xf[16384]; float lg[NOUTP];
    norm(x_[SEQ-1],xf);
    mvb(D,NOUTP,wout,bout,xf,lg);
    memcpy(logits,lg,NOUT*4);
}

/* ---------- v4 forward: token-fused projections ---------- */
static void v4_forward(const float*restrict in,float*restrict logits)
{
    memcpy(x_,in,(long)SEQ*D*4);
    for(int l=0;l<LAYERS;l++){
        const PL*L=&Lv[l]; int last=(l==LAYERS-1), t0=last?SEQ-1:0;
        for(int t=0;t<SEQ;t++) norm(x_[t],xh[t]);
#if FUSE_UV
        gemm_tp(D,2*D,L->wuv,L->buv,&xh[0][0],D,&uv[0][0],2*D,SEQ);
#else
        for(int t=0;t<SEQ;t++) mvb(D,2*D,L->wuv,L->buv,xh[t],uv[t]);
#endif
        attention(t0);
        int nt=SEQ-t0;
        static float xb[SEQ][FF+16];
        for(int t=t0;t<SEQ;t++) norm(x_[t],xb[t-t0]);
#if FUSE_W1
        gemm_tp(D,FF,L->w1,L->b1,&xb[0][0],FF+16,h1buf,FF,nt);
#else
        for(int t=0;t<nt;t++) mvb(D,FF,L->w1,L->b1,xb[t],h1buf+(long)t*FF);
#endif
        for(long i=0;i<(long)nt*FF;i++) h1buf[i]=h1buf[i]>0?h1buf[i]:0;
        static float tb[SEQ][D+16];
#if FUSE_W2
        gemm_tp(FF,D,L->w2,L->b2,h1buf,FF,&tb[0][0],D+16,nt);
#else
        for(int t=0;t<nt;t++) mvb(FF,D,L->w2,L->b2,h1buf+(long)t*FF,tb[t]);
#endif
        for(int t=t0;t<SEQ;t++)
            for(int j=0;j<D;j++)x_[t][j]+=tb[t-t0][j];
    }
    static float xf[16384]; float lg[NOUTP];
    norm(x_[SEQ-1],xf);
    mvb(D,NOUTP,wout,bout,xf,lg);
    memcpy(logits,lg,NOUT*4);
}

int main(void)
{
    for(int l=0;l<LAYERS;l++){
        PL*p=&Lv[l];
        p->wuv=alloc((long)D*2*D); p->buv=alloc(2*D);
        p->w1=alloc((long)D*FF);   p->b1=alloc(FF);
        p->w2=alloc((long)FF*D);   p->b2=alloc(D);
        fill(p->wuv,(long)D*2*D,0.01f); fill(p->buv,2*D,0.01f);
        fill(p->w1,(long)D*FF,0.03f);   fill(p->b1,FF,0.02f);
        fill(p->w2,(long)FF*D,0.01f);   fill(p->b2,D,0.02f);
    }
    wout=alloc((long)D*NOUTP); bout=alloc(NOUTP);
    fill(wout,(long)D*NOUTP,0.05f); fill(bout,NOUTP,0.02f);
    x_=(void*)alloc((long)SEQ*D); xh=(void*)alloc((long)SEQ*D);
    uv=(void*)alloc((long)SEQ*2*D); h1buf=alloc((long)SEQ*FF);

    float *in=alloc((long)SEQ*D); fill(in,(long)SEQ*D,0.5f);
    float l3[NOUT],l4[NOUT];
    v3_forward(in,l3); v4_forward(in,l4);
    double dd=0,mag=1e-9;
    for(int i=0;i<NOUT;i++){dd=fmax(dd,fabs(l3[i]-l4[i]));
                            mag=fmax(mag,fabs(l3[i]));}

    volatile float s=0; double t3=1e9,t4=1e9;
    for(int r=0;r<3;r++){
        double t0=now();
        for(int i=0;i<ITERS;i++){in[0]+=s*1e-30f;v3_forward(in,l3);s=l3[0];}
        t3=fmin(t3,(now()-t0)/ITERS);
        t0=now();
        for(int i=0;i<ITERS;i++){in[0]+=s*1e-30f;v4_forward(in,l4);s=l4[0];}
        t4=fmin(t4,(now()-t0)/ITERS);
    }
    double m=(SEQ*2.0*D*D+SEQ*2.0*D*FF+2*55.0*D)
            +(SEQ*2.0*D*D+2.0*D*FF+2*10.0*D);   /* same MACs both */
    printf("D=%4d | v3 per-token %9.1f us %5.1f GF/s | "
           "lp-config      %9.1f us %5.1f GF/s | v4/v3 %.2fx  relerr %.1e\n",
           D, t3*1e6, 2*m/t3/1e9, t4*1e6, 2*m/t4/1e9, t3/t4, dd/mag);
    return 0;
}
