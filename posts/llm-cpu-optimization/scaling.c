/* scaling.c — v2 (layout) vs v3 (algebra) as D grows.  DIM set via -DDIM=N
 * (multiple of 16).  FF = 4*DIM, SEQ=10, LAYERS=2.  For DIM<=256 we run
 * the full offline prep + numeric cross-check; above that, prep's O(D^3)
 * matrix products get slow, so we fill prepared weights randomly and only
 * measure runtime (algebra already verified at small D).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef DIM
#define DIM 128
#endif
#define D   DIM
#define FF  (4*DIM)
#define SEQ 10
#define NOUT 10
#define NOUTP 16
#define LAYERS 2
#ifndef ITERS
#define ITERS 200
#endif

/* ---------------- shared ---------------- */
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
    rng = rng*6364136223846793005ull + 1442695040888963407ull;
    return s * (((rng>>33)&0xFFFFFF)/8388608.0f - 1.0f);
}
static void fill(float *p, long n, float s){ for(long i=0;i<n;i++) p[i]=frand(s); }
static double now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+1e-9*ts.tv_nsec; }

/* ---------------- v2: separate q/k/v/o, runtime layernorm ---------------- */
typedef struct {
    float *wq,*wk,*wv,*wo;        /* [D][D] axpy layout */
    float *w1,*b1,*w2,*b2;        /* [D][FF],[FF] ; [FF][D],[D] */
    float *g1,*bt1,*g2,*bt2;
} V2L;
static V2L L2v[LAYERS];
static float *v2_wout, *v2_gf, *v2_bf;
static float (*x2_)[D], (*xn2)[D], (*q2)[D], (*k2)[D], (*v2a)[D];
static float att2[SEQ][SEQ], *h2;

__attribute__((always_inline)) static inline void mv(int id, int od, const float *restrict W,
                      const float *restrict xin, float *restrict y, int acc0)
{
    for (int o=0;o<od;o+=64){
        int w=od-o; if (w>64) w=64;
        float a2[64];
        if (acc0) for (int i=0;i<w;i++) a2[i]=0;
        else      for (int i=0;i<w;i++) a2[i]=y[o+i];
        for (int j=0;j<id;j++){
            float a=xin[j];
            const float *restrict wr=W+(long)j*od+o;
            for (int i=0;i<w;i++) a2[i]+=a*wr[i];
        }
        for (int i=0;i<w;i++) y[o+i]=a2[i];
    }
}
static inline void v2_ln(const float *restrict xin, const float *restrict g,
                         const float *restrict b, float *restrict y)
{
    float s=0,ss=0;
    for (int i=0;i<D;i++){ s+=xin[i]; ss+=xin[i]*xin[i]; }
    float mean=s/D, var=ss/D-mean*mean;
    float rstd=1.0f/sqrtf(var+1e-5f);
    for (int i=0;i<D;i++) y[i]=(xin[i]-mean)*rstd*g[i]+b[i];
}
static void v2_layer(const V2L *L)
{
    const float scale=1.0f/sqrtf((float)D);
    for (int t=0;t<SEQ;t++){
        v2_ln(x2_[t],L->g1,L->bt1,xn2[t]);
        mv(D,D,L->wq,xn2[t],q2[t],1);
        mv(D,D,L->wk,xn2[t],k2[t],1);
        mv(D,D,L->wv,xn2[t],v2a[t],1);
    }
    for (int t=0;t<SEQ;t++){
        float mx=-3.4e38f;
        for (int s=0;s<=t;s++){
            float acc=0;
            for (int j=0;j<D;j++) acc+=q2[t][j]*k2[s][j];
            att2[t][s]=acc*scale; mx=att2[t][s]>mx?att2[t][s]:mx;
        }
        float sum=0;
        for (int s=0;s<=t;s++){ att2[t][s]=fexp(att2[t][s]-mx); sum+=att2[t][s]; }
        float inv=1.0f/sum;
        for (int s=0;s<=t;s++) att2[t][s]*=inv;
    }
    for (int t=0;t<SEQ;t++){
        float *tmp=xn2[t], ao[112];  /* reuse xn2 row as scratch */
        for (int j=0;j<D;j++) tmp[j]=0;
        for (int s=0;s<=t;s++){
            float a=att2[t][s];
            for (int j=0;j<D;j++) tmp[j]+=a*v2a[s][j];
        }
        (void)ao;
        float *out=q2[t];            /* scratch */
        mv(D,D,L->wo,tmp,out,1);
        for (int j=0;j<D;j++) x2_[t][j]+=out[j];
    }
    for (int t=0;t<SEQ;t++){
        v2_ln(x2_[t],L->g2,L->bt2,xn2[t]);
        mv(D,FF,L->w1,xn2[t],h2,1);
        for (int i=0;i<FF;i++){ float z=h2[i]+L->b1[i]; h2[i]=z>0?z:0; }
        float *tmp=xn2[t];
        mv(FF,D,L->w2,h2,tmp,1);
        for (int j=0;j<D;j++) x2_[t][j]+=tmp[j]+L->b2[j];
    }
}
static void v2_forward(const float *restrict in, float *restrict logits)
{
    memcpy(x2_,in,(long)SEQ*D*sizeof(float));
    for (int l=0;l<LAYERS;l++) v2_layer(&L2v[l]);
    float xf[16384>D?16384:D], lg[NOUTP];
    v2_ln(x2_[SEQ-1],v2_gf,v2_bf,xf);
    mv(D,NOUTP,v2_wout,xf,lg,1);
    memcpy(logits,lg,NOUT*sizeof(float));
}

/* ---------------- v3: merged uv, folded ln, last-token pruning ------------ */
typedef struct {
    float *wuv,*buv;              /* [D][2D],[2D] */
    float *w1,*b1,*w2,*b2;
} V3L;
static V3L L3v[LAYERS];
static float *v3_wout,*v3_bout;
static float (*x3_)[D], (*xh3)[D], (*uv3)[2*D];
static float *h3;

static inline void v3_norm(const float *restrict xin, float *restrict out)
{
    float s=0,ss=0;
    for (int i=0;i<D;i++){ s+=xin[i]; ss+=xin[i]*xin[i]; }
    float mean=s/D, var=ss/D-mean*mean;
    float rstd=1.0f/sqrtf(var+1e-5f);
    for (int i=0;i<D;i++) out[i]=(xin[i]-mean)*rstd;
}
__attribute__((always_inline)) static inline void mvb(int id, int od, const float *restrict W,
                       const float *restrict bias,
                       const float *restrict xin, float *restrict y)
{
    for (int o=0;o<od;o+=64){
        int w=od-o; if (w>64) w=64;
        float a2[64];
        for (int i=0;i<w;i++) a2[i]=bias[o+i];
        for (int j=0;j<id;j++){
            float a=xin[j];
            const float *restrict wr=W+(long)j*od+o;
            for (int i=0;i<w;i++) a2[i]+=a*wr[i];
        }
        for (int i=0;i<w;i++) y[o+i]=a2[i];
    }
}
static void v3_layer(const V3L *L, int last)
{
    for (int t=0;t<SEQ;t++){
        v3_norm(x3_[t],xh3[t]);
        mvb(D,2*D,L->wuv,L->buv,xh3[t],uv3[t]);
    }
    int t0=last?SEQ-1:0;
    for (int t=t0;t<SEQ;t++){
        const float *restrict u=uv3[t];
        float w[SEQ],mx=-3.4e38f;
        for (int s=0;s<=t;s++){
            float acc=0;
            for (int j=0;j<D;j++) acc+=u[j]*xh3[s][j];
            w[s]=acc; mx=acc>mx?acc:mx;
        }
        float sum=0;
        for (int s=0;s<=t;s++){ w[s]=fexp(w[s]-mx); sum+=w[s]; }
        float *tmp=xh3[t];          /* done with query's xh? NO - keys! use scratch */
        float tacc[112];
        (void)tacc; (void)tmp;
        static float tbuf[16384];
        for (int j=0;j<D;j++) tbuf[j]=0;
        for (int s=0;s<=t;s++){
            float a=w[s];
            const float *restrict vv=uv3[s]+D;
            for (int j=0;j<D;j++) tbuf[j]+=a*vv[j];
        }
        float inv=1.0f/sum;
        for (int j=0;j<D;j++) x3_[t][j]+=tbuf[j]*inv;
    }
    for (int t=t0;t<SEQ;t++){
        static float xhb[16384], tbuf[16384];
        v3_norm(x3_[t],xhb);
        mvb(D,FF,L->w1,L->b1,xhb,h3);
        for (int i=0;i<FF;i++) h3[i]=h3[i]>0?h3[i]:0;
        mvb(FF,D,L->w2,L->b2,h3,tbuf);
        for (int j=0;j<D;j++) x3_[t][j]+=tbuf[j];
    }
}
static void v3_forward(const float *restrict in, float *restrict logits)
{
    memcpy(x3_,in,(long)SEQ*D*sizeof(float));
    for (int l=0;l<LAYERS;l++) v3_layer(&L3v[l], l==LAYERS-1);
    static float xf[16384]; float lg[NOUTP];
    v3_norm(x3_[SEQ-1],xf);
    mvb(D,NOUTP,v3_wout,v3_bout,xf,lg);
    memcpy(logits,lg,NOUT*sizeof(float));
}

/* ---------------- setup ---------------- */
static float *alloc(long n){ float *p; posix_memalign((void**)&p,64,n*sizeof(float));
    return p; }

int main(void)
{
    for (int l=0;l<LAYERS;l++){
        V2L *a=&L2v[l];
        a->wq=alloc((long)D*D); a->wk=alloc((long)D*D);
        a->wv=alloc((long)D*D); a->wo=alloc((long)D*D);
        a->w1=alloc((long)D*FF); a->b1=alloc(FF);
        a->w2=alloc((long)FF*D); a->b2=alloc(D);
        a->g1=alloc(D); a->bt1=alloc(D); a->g2=alloc(D); a->bt2=alloc(D);
        fill(a->wq,(long)D*D,0.05f); fill(a->wk,(long)D*D,0.05f);
        fill(a->wv,(long)D*D,0.05f); fill(a->wo,(long)D*D,0.05f);
        fill(a->w1,(long)D*FF,0.05f); fill(a->b1,FF,0.02f);
        fill(a->w2,(long)FF*D,0.02f); fill(a->b2,D,0.02f);
        for (int i=0;i<D;i++){ a->g1[i]=1+frand(0.2f); a->bt1[i]=frand(0.1f);
                               a->g2[i]=1+frand(0.2f); a->bt2[i]=frand(0.1f); }
        V3L *b=&L3v[l];
        b->wuv=alloc((long)D*2*D); b->buv=alloc(2*D);
        b->w1=alloc((long)D*FF); b->b1=alloc(FF);
        b->w2=alloc((long)FF*D); b->b2=alloc(D);
    }
    v2_wout=alloc((long)D*NOUTP); v2_gf=alloc(D); v2_bf=alloc(D);
    fill(v2_wout,(long)D*NOUTP,0.05f);
    for (int i=0;i<D;i++){ v2_gf[i]=1+frand(0.2f); v2_bf[i]=frand(0.1f); }
    v3_wout=alloc((long)D*NOUTP); v3_bout=alloc(NOUTP);

    /* offline prep (with cross-check) only when O(D^3) is cheap */
    int checked = 0;
#if DIM <= 256
    for (int l=0;l<LAYERS;l++){
        V2L *a=&L2v[l]; V3L *b=&L3v[l];
        const float scale=1.0f/sqrtf((float)D);
        /* A = Wq^T Wk in axpy layout: a->wq[m][i] holds Wq[i][m]?  Our v2
         * weights are already axpy [in][out] = W^T row-major, i.e.
         * a->wq[j*D+o] = Wq[o][j].  q.k = xn_t^T Wq^T Wk xn_s:
         * A[i][j] = sum_m Wq[m][i] Wk[m][j] = sum_m a->wq[i*D+m'] ...
         * easier: reconstruct Wq[o][i]=a->wq[i*D+o]. */
        static float A[256*256], Wov[256*256];
        for (int i=0;i<D;i++) for (int j=0;j<D;j++){
            float acc=0;
            for (int m=0;m<D;m++) acc+=a->wq[(long)i*D+m]*a->wk[(long)j*D+m];
            A[i*D+j]=acc;   /* = sum_m Wq[m][i] Wk[m][j] */
        }
        for (int j=0;j<D;j++) for (int o=0;o<D;o++)
            b->wuv[(long)j*2*D+o]=scale*a->g1[j]*A[j*D+o]*a->g1[o];
        for (int o=0;o<D;o++){
            float acc=0;
            for (int i=0;i<D;i++) acc+=a->bt1[i]*A[i*D+o];
            b->buv[o]=scale*acc*a->g1[o];
        }
        for (int o=0;o<D;o++) for (int i=0;i<D;i++){
            float acc=0;   /* Wov[o][i] = sum_m Wo[o][m] Wv[m][i]
                              Wo[o][m]=a->wo[m*D+o], Wv[m][i]=a->wv[i*D+m] */
            for (int m=0;m<D;m++) acc+=a->wo[(long)m*D+o]*a->wv[(long)i*D+m];
            Wov[o*D+i]=acc;
        }
        for (int j=0;j<D;j++) for (int o=0;o<D;o++)
            b->wuv[(long)j*2*D+D+o]=Wov[o*D+j]*a->g1[j];
        for (int o=0;o<D;o++){
            float acc=0;
            for (int i=0;i<D;i++) acc+=Wov[o*D+i]*a->bt1[i];
            b->buv[D+o]=acc;
        }
        for (int j=0;j<D;j++) for (int f=0;f<FF;f++)
            b->w1[(long)j*FF+f]=a->w1[(long)j*FF+f]*a->g2[j];
        for (int f=0;f<FF;f++){
            float acc=a->b1[f];   /* + W1[f][j] bt2[j], W1[f][j]=a->w1[j*FF+f] */
            for (int j=0;j<D;j++) acc+=a->w1[(long)j*FF+f]*a->bt2[j];
            b->b1[f]=acc;
        }
        memcpy(b->w2,a->w2,(long)FF*D*sizeof(float));
        memcpy(b->b2,a->b2,D*sizeof(float));
    }
    for (int j=0;j<D;j++) for (int o=0;o<NOUTP;o++)
        v3_wout[(long)j*NOUTP+o]=v2_wout[(long)j*NOUTP+o]*v2_gf[j];
    for (int o=0;o<NOUTP;o++){
        float acc=0;
        for (int j=0;j<D;j++) acc+=v2_wout[(long)j*NOUTP+o]*v2_bf[j];
        v3_bout[o]=acc;
    }
    checked=1;
#else
    for (int l=0;l<LAYERS;l++){
        V3L *b=&L3v[l];
        fill(b->wuv,(long)D*2*D,0.01f); fill(b->buv,2*D,0.01f);
        fill(b->w1,(long)D*FF,0.05f);   fill(b->b1,FF,0.02f);
        fill(b->w2,(long)FF*D,0.02f);   fill(b->b2,D,0.02f);
    }
    fill(v3_wout,(long)D*NOUTP,0.05f); fill(v3_bout,NOUTP,0.02f);
#endif

    x2_=(void*)alloc((long)SEQ*D); xn2=(void*)alloc((long)SEQ*D);
    q2=(void*)alloc((long)SEQ*D);  k2=(void*)alloc((long)SEQ*D);
    v2a=(void*)alloc((long)SEQ*D); h2=alloc(FF);
    x3_=(void*)alloc((long)SEQ*D); xh3=(void*)alloc((long)SEQ*D);
    uv3=(void*)alloc((long)SEQ*2*D); h3=alloc(FF);

    float *in=alloc((long)SEQ*D);
    fill(in,(long)SEQ*D,0.5f);
    float l2[NOUT],l3[NOUT];
    v2_forward(in,l2); v3_forward(in,l3);
    double dd=0,mag=0;
    for (int i=0;i<NOUT;i++){ dd=fmax(dd,fabs(l2[i]-l3[i]));
                              mag=fmax(mag,fabs(l2[i])); }

    volatile float sink=0;
    double t2=1e9,t3=1e9;
    for (int r=0;r<3;r++){
        double t0=now();
        for (int i=0;i<ITERS;i++){ in[0]+=sink*1e-30f; v2_forward(in,l2); sink=l2[0]; }
        t2=fmin(t2,(now()-t0)/ITERS);
        t0=now();
        for (int i=0;i<ITERS;i++){ in[0]+=sink*1e-30f; v3_forward(in,l3); sink=l3[0]; }
        t3=fmin(t3,(now()-t0)/ITERS);
    }
    double m2=2.0*(SEQ*(4.0*D*D+2.0*D*FF)+2*55.0*D);          /* MACs */
    double m3=(SEQ*2.0*D*D + SEQ*2.0*D*FF + 2*55.0*D)
            + (SEQ*2.0*D*D + 1.0*2.0*D*FF + 2*10.0*D);
    double by2=2.0*(4.0*D*D+8.0*D*D)*4.0;   /* weight bytes/pass v2 (FF=4D) */
    double by3=2.0*(2.0*D*D+8.0*D*D)*4.0;
    printf("D=%4d  wts v2/v3 %5.1f/%5.1f MB | v2 %9.1f us %6.1f GF/s %5.1f GB/s"
           " | v3 %9.1f us %6.1f GF/s %5.1f GB/s | v3/v2 %.2fx %s\n",
           D, by2/1e6, by3/1e6,
           t2*1e6, 2*m2/t2/1e9, by2/t2/1e9,
           t3*1e6, 2*m3/t3/1e9, by3/t3/1e9,
           t2/t3, checked ? (dd/mag<1e-3?"[check ok]":"[CHECK FAIL]") : "");
    return 0;
}
