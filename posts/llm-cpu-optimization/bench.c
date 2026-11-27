/* bench.c — v1 vs v2 vs v3 tiny transformer: correctness + timing.
 * v1: naive row-major dot-product form (ln_f load bug FIXED)
 * v2: col-major axpy form, padded dims (acc overflow FIXED: acc[FF])
 * v3: merged QK/OV, folded layernorms, last-token pruning, deferred
 *     softmax norm, fast_exp, strip-mined matvec
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SEQ 10
#define D 100
#define DP 112
#define RAW_FF 400
#define FF3 448
#define NOUT 10
#define NOUTP 16
#define LAYERS 2

/* ================= raw model (v1 layout) ================= */
typedef struct {
    float wq[D][D], wk[D][D], wv[D][D], wo[D][D];
    float w1[RAW_FF][D], b1[RAW_FF];
    float w2[D][RAW_FF], b2[D];
    float ln1_g[D], ln1_b[D], ln2_g[D], ln2_b[D];
} RawLayer;
typedef struct {
    RawLayer layer[LAYERS];
    float w_out[NOUT][D];
    float ln_f_g[D], ln_f_b[D];
} RawModel;
static RawModel R;

/* ============================ v1 ============================ */
static float x1_[SEQ][D], xn1[SEQ][D];
static float q1[SEQ][D], k1[SEQ][D], v1a[SEQ][D];
static float att1[SEQ][SEQ], ao1[SEQ][D], h1[SEQ][RAW_FF];

static void v1_matvec(int od, int id, const float *restrict W,
                      const float *restrict xin, float *restrict y)
{
    for (int i = 0; i < od; i++) {
        float acc = 0.0f;
        for (int j = 0; j < id; j++) acc += W[i*id+j]*xin[j];
        y[i] = acc;
    }
}
static void v1_ln(const float *restrict xin, const float *restrict g,
                  const float *restrict b, float *restrict y)
{
    float mean=0, var=0;
    for (int i=0;i<D;i++) mean += xin[i];
    mean /= D;
    for (int i=0;i<D;i++){ float d=xin[i]-mean; var+=d*d; }
    var /= D;
    float rstd = 1.0f/sqrtf(var+1e-5f);
    for (int i=0;i<D;i++) y[i]=(xin[i]-mean)*rstd*g[i]+b[i];
}
static void v1_softmax(float *a, int n)
{
    float mx=a[0]; for(int i=1;i<n;i++) if(a[i]>mx) mx=a[i];
    float s=0; for(int i=0;i<n;i++){ a[i]=expf(a[i]-mx); s+=a[i]; }
    float inv=1.0f/s; for(int i=0;i<n;i++) a[i]*=inv;
}
static void v1_layer(const RawLayer *L)
{
    const float scale = 1.0f/sqrtf((float)D);
    for (int t=0;t<SEQ;t++){
        v1_ln(x1_[t], L->ln1_g, L->ln1_b, xn1[t]);
        v1_matvec(D,D,&L->wq[0][0],xn1[t],q1[t]);
        v1_matvec(D,D,&L->wk[0][0],xn1[t],k1[t]);
        v1_matvec(D,D,&L->wv[0][0],xn1[t],v1a[t]);
    }
    for (int t=0;t<SEQ;t++){
        for (int s=0;s<=t;s++){
            float acc=0;
            for (int j=0;j<D;j++) acc += q1[t][j]*k1[s][j];
            att1[t][s]=acc*scale;
        }
        v1_softmax(att1[t], t+1);
    }
    for (int t=0;t<SEQ;t++){
        float tmp[D]; memset(tmp,0,sizeof tmp);
        for (int s=0;s<=t;s++){
            float a=att1[t][s];
            for (int j=0;j<D;j++) tmp[j]+=a*v1a[s][j];
        }
        v1_matvec(D,D,&L->wo[0][0],tmp,ao1[t]);
        for (int j=0;j<D;j++) x1_[t][j]+=ao1[t][j];
    }
    for (int t=0;t<SEQ;t++){
        v1_ln(x1_[t], L->ln2_g, L->ln2_b, xn1[t]);
        v1_matvec(RAW_FF,D,&L->w1[0][0],xn1[t],h1[t]);
        for (int i=0;i<RAW_FF;i++){
            float z=h1[t][i]+L->b1[i];
            h1[t][i]= z>0?z:0;
        }
        float tmp[D];
        v1_matvec(D,RAW_FF,&L->w2[0][0],h1[t],tmp);
        for (int j=0;j<D;j++) x1_[t][j]+=tmp[j]+L->b2[j];
    }
}
void v1_forward(const float *restrict in, float *restrict logits)
{
    memcpy(x1_, in, sizeof x1_);
    for (int l=0;l<LAYERS;l++) v1_layer(&R.layer[l]);
    float xf[D];
    v1_ln(x1_[SEQ-1], R.ln_f_g, R.ln_f_b, xf);   /* bug fixed: loaded */
    v1_matvec(NOUT,D,&R.w_out[0][0],xf,logits);
}

/* ============================ v2 ============================ */
#define FF2 RAW_FF  /* 400, as in the original v2 file */
typedef struct {
    float wq[DP][DP], wk[DP][DP], wv[DP][DP], wo[DP][DP];
    float w1[DP][FF2], b1[FF2];
    float w2[FF2][DP], b2[DP];
    float ln1_g[DP], ln1_b[DP], ln2_g[DP], ln2_b[DP];
} V2Layer;
static struct { V2Layer layer[LAYERS]; float w_out[DP][NOUTP];
                float ln_f_g[DP], ln_f_b[DP]; } M2;

static float x2_[SEQ][DP], xn2[SEQ][DP];
static float q2[SEQ][DP], k2[SEQ][DP], v2a[SEQ][DP];
static float att2[SEQ][SEQ], h2[SEQ][FF2];

static inline void v2_matvec(int id, int od, const float *restrict W,
                             const float *restrict xin, float *restrict y)
{
    float acc[FF2];                       /* FIXED: was acc[DP] */
    for (int i=0;i<od;i++) acc[i]=0.0f;
    for (int j=0;j<id;j++){
        float a=xin[j];
        const float *restrict w = W + (long)j*od;
        for (int i=0;i<od;i++) acc[i]+=a*w[i];
    }
    for (int i=0;i<od;i++) y[i]=acc[i];
}
static inline void v2_ln(const float *restrict xin, const float *restrict g,
                         const float *restrict b, float *restrict y)
{
    float mean=0, var=0;
    for (int i=0;i<DP;i++) mean+=xin[i];
    mean *= (1.0f/D);
    for (int i=0;i<D;i++){ float d=xin[i]-mean; var+=d*d; }
    float rstd=1.0f/sqrtf(var*(1.0f/D)+1e-5f);
    for (int i=0;i<DP;i++) y[i]=(xin[i]-mean)*rstd*g[i]+b[i];
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
static void v2_softmax(float *a, int n)
{
    float mx=a[0]; for(int i=1;i<n;i++) mx=a[i]>mx?a[i]:mx;
    float s=0; for(int i=0;i<n;i++){ a[i]=fexp(a[i]-mx); s+=a[i]; }
    float inv=1.0f/s; for(int i=0;i<n;i++) a[i]*=inv;
}
static void v2_layer(const V2Layer *L)
{
    const float scale=1.0f/sqrtf((float)D);
    for (int t=0;t<SEQ;t++){
        v2_ln(x2_[t],L->ln1_g,L->ln1_b,xn2[t]);
        v2_matvec(DP,DP,&L->wq[0][0],xn2[t],q2[t]);
        v2_matvec(DP,DP,&L->wk[0][0],xn2[t],k2[t]);
        v2_matvec(DP,DP,&L->wv[0][0],xn2[t],v2a[t]);
    }
    for (int t=0;t<SEQ;t++){
        for (int s=0;s<=t;s++){
            float acc=0;
            for (int j=0;j<DP;j++) acc+=q2[t][j]*k2[s][j];
            att2[t][s]=acc*scale;
        }
        v2_softmax(att2[t],t+1);
    }
    for (int t=0;t<SEQ;t++){
        float tmp[DP], ao[DP];
        for (int j=0;j<DP;j++) tmp[j]=0;
        for (int s=0;s<=t;s++){
            float a=att2[t][s];
            for (int j=0;j<DP;j++) tmp[j]+=a*v2a[s][j];
        }
        v2_matvec(DP,DP,&L->wo[0][0],tmp,ao);
        for (int j=0;j<DP;j++) x2_[t][j]+=ao[j];
    }
    for (int t=0;t<SEQ;t++){
        v2_ln(x2_[t],L->ln2_g,L->ln2_b,xn2[t]);
        v2_matvec(DP,FF2,&L->w1[0][0],xn2[t],h2[t]);
        for (int i=0;i<FF2;i++){
            float z=h2[t][i]+L->b1[i];
            h2[t][i]=z>0?z:0;
        }
        float tmp[DP];
        v2_matvec(FF2,DP,&L->w2[0][0],h2[t],tmp);
        for (int j=0;j<DP;j++) x2_[t][j]+=tmp[j]+L->b2[j];
    }
}
void v2_forward(const float *restrict in, float *restrict logits)
{
    for (int t=0;t<SEQ;t++){
        memcpy(x2_[t], in+(long)t*D, D*sizeof(float));
        for (int j=D;j<DP;j++) x2_[t][j]=0;
    }
    for (int l=0;l<LAYERS;l++) v2_layer(&M2.layer[l]);
    float xf[DP], lg[NOUTP];
    v2_ln(x2_[SEQ-1],M2.ln_f_g,M2.ln_f_b,xf);
    v2_matvec(DP,NOUTP,&M2.w_out[0][0],xf,lg);
    memcpy(logits,lg,NOUT*sizeof(float));
}

/* ============================ v3 ============================ */
typedef struct {
    float wuv[DP][2*DP]; float buv[2*DP];
    float w1[DP][FF3], b1[FF3];
    float w2[FF3][DP], b2[DP];
} PLayer;
static struct { PLayer layer[LAYERS];
                float w_out[DP][NOUTP], b_out[NOUTP]; } M3;

static float x3_[SEQ][DP], xh3[SEQ][DP], uv3[SEQ][2*DP], hb3[FF3];

static inline void v3_matvec(int id, int od, const float *restrict W,
                             const float *restrict bias,
                             const float *restrict xin, float *restrict y)
{
    for (int o=0;o<od;o+=112){
        int w = od-o; if (w>112) w=112;
        float acc[112];
        for (int i=0;i<w;i++) acc[i]=bias[o+i];
        for (int j=0;j<id;j++){
            float a=xin[j];
            const float *restrict wr = W+(long)j*od+o;
            for (int i=0;i<w;i++) acc[i]+=a*wr[i];
        }
        for (int i=0;i<w;i++) y[o+i]=acc[i];
    }
}
static inline void v3_norm(const float *restrict xin, float *restrict out)
{
    float s=0, ss=0;
    for (int i=0;i<DP;i++){ s+=xin[i]; ss+=xin[i]*xin[i]; }
    float mean=s*(1.0f/D);
    float var=ss*(1.0f/D)-mean*mean;
    float rstd=1.0f/sqrtf(var+1e-5f);
    for (int i=0;i<DP;i++) out[i]=(xin[i]-mean)*rstd;
}
static void v3_layer(const PLayer *restrict L, int last)
{
    for (int t=0;t<SEQ;t++){
        v3_norm(x3_[t], xh3[t]);
        v3_matvec(DP,2*DP,&L->wuv[0][0],L->buv,xh3[t],uv3[t]);
    }
    int t0 = last ? SEQ-1 : 0;
    for (int t=t0;t<SEQ;t++){
        const float *restrict u = uv3[t];
        float w[SEQ], mx=-3.4e38f;
        for (int s=0;s<=t;s++){
            float acc=0;
            for (int j=0;j<DP;j++) acc+=u[j]*xh3[s][j];
            w[s]=acc; mx=acc>mx?acc:mx;
        }
        float sum=0;
        for (int s=0;s<=t;s++){ w[s]=fexp(w[s]-mx); sum+=w[s]; }
        float tmp[DP];
        for (int j=0;j<DP;j++) tmp[j]=0;
        for (int s=0;s<=t;s++){
            float a=w[s];
            const float *restrict vv=uv3[s]+DP;
            for (int j=0;j<DP;j++) tmp[j]+=a*vv[j];
        }
        float inv=1.0f/sum;
        for (int j=0;j<DP;j++) x3_[t][j]+=tmp[j]*inv;
    }
    for (int t=t0;t<SEQ;t++){
        float xh2b[DP], tmp[DP];
        v3_norm(x3_[t], xh2b);
        v3_matvec(DP,FF3,&L->w1[0][0],L->b1,xh2b,hb3);
        for (int i=0;i<FF3;i++) hb3[i]=hb3[i]>0?hb3[i]:0;
        v3_matvec(FF3,DP,&L->w2[0][0],L->b2,hb3,tmp);
        for (int j=0;j<DP;j++) x3_[t][j]+=tmp[j];
    }
}
void v3_forward(const float *restrict in, float *restrict logits)
{
    for (int t=0;t<SEQ;t++){
        memcpy(x3_[t], in+(long)t*D, D*sizeof(float));
        for (int j=D;j<DP;j++) x3_[t][j]=0;
    }
    for (int l=0;l<LAYERS;l++) v3_layer(&M3.layer[l], l==LAYERS-1);
    float xf[DP], lg[NOUTP];
    v3_norm(x3_[SEQ-1], xf);
    v3_matvec(DP,NOUTP,&M3.w_out[0][0],M3.b_out,xf,lg);
    memcpy(logits,lg,NOUT*sizeof(float));
}

/* ================= weight preparation ================= */
static float A_[D][D], Wov_[D][D];

static void prep_v2(void)
{
    memset(&M2,0,sizeof M2);
    for (int l=0;l<LAYERS;l++){
        const RawLayer *rl=&R.layer[l]; V2Layer *p=&M2.layer[l];
        for (int i=0;i<D;i++) for (int j=0;j<D;j++){
            p->wq[j][i]=rl->wq[i][j];  /* row-major [o][i] -> axpy [i][o] */
            p->wk[j][i]=rl->wk[i][j];
            p->wv[j][i]=rl->wv[i][j];
            p->wo[j][i]=rl->wo[i][j];
        }
        for (int f=0;f<RAW_FF;f++) for (int j=0;j<D;j++)
            p->w1[j][f]=rl->w1[f][j];
        for (int f=0;f<RAW_FF;f++) p->b1[f]=rl->b1[f];
        for (int o=0;o<D;o++) for (int f=0;f<RAW_FF;f++)
            p->w2[f][o]=rl->w2[o][f];
        for (int o=0;o<D;o++) p->b2[o]=rl->b2[o];
        memcpy(p->ln1_g,rl->ln1_g,sizeof rl->ln1_g);
        memcpy(p->ln1_b,rl->ln1_b,sizeof rl->ln1_b);
        memcpy(p->ln2_g,rl->ln2_g,sizeof rl->ln2_g);
        memcpy(p->ln2_b,rl->ln2_b,sizeof rl->ln2_b);
    }
    for (int o=0;o<NOUT;o++) for (int j=0;j<D;j++)
        M2.w_out[j][o]=R.w_out[o][j];
    memcpy(M2.ln_f_g,R.ln_f_g,sizeof R.ln_f_g);
    memcpy(M2.ln_f_b,R.ln_f_b,sizeof R.ln_f_b);
}

static void prep_v3(void)
{
    memset(&M3,0,sizeof M3);
    const float scale=1.0f/sqrtf((float)D);
    for (int l=0;l<LAYERS;l++){
        const RawLayer *rl=&R.layer[l]; PLayer *p=&M3.layer[l];
        const float *g1=rl->ln1_g,*b1n=rl->ln1_b;
        const float *g2=rl->ln2_g,*b2n=rl->ln2_b;
        for (int i=0;i<D;i++) for (int j=0;j<D;j++){
            float acc=0;
            for (int m=0;m<D;m++) acc+=rl->wq[m][i]*rl->wk[m][j];
            A_[i][j]=acc;
        }
        for (int j=0;j<D;j++) for (int o=0;o<D;o++)
            p->wuv[j][o]=scale*g1[j]*A_[j][o]*g1[o];
        for (int o=0;o<D;o++){
            float acc=0;
            for (int i=0;i<D;i++) acc+=b1n[i]*A_[i][o];
            p->buv[o]=scale*acc*g1[o];
        }
        for (int o=0;o<D;o++) for (int i=0;i<D;i++){
            float acc=0;
            for (int m=0;m<D;m++) acc+=rl->wo[o][m]*rl->wv[m][i];
            Wov_[o][i]=acc;
        }
        for (int j=0;j<D;j++) for (int o=0;o<D;o++)
            p->wuv[j][DP+o]=Wov_[o][j]*g1[j];
        for (int o=0;o<D;o++){
            float acc=0;
            for (int i=0;i<D;i++) acc+=Wov_[o][i]*b1n[i];
            p->buv[DP+o]=acc;
        }
        for (int j=0;j<D;j++) for (int f=0;f<RAW_FF;f++)
            p->w1[j][f]=rl->w1[f][j]*g2[j];
        for (int f=0;f<RAW_FF;f++){
            float acc=rl->b1[f];
            for (int j=0;j<D;j++) acc+=rl->w1[f][j]*b2n[j];
            p->b1[f]=acc;
        }
        for (int f=0;f<RAW_FF;f++) for (int o=0;o<D;o++)
            p->w2[f][o]=rl->w2[o][f];
        for (int o=0;o<D;o++) p->b2[o]=rl->b2[o];
    }
    for (int j=0;j<D;j++) for (int o=0;o<NOUT;o++)
        M3.w_out[j][o]=R.w_out[o][j]*R.ln_f_g[j];
    for (int o=0;o<NOUT;o++){
        float acc=0;
        for (int j=0;j<D;j++) acc+=R.w_out[o][j]*R.ln_f_b[j];
        M3.b_out[o]=acc;
    }
}

/* ================= harness ================= */
static unsigned long long rng=0x243F6A8885A308D3ull;
static float frand(float s)
{
    rng = rng*6364136223846793005ull + 1442695040888963407ull;
    return s * (((rng>>33) & 0xFFFFFF) / 8388608.0f - 1.0f);
}
static void fill(float *p, long n, float s){ for(long i=0;i<n;i++) p[i]=frand(s); }

static double now(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + 1e-9*ts.tv_nsec;
}

int main(void)
{
    for (int l=0;l<LAYERS;l++){
        RawLayer *rl=&R.layer[l];
        fill(&rl->wq[0][0],D*D,0.1f); fill(&rl->wk[0][0],D*D,0.1f);
        fill(&rl->wv[0][0],D*D,0.1f); fill(&rl->wo[0][0],D*D,0.1f);
        fill(&rl->w1[0][0],RAW_FF*D,0.1f); fill(rl->b1,RAW_FF,0.05f);
        fill(&rl->w2[0][0],D*RAW_FF,0.05f); fill(rl->b2,D,0.05f);
        for (int i=0;i<D;i++){
            rl->ln1_g[i]=1.0f+frand(0.2f); rl->ln1_b[i]=frand(0.1f);
            rl->ln2_g[i]=1.0f+frand(0.2f); rl->ln2_b[i]=frand(0.1f);
        }
    }
    fill(&R.w_out[0][0],NOUT*D,0.1f);
    for (int i=0;i<D;i++){ R.ln_f_g[i]=1.0f+frand(0.2f); R.ln_f_b[i]=frand(0.1f); }

    prep_v2(); prep_v3();

    static float in[SEQ][D];
    fill(&in[0][0],SEQ*D,1.0f);

    float l1[NOUT], l2[NOUT], l3[NOUT];
    v1_forward(&in[0][0],l1);
    v2_forward(&in[0][0],l2);
    v3_forward(&in[0][0],l3);

    double d12=0,d13=0,mag=0;
    for (int i=0;i<NOUT;i++){
        d12=fmax(d12,fabs(l1[i]-l2[i]));
        d13=fmax(d13,fabs(l1[i]-l3[i]));
        mag=fmax(mag,fabs(l1[i]));
    }
    printf("logits v1: "); for(int i=0;i<NOUT;i++) printf("% .4f ",l1[i]);
    printf("\nlogits v3: "); for(int i=0;i<NOUT;i++) printf("% .4f ",l3[i]);
    printf("\nmax |v1-v2| = %.3e   max |v1-v3| = %.3e   (|logit| max %.3f)\n\n",
           d12,d13,mag);

    const int ITERS=20000, REPS=5;
    volatile float sink=0;
    double t1=1e9,t2=1e9,t3=1e9;
    for (int r=0;r<REPS;r++){
        double t0=now();
        for (int i=0;i<ITERS;i++){ in[0][0]+=sink*1e-30f;
            v1_forward(&in[0][0],l1); sink=l1[0]; }
        t1=fmin(t1,(now()-t0)/ITERS);
        t0=now();
        for (int i=0;i<ITERS;i++){ in[0][0]+=sink*1e-30f;
            v2_forward(&in[0][0],l2); sink=l2[0]; }
        t2=fmin(t2,(now()-t0)/ITERS);
        t0=now();
        for (int i=0;i<ITERS;i++){ in[0][0]+=sink*1e-30f;
            v3_forward(&in[0][0],l3); sink=l3[0]; }
        t3=fmin(t3,(now()-t0)/ITERS);
    }
    /* MAC counts, matmul terms only */
    double m1 = 2.0*(SEQ*(4.0*D*D + 2.0*D*RAW_FF) + 2*55.0*D);
    double m2 = 2.0*(SEQ*(4.0*DP*DP + 2.0*DP*FF2) + 2*55.0*DP);
    double m3 = (SEQ*(DP*2.0*DP) + SEQ*2.0*DP*FF3 + 2*55.0*DP)
              + (SEQ*(DP*2.0*DP) + 1.0*2.0*DP*FF3 + 2*10.0*DP);
    printf("            time/pass    GFLOP/s     MACs\n");
    printf("v1 naive    %7.2f us   %7.2f     %.0f\n", t1*1e6, 2*m1/t1/1e9, m1);
    printf("v2 layout   %7.2f us   %7.2f     %.0f\n", t2*1e6, 2*m2/t2/1e9, m2);
    printf("v3 algebra  %7.2f us   %7.2f     %.0f\n", t3*1e6, 2*m3/t3/1e9, m3);
    printf("\nspeedups: v1->v2 %.2fx   v2->v3 %.2fx   v1->v3 %.2fx\n",
           t1/t2, t2/t3, t1/t3);
    return 0;
}
