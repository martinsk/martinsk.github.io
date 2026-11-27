/* calib.c — measure the machine constants for the LP:
 *   rate_pt : per-token 64-strip matvec kernel, weights in L2 (GF/s)
 *   rate_fu : fused mr=2 x 32-strip gemm kernel, weights in L2 (GF/s)
 *   bw_l3   : streaming read bandwidth, 24MB working set (GB/s)
 *   bw_dram : streaming read bandwidth, 1.2GB working set (GB/s)
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+1e-9*t.tv_nsec;}
static float *alloc(long n){float*p;posix_memalign((void**)&p,64,n*4);return p;}

#define ID 512
#define OD 256
__attribute__((always_inline))
static inline void mv_pt(const float*restrict W,const float*restrict x,
                         float*restrict y)
{
    for(int o=0;o<OD;o+=64){
        float a2[64];
        for(int i=0;i<64;i++)a2[i]=0;
        for(int j=0;j<ID;j++){
            float a=x[j];
            const float*restrict wr=W+(long)j*OD+o;
            for(int i=0;i<64;i++)a2[i]+=a*wr[i];
        }
        for(int i=0;i<64;i++)y[o+i]=a2[i];
    }
}
#define NT 10
__attribute__((always_inline))
static inline void mv_fu(const float*restrict W,const float*restrict X,
                         float*restrict Y)
{
    for(int t=0;t<NT;t++)
        for(int o=0;o<OD;o++)Y[(long)t*OD+o]=0;
    for(int o=0;o<OD;o+=32){
        for(int t=0;t<NT;t+=2){
            float a0[32],a1[32];
            float*restrict y0=Y+(long)t*OD+o;
            float*restrict y1=Y+(long)(t+1)*OD+o;
            for(int i=0;i<32;i++){a0[i]=y0[i];a1[i]=y1[i];}
            const float*restrict x0=X+(long)t*ID;
            const float*restrict x1=X+(long)(t+1)*ID;
            for(int j=0;j<ID;j++){
                float b0=x0[j],b1=x1[j];
                const float*restrict wr=W+(long)j*OD+o;
                for(int i=0;i<32;i++){a0[i]+=b0*wr[i];a1[i]+=b1*wr[i];}
            }
            for(int i=0;i<32;i++){y0[i]=a0[i];y1[i]=a1[i];}
        }
    }
}

int main(void)
{
    float *W=alloc((long)ID*OD), *X=alloc((long)NT*ID), *Y=alloc((long)NT*OD);
    for(long i=0;i<(long)ID*OD;i++)W[i]=0.001f*(i%13);
    for(long i=0;i<(long)NT*ID;i++)X[i]=0.001f*(i%7);
    volatile float s=0;
    double best;

    best=1e9;
    for(int r=0;r<3;r++){
        double t0=now();
        for(int n=0;n<4000;n++){
            X[0]+=s*1e-30f;
            for(int t=0;t<NT;t++) mv_pt(W,X+(long)t*ID,Y+(long)t*OD);
            s=Y[0];
        }
        double t=(now()-t0)/4000; if(t<best)best=t;
    }
    double rate_pt=2.0*ID*OD*NT/best/1e9;

    best=1e9;
    for(int r=0;r<3;r++){
        double t0=now();
        for(int n=0;n<4000;n++){ X[0]+=s*1e-30f; mv_fu(W,X,Y); s=Y[0]; }
        double t=(now()-t0)/4000; if(t<best)best=t;
    }
    double rate_fu=2.0*ID*OD*NT/best/1e9;

    long n_l3=6*1000*1000;        /* 24 MB */
    long n_dr=300*1000*1000;      /* 1.2 GB */
    float *B=alloc(n_dr);
    for(long i=0;i<n_dr;i++)B[i]=0.001f*(i%11);
    best=1e9;
    for(int r=0;r<20;r++){
        double t0=now();
        float a0=0,a1=0,a2=0,a3=0;
        for(long i=0;i<n_l3;i+=4){a0+=B[i];a1+=B[i+1];a2+=B[i+2];a3+=B[i+3];}
        s=a0+a1+a2+a3;
        double t=now()-t0; if(t<best)best=t;
    }
    double bw_l3=4.0*n_l3/best/1e9;
    best=1e9;
    for(int r=0;r<3;r++){
        double t0=now();
        float a0=0,a1=0,a2=0,a3=0;
        for(long i=0;i<n_dr;i+=4){a0+=B[i];a1+=B[i+1];a2+=B[i+2];a3+=B[i+3];}
        s=a0+a1+a2+a3;
        double t=now()-t0; if(t<best)best=t;
    }
    double bw_dram=4.0*n_dr/best/1e9;

    printf("rate_pt=%.1f\nrate_fu=%.1f\nbw_l3=%.1f\nbw_dram=%.1f\n",
           rate_pt,rate_fu,bw_l3,bw_dram);
    return 0;
}
