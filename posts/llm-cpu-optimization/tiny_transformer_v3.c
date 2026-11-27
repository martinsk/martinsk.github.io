/* tiny_transformer_v3.c — the algebra pass.  Same function as v1/v2 up to
 * float rounding, roughly half the FLOPs, and structurally simpler asm.
 *
 * FIX FROM v2:  matvec_col's scratch was float acc[DP]=112 but was called
 * with out_dim=FF -> stack overflow -> UB -> LLVM deleted the loop exit and
 * everything after it (the orphaned .LBB0_27 infinite loop).  v3's matvec
 * strip-mines the output in 112-wide tiles, so the scratch bound is exact
 * by construction.
 *
 * MATH REWRITES (all done offline in prepare_weights; runtime never sees
 * the original parameterization):
 *
 *  R1. QK merge:      q.k = xn_t^T (Wq^T Wk) xn_s.  Only the product is
 *      observable, so precompute it.  The K projection disappears; the
 *      "keys" are the normalized activations themselves.
 *  R2. OV merge:      Wo (sum_s a_s Wv xn_s) = sum_s a_s (Wo Wv) xn_s.
 *      No nonlinearity between V and O, so they compose.  The output
 *      projection disappears.  4 DxD projections/token -> 1 fused one.
 *  R3. LayerNorm fold: xn = g*xhat + b feeds only linear maps, so
 *      diag(g) and b are absorbed into every consuming matrix offline.
 *      Runtime layernorm is just (x - mean) * rstd.
 *  R4. Softmax shift-invariance: after folding b into the QK form,
 *      score(t,s) = xhat_t^T A2 xhat_s + r.xhat_s + c.xhat_t + const.
 *      The last two terms are constant in s -> invisible to softmax ->
 *      DELETED.  r.xhat_s folds into the query as u' = A2^T xhat_t + r.
 *  R5. Softmax sums to one: the attention-value bias bv = (WoWv)b
 *      satisfies sum_s a_s bv = bv, so it becomes a plain bias on the
 *      fused value projection.  No correction term anywhere.
 *  R6. Dead code above the compiler: only logits of the LAST token are
 *      read, so in the final layer the attention rows, residual updates,
 *      and (critically) the MLP for tokens 0..8 are never computed.
 *      Keys/values (xhat, v') are still computed for all tokens.
 *  R7. Deferred softmax normalization: accumulate the value sum with
 *      unnormalized exp weights, scale once by 1/sum at the end.
 *  R8. One-pass layernorm stats: Var = E[x^2] - E[x]^2, both sums in a
 *      single traversal over DP (pads are zero, so no tail loop).
 *
 * PAD INVARIANTS (replaces v2's "activations pads are 0" rule):
 *   - x pads are always 0 (inputs zero-padded; all projection outputs
 *     and biases have zero pad lanes by memset in prepare_weights).
 *   - xhat pads are NOT zero (-mean*rstd), which is harmless: every
 *     matrix consuming xhat has zero ROWS for j >= D, and score dots
 *     u'.xhat are safe because u' pads are zero.
 *
 * Every projection in the network is now the same primitive:
 * matvec_bias, axpy form, strip-mined to 112-wide output tiles
 * (7 zmm accumulators per strip -- the LBB0_2 shape from v2).
 *
 * Flags as before:  -O3 -ffast-math -march=znver5
 */

#include <math.h>
#include <string.h>

#define SEQ    10
#define D      100
#define DP     112            /* 7 x 16 */
#define RAW_FF 400
#define FF     448            /* 4 x 112: four clean output strips */
#define NOUT   10
#define NOUTP  16
#define LAYERS 2

/* ---------------- prepared (runtime) model ---------------- */
typedef struct {
    /* fused query/value projection: [ A2^T | WoWv*diag(g1) ],
     * axpy layout [in][out], bias [ r | (WoWv)b1 ]. */
    float wuv[DP][2 * DP];
    float buv[2 * DP];
    float w1[DP][FF],  b1[FF];      /* W1*diag(g2),  W1*b2ln + b1 */
    float w2[FF][DP],  b2[DP];
} PLayer;

typedef struct {
    PLayer layer[LAYERS];
    float w_out[DP][NOUTP], b_out[NOUTP];   /* ln_f folded in */
} Model;

static Model M;

/* ---------------- activations ---------------- */
static float x  [SEQ][DP];
static float xh [SEQ][DP];        /* xhat: shared by scores AND uv proj */
static float uv [SEQ][2 * DP];    /* [ u' | v'' ] per token             */
static float hb [FF];

/* y[o] = bias[o] + sum_j xin[j] * W[j][o], output strip-mined by 112.
 * All out_dims here are compile-time constants at the call sites, so
 * after inlining each strip is a fixed 7-accumulator (or 1 for NOUTP)
 * axpy loop with no tails and no horizontal reductions. */
static inline void matvec_bias(int in_dim, int out_dim,
                               const float *restrict W,     /* [in][out] */
                               const float *restrict bias,
                               const float *restrict xin,
                               float *restrict y)
{
    for (int o = 0; o < out_dim; o += 112) {
        int wdt = out_dim - o;
        if (wdt > 112) wdt = 112;
        float acc[112];
        for (int i = 0; i < wdt; i++) acc[i] = bias[o + i];
        for (int j = 0; j < in_dim; j++) {
            float a = xin[j];
            const float *restrict wr = W + (long)j * out_dim + o;
            for (int i = 0; i < wdt; i++)
                acc[i] += a * wr[i];
        }
        for (int i = 0; i < wdt; i++) y[o + i] = acc[i];
    }
}

/* single-pass stats (R8) + normalize.  No gamma/beta (R3). */
static inline void norm_hat(const float *restrict xin, float *restrict out)
{
    float s = 0.0f, ss = 0.0f;
    for (int i = 0; i < DP; i++) {        /* pads are 0: contribute 0/0 */
        s  += xin[i];
        ss += xin[i] * xin[i];
    }
    float mean = s * (1.0f / D);
    float var  = ss * (1.0f / D) - mean * mean;
    float rstd = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < DP; i++)
        out[i] = (xin[i] - mean) * rstd;  /* pads: -mean*rstd, see notes */
}

static inline float fast_exp(float xx)
{
    float t  = xx * 1.4426950408889634f;
    float fl = floorf(t);
    float f  = t - fl;
    float p  = 1.0f + f * (0.69314718f + f * (0.24022651f +
               f * (0.05550411f + f * (0.00898934f + f * 0.00187757f))));
    union { float f; int i; } u;
    u.i = ((int)fl + 127) << 23;
    return p * u.f;
}

static void forward_layer(const PLayer *restrict L, int last)
{
    /* keys (= xh) and fused query/value proj: ALL tokens (later tokens
     * attend to them), even in the pruned last layer.  Computing u' for
     * tokens 0..8 in the last layer is wasted but tiny; splitting the
     * fused projection to avoid it would cost more than it saves. */
    for (int t = 0; t < SEQ; t++) {
        norm_hat(x[t], xh[t]);
        matvec_bias(DP, 2 * DP, &L->wuv[0][0], L->buv, xh[t], uv[t]);
    }

    int t0 = last ? SEQ - 1 : 0;          /* R6 */

    /* attention rows + residual */
    for (int t = t0; t < SEQ; t++) {
        const float *restrict u = uv[t];  /* query, scale & r included */
        float w[SEQ];
        float mx = -3.4e38f;
        for (int s = 0; s <= t; s++) {    /* true dots: SEQ^2/2 of them */
            float acc = 0.0f;
            for (int j = 0; j < DP; j++)
                acc += u[j] * xh[s][j];
            w[s] = acc;
            mx = acc > mx ? acc : mx;
        }
        float sum = 0.0f;
        for (int s = 0; s <= t; s++) {
            w[s] = fast_exp(w[s] - mx);
            sum += w[s];
        }
        float tmp[DP];
        for (int j = 0; j < DP; j++) tmp[j] = 0.0f;
        for (int s = 0; s <= t; s++) {    /* unnormalized weights (R7) */
            float a = w[s];
            const float *restrict vv = uv[s] + DP;
            for (int j = 0; j < DP; j++)
                tmp[j] += a * vv[j];
        }
        float inv = 1.0f / sum;
        for (int j = 0; j < DP; j++)
            x[t][j] += tmp[j] * inv;      /* bv rides inside v'' (R5) */
    }

    /* MLP */
    for (int t = t0; t < SEQ; t++) {
        float xh2[DP], tmp[DP];
        norm_hat(x[t], xh2);
        matvec_bias(DP, FF, &L->w1[0][0], L->b1, xh2, hb);
        for (int i = 0; i < FF; i++)
            hb[i] = hb[i] > 0.0f ? hb[i] : 0.0f;
        matvec_bias(FF, DP, &L->w2[0][0], L->b2, hb, tmp);
        for (int j = 0; j < DP; j++)
            x[t][j] += tmp[j];
    }
}

void transformer_forward(const float *restrict in,   /* [SEQ][D] packed */
                         float *restrict logits)     /* [NOUT]          */
{
    for (int t = 0; t < SEQ; t++) {
        memcpy(x[t], in + (long)t * D, D * sizeof(float));
        for (int j = D; j < DP; j++) x[t][j] = 0.0f;
    }
    for (int l = 0; l < LAYERS; l++)
        forward_layer(&M.layer[l], l == LAYERS - 1);

    float xf[DP], lg[NOUTP];
    norm_hat(x[SEQ - 1], xf);
    matvec_bias(DP, NOUTP, &M.w_out[0][0], M.b_out, xf, lg);
    memcpy(logits, lg, NOUT * sizeof(float));
}

/* =============== offline: raw weights -> prepared model =============== */
/* v1-style raw parameterization, row-major [out][in], unpadded. */
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

static float A[D][D], Wov[D][D];   /* scratch */

void prepare_weights(const RawModel *restrict R)
{
    memset(&M, 0, sizeof M);                 /* zero pads everywhere */
    const float scale = 1.0f / sqrtf((float)D);

    for (int l = 0; l < LAYERS; l++) {
        const RawLayer *rl = &R->layer[l];
        PLayer *pl = &M.layer[l];
        const float *g1 = rl->ln1_g, *b1n = rl->ln1_b;
        const float *g2 = rl->ln2_g, *b2n = rl->ln2_b;

        /* R1: A = Wq^T Wk  (q.k = xn_t^T A xn_s) */
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++) {
                float acc = 0.0f;
                for (int m = 0; m < D; m++)
                    acc += rl->wq[m][i] * rl->wk[m][j];
                A[i][j] = acc;
            }
        /* R3+R4: A2 = scale * diag(g1) A diag(g1), query bias
         * r[o] = scale * sum_i b1n[i] A[i][o] g1[o]  (the s-side cross
         * term; the t-side term and the constant died to softmax
         * shift-invariance). axpy layout: wuv[j][o] = A2[j][o]. */
        for (int j = 0; j < D; j++)
            for (int o = 0; o < D; o++)
                pl->wuv[j][o] = scale * g1[j] * A[j][o] * g1[o];
        for (int o = 0; o < D; o++) {
            float acc = 0.0f;
            for (int i = 0; i < D; i++)
                acc += b1n[i] * A[i][o];
            pl->buv[o] = scale * acc * g1[o];
        }

        /* R2: Wov = Wo Wv */
        for (int o = 0; o < D; o++)
            for (int i = 0; i < D; i++) {
                float acc = 0.0f;
                for (int m = 0; m < D; m++)
                    acc += rl->wo[o][m] * rl->wv[m][i];
                Wov[o][i] = acc;
            }
        /* R3+R5: value proj = Wov diag(g1), bias bv = Wov b1n
         * (legal as a per-token bias because attention weights sum
         * to 1). axpy layout in the second half of wuv. */
        for (int j = 0; j < D; j++)
            for (int o = 0; o < D; o++)
                pl->wuv[j][DP + o] = Wov[o][j] * g1[j];
        for (int o = 0; o < D; o++) {
            float acc = 0.0f;
            for (int i = 0; i < D; i++)
                acc += Wov[o][i] * b1n[i];
            pl->buv[DP + o] = acc;
        }

        /* R3 on the MLP: fold ln2 gamma into W1 columns, beta into b1 */
        for (int j = 0; j < D; j++)
            for (int f = 0; f < RAW_FF; f++)
                pl->w1[j][f] = rl->w1[f][j] * g2[j];
        for (int f = 0; f < RAW_FF; f++) {
            float acc = rl->b1[f];
            for (int j = 0; j < D; j++)
                acc += rl->w1[f][j] * b2n[j];
            pl->b1[f] = acc;
        }
        for (int f = 0; f < RAW_FF; f++)     /* transpose only */
            for (int o = 0; o < D; o++)
                pl->w2[f][o] = rl->w2[o][f];
        for (int o = 0; o < D; o++)
            pl->b2[o] = rl->b2[o];
    }

    /* R3 on the head: fold ln_f */
    for (int j = 0; j < D; j++)
        for (int o = 0; o < NOUT; o++)
            M.w_out[j][o] = R->w_out[o][j] * R->ln_f_g[j];
    for (int o = 0; o < NOUT; o++) {
        float acc = 0.0f;
        for (int j = 0; j < D; j++)
            acc += R->w_out[o][j] * R->ln_f_b[j];
        M.b_out[o] = acc;
    }
}
