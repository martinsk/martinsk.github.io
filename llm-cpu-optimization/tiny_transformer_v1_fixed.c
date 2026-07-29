/* tiny_transformer.c — naive forward pass, written for codegen inspection.
 *
 * Shape:  SEQ=10 tokens, D=100 model dim, FF=400 mlp dim, LAYERS=2,
 *         single attention head, final projection to NOUT=10 logits
 *         (take logits of last token as "the output").
 *
 * Everything is float, static storage, plain loops. No BLAS, no SIMD
 * intrinsics — the point is to see what -O3 -march=native does on its own.
 *
 * Suggested Godbolt flags:  -O3 -march=znver4 -ffast-math -fno-math-errno
 * (or -march=skylake-avx512 / -mcpu=apple-m1 on clang). Try with and
 * without -ffast-math: softmax/layernorm reductions won't vectorize
 * without it because FP reassociation is forbidden.
 */

#include <math.h>
#include <string.h>

#define SEQ    10
#define D      100
#define FF     400
#define NOUT   10
#define LAYERS 2

/* ---- weights (would be loaded; zero-init here, marked volatile-ish via
 * a init function so the compiler can't constant-fold the whole net) ---- */
typedef struct {
    float wq[D][D], wk[D][D], wv[D][D], wo[D][D];
    float w1[FF][D], b1[FF];          /* mlp up   */
    float w2[D][FF], b2[D];           /* mlp down */
    float ln1_g[D], ln1_b[D];
    float ln2_g[D], ln2_b[D];
} Layer;

static Layer layers[LAYERS];
static float w_out[NOUT][D];
static float ln_f_g[D], ln_f_b[D];

/* ---- activations ---- */
static float x[SEQ][D];               /* residual stream */
static float xn[SEQ][D];              /* post-layernorm scratch */
static float q[SEQ][D], k[SEQ][D], v[SEQ][D];
static float att[SEQ][SEQ];
static float ao[SEQ][D];              /* attention output */
static float h[SEQ][FF];              /* mlp hidden */

/* y = W x  (out_dim rows, in_dim cols) — the workhorse */
static void matvec(int out_dim, int in_dim,
                   const float *restrict W,   /* [out_dim][in_dim] */
                   const float *restrict xin,
                   float *restrict y)
{
    for (int i = 0; i < out_dim; i++) {
        float acc = 0.0f;
        for (int j = 0; j < in_dim; j++)
            acc += W[i * in_dim + j] * xin[j];
        y[i] = acc;
    }
}

static void layernorm(const float *restrict xin, const float *restrict g,
                      const float *restrict b, float *restrict y)
{
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < D; i++) mean += xin[i];
    mean /= D;
    for (int i = 0; i < D; i++) {
        float d = xin[i] - mean;
        var += d * d;
    }
    var /= D;
    float rstd = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < D; i++)
        y[i] = (xin[i] - mean) * rstd * g[i] + b[i];
}

static void softmax_row(float *restrict a, int n)
{
    float mx = a[0];
    for (int i = 1; i < n; i++) if (a[i] > mx) mx = a[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { a[i] = expf(a[i] - mx); sum += a[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) a[i] *= inv;
}

static void forward_layer(const Layer *restrict L)
{
    const float scale = 1.0f / sqrtf((float)D);

    /* ---- attention block ---- */
    for (int t = 0; t < SEQ; t++) {
        layernorm(x[t], L->ln1_g, L->ln1_b, xn[t]);
        matvec(D, D, &L->wq[0][0], xn[t], q[t]);
        matvec(D, D, &L->wk[0][0], xn[t], k[t]);
        matvec(D, D, &L->wv[0][0], xn[t], v[t]);
    }

    /* scores: att[t][s] = q[t]·k[s] * scale   (causal mask) */
    for (int t = 0; t < SEQ; t++) {
        for (int s = 0; s <= t; s++) {
            float acc = 0.0f;
            for (int j = 0; j < D; j++)
                acc += q[t][j] * k[s][j];
            att[t][s] = acc * scale;
        }
        softmax_row(att[t], t + 1);
    }

    /* weighted sum of values + output proj + residual */
    for (int t = 0; t < SEQ; t++) {
        float tmp[D];
        memset(tmp, 0, sizeof tmp);
        for (int s = 0; s <= t; s++) {
            float a = att[t][s];
            for (int j = 0; j < D; j++)
                tmp[j] += a * v[s][j];
        }
        matvec(D, D, &L->wo[0][0], tmp, ao[t]);
        for (int j = 0; j < D; j++)
            x[t][j] += ao[t][j];
    }

    /* ---- mlp block ---- */
    for (int t = 0; t < SEQ; t++) {
        layernorm(x[t], L->ln2_g, L->ln2_b, xn[t]);
        matvec(FF, D, &L->w1[0][0], xn[t], h[t]);
        for (int i = 0; i < FF; i++) {
            float z = h[t][i] + L->b1[i];
            h[t][i] = z > 0.0f ? z : 0.0f;          /* relu */
        }
        float tmp[D];
        matvec(D, FF, &L->w2[0][0], h[t], tmp);
        for (int j = 0; j < D; j++)
            x[t][j] += tmp[j] + L->b2[j];
    }
}

/* logits for last token */
void transformer_forward(const float *restrict in /*[SEQ][D]*/,
                         float *restrict logits   /*[NOUT]*/)
{
    memcpy(x, in, sizeof x);
    for (int l = 0; l < LAYERS; l++)
        forward_layer(&layers[l]);

    float xf[D];
    layernorm(x[SEQ - 1], ln_f_g, ln_f_b, xf);
    matvec(NOUT, D, &w_out[0][0], xf, logits);
}

/* keep weights opaque so nothing constant-folds */
void load_weights(const float *blob)
{
    memcpy(layers, blob, sizeof layers);
    blob += sizeof(layers) / sizeof(float);
    memcpy(w_out, blob, sizeof w_out);
    blob += sizeof(w_out) / sizeof(float);
    memcpy(ln_f_g, blob, sizeof ln_f_g);
    blob += sizeof(ln_f_g) / sizeof(float);
    memcpy(ln_f_b, blob, sizeof ln_f_b);
}
