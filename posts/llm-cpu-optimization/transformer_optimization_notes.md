# From "how baked-in is the OS process?" to a robust-LP-optimized transformer
### Session notes — CPU microarchitecture, algebraic optimization, and the limits of cost models

---

## The arc

What started as a question about whether the OS process model limits how much compute
you can push through a CPU turned into a complete optimization campaign on a tiny
transformer (SEQ=10, D=100→128, FF=4D, 2 layers, single head), taken through five
generations of implementation, measured exhaustively, modeled with an integer linear
program, and finally projected to multicore. Final result: **2.44× end-to-end on one
core (213.4µs → 87.3µs at D=128), with every variant bit-agreeing to ~1e-6, and the
last configuration chosen by a solver, not a human.**

The deeper output isn't the 2.44× — it's the catalog of failure modes and the method
that survived them.

---

## The versions

| version | change | mechanism |
|---|---|---|
| v1 | naive C | row-form dot products, runtime layernorm, libm expf |
| v2 | layout | column-major/axpy matvec, dims padded to 112, polynomial exp |
| v3 | algebra | QK merge, OV merge, layernorm folding, last-token pruning, deferred softmax norm, one-pass stats |
| v4 | fusion | token-fused GEMM projections (weights streamed once per pass, not once per token) |
| LP  | solver-selected | fuse **only** the uv projection; unrolling on — a config no human proposed |

### The algebra (v3) in one place — the part that generalizes
- **QK merge:** q·k = xnᵗ(Wqᵀ·Wk)xn — only the product is observable; the K projection
  disappears, keys become the normalized activations themselves.
- **OV merge:** Wo·Σaₛ(Wv·xnₛ) = Σaₛ(WoWv)xnₛ — no nonlinearity between V and O, so they
  compose. 4 D×D projections/token → 1 fused one at half the size.
- **Layernorm folding:** γ, β feed only linear maps → absorb offline. Runtime LN is
  just (x−mean)·rstd.
- **Softmax shift-invariance** deletes the query-side bias cross-terms outright;
  **softmax-sums-to-one** turns the value bias into a plain bias — two "free" deletions
  that exist only in the math.
- **Last-token pruning:** only logits[last] are read → final layer's MLP/attention for
  tokens 0..8 is dead code *no compiler can see*. Decays as 1/L with depth — but is
  exactly incremental decoding: we accidentally derived the KV cache.
- Caveat filed: QK/OV merges invert into a pessimization for multi-head with H ≥ 3
  (rank D/H products vs dense D²). Single-head only, or keep factored.

---

## Headline numbers (all same-window, controlled)

**D=128, quiet machine state:**

| variant | MACs | time | GF/s | vs v1 |
|---|---|---|---|---|
| v1 | 3.96M | 213.4 µs | 37.1 | 1.00× |
| v2 | 3.96M | 171.5 µs | 46.2 | 1.24× |
| v3 | 2.11M | 89.2 µs | 47.4 | 2.39× |
| v4 all-fuse | 2.11M | 115.0 µs | 36.8 | 1.86× |
| LP-robust | 2.11M | **87.3 µs** | 48.4 | **2.44×** |

**Regime dependence** (v-later ÷ v3): fusion is **0.92×** at D=128 (L2-resident),
**1.52×** at D=512, **1.88×** at D=2048 (DRAM-bound), and **0.33×** under frontend
contention with the unrolled build. The algebra's ~1.9× held in *every* regime.

**Decomposition law:** speedup = (work ratio) × (rate ratio). v3's work cut is
compiler-, ISA-, and memory-regime-invariant; every rate gain was contingent.

**Why the algebra survives the memory wall:** in matvec, FLOPs and bytes are locked
in proportion (2 FLOPs per 4 weight bytes), so every FLOP deleted was a byte deleted.
Measured byte-ratio prediction at D=2048: 1.88×; measured speedup: 1.90×.

---

## Discoveries, chronologically

1. **The process abstraction costs ~nothing.** The CPU has no concept of a process —
   privilege levels, CR3, a timer. `isolcpus`+`nohz_full`+huge pages+pinning ≈
   bare metal. The real ceilings are ports, dependency chains, cache BW, DRAM BW.

2. **LLVM deleted the output (v1).** `load_weights` never wrote `ln_f_g/b`; IPA proved
   them zero; the entire final projection constant-folded to `memset(logits, 0, 40)`.
   Whole-program analysis reaches further than intuition says.

3. **LLVM deleted half the program (v2).** `float acc[DP]` called with `out_dim=FF`
   → stack overflow → UB → the loop exit and everything downstream removed as
   unreachable, leaving an orphaned infinite loop in the asm. **At -O3, UB doesn't
   corrupt data; it deletes code, silently.** ASan/UBSan belongs in the Godbolt loop.

4. **Row-form vs column-form matvec** is the whole codegen game: dot products give a
   serial FMA chain + a 7-instruction horizontal-reduction ladder per output;
   axpy form gives persistent register accumulators and zero reductions. The rule:
   reductions across vector lanes are poison; reductions across loop iterations into
   vector accumulators are free.

5. **gcc vs clang, identical source, 40% apart** — and gcc made v1 ≈ v2 by
   auto-vectorizing the reductions itself. The "v1 is latency-bound" diagnosis was a
   property of clang, not the source. Never conclude "this code is slow" from one
   compiler's asm.

6. **Inliner roulette:** `static inline` was honored in one TU and ignored in another;
   without inlining, no constant propagation, no strip specialization → 6× slower.
   All the specialization was load-bearing on a heuristic we don't control.
   `always_inline` when correctness-of-performance depends on it.

7. **The ragged-strip cliff:** strips of 112 at od=112/224/448 → 36 GF/s; od=128
   (112+16 ragged) → 5.3 GF/s. Mixed strip widths defeat register promotion — the
   accumulator array stays in memory: 14 loads + 7 FMAs + 7 stores per iteration
   instead of 7 register FMAs. **7× from a divisibility property.** Strip width must
   divide every output dimension.

8. **The L3 does implicit temporal fusion.** v3's per-token weight re-streams at
   D=2048 ran at ~47 GB/s effective — L3 speed, not DRAM — because back-to-back
   streams of a matrix smaller than L3 hit cache. Explicit fusion only collects the
   increment between "L3 catches most of it" and "DRAM sees it once." Expected 10×,
   measured 1.88×, and the gap is the cache hierarchy already doing the optimization.

9. **v4 reached the actual roofline:** 1.34 GB of weights per pass at ~14 GB/s
   sustained = the machine's single-core streaming rate. Past that point "more ops
   through the CPU" is the wrong question — you're pumping bytes. Remaining levers:
   quantization (÷2–4 bytes), batch (reuse), cores (aggregate BW).

10. **GPU crossover:** at this model size the GPU *loses* — 30–50 kernel launches
    ≈ 200µs vs 87µs on CPU, and PCIe round-trip ≈ the whole computation. GPUs win at
    ≳1ms of work per invocation or when the working set wants HBM bandwidth. Tiny
    latency-critical models belong on a pinned CPU core (or FPGA).

---

## The linear program: two iterations

**LP-1** (compute rate + L3 BW + DRAM BW rows, calibrated by measurement):
instantly re-derived every dominant algebraic choice (merge, prune — exact linear
coefficients, they're theorems). Picked all-fuse everywhere. Scorecard: **right at
D=2048 (1.45×), wrong at 128/512 (0.34×/0.65×)** — its pick ranked **15th of 16**
measured configs at D=128. Regret 3.11×.

**Root cause of the miss** (a debugging saga in itself): the fused kernel ran
61–65 GF/s standalone and 14 GF/s in context — same shapes, same minute. The asm:
**8 FMA instructions standalone vs 1006 in the composed binary** — unrolling blew the
hot code past the µop cache, a resource with no row in the model. And the penalty was
*time-varying*: identical binaries swung 98µs ↔ 382µs across windows while tiny
calibration kernels never moved — an unseen SMT sibling / co-tenant contending for
the shared frontend. (Also: `uptime 7 min` mid-session — the VM had been silently
re-placed. Some of "the machine changed" was literally a different machine.)

**LP-2** additions: (a) a **code-footprint constraint row** — and the footprint turned
out to be *exactly additive* in the decision variables (base 526 FMAs; marginals
uv −24, w1 +248, w2 +256), so it's legitimately linear; (b) **two-scenario robust
objective** (quiet / contended frontend) instead of a point estimate; (c) steady-state
memory accounting (LP-1 charged cold-pass DRAM to hot loops — a bug).

**LP-2 result:** the robust objective selected **fuse-uv-only** — optimal because uv's
footprint marginal is *negative* (its gemm replaced a larger per-token loop), so it
banks the 64 GF/s fused rate without crossing the contention cliff. Nobody had tried
this config. It was the **measured optimum of all 16 in both machine states**
(contended: 122.2µs, predicted 121.6 — 0.5% error; quiet: 87.3µs). Rank correlation
model-vs-measured: **ρ = 0.906**. Regret: 3.11× → **1.00×**.

**The formalism verdict:** LP/roofline is exact for the algebra (FLOPs/bytes are
theorems) and for regime identification (duals = which constraint binds); cut/partition
(hypergraph — weights-as-nets, VLSI-style) is the right structure for fusion and
device placement (CPU/GPU split is a literal poly-time s-t mincut); **measurement owns
everything contextual, compositional, or non-stationary** — which this machine
demonstrated four separate times is not a small residue. Trust models for 10×
decisions, the stopwatch for 2× decisions.

---

## Multicore projection (design, not measurement — box had 1 vCPU)

- **Splitting one pass** across cores has a size threshold: ~14 barriers × 1–5µs
  against an 87µs pass caps 8 cores below 2×. Worth it only when phase work ≫ 50×
  barrier cost, i.e. D ≳ 1024.
- **Replication** (one stream per pinned core, weights shared read-only) scales
  linearly until a *shared* resource saturates. At D=2048: unfused = 2.15 GB/pass →
  ~230 passes/s ceiling at ~500 GB/s, saturated by ~34 cores. Fused: 0.34 GB →
  ~1,470/s (6.4×). Fused + batch-8: ~11,000 inf/s (48×).
- **Currency change:** with many cores, bytes-per-pass (shared) replaces
  seconds-per-pass (private) as the objective. Fusion's 1.5× single-core value becomes
  6.4× socket value. Shadow prices migrate with core count.
- **Ordering: batch first, then replicate.** Batching attacks the shared constraint;
  cores only the private one; they compose multiplicatively.
- **SMT becomes a design variable:** our contended-state data (small-footprint build
  loses 1.3× under frontend sharing, big build 3.5×) implies two compact streams per
  core (~1.54×) beat one unrolled stream with an idle sibling. The robust LP's
  "adversarial scenario" becomes a deliberate operating point.

---

## The surprises, ranked

1. **UB deletes programs, not data** — twice, silently, at -O3.
2. **A divisibility property was worth 7×** (ragged strips → memory accumulators).
3. **The measured optimum config was one no human proposed**, found via a *negative*
   footprint coefficient the solver could see and we couldn't.
4. **Identical binaries varied 4.3×** across time on shared infra while micro-
   benchmarks held steady — the missing constraint was someone else's process.
5. **gcc quietly performed our hand optimization** (v1≈v2), invalidating a diagnosis
   built on clang's output.
6. **The L3 was already doing fusion** — the 10× theoretical fusion win was mostly
   pre-collected by LRU.
7. **Standalone kernel benchmarks lied by 4.4×** about in-context performance
   (composition through the frontend). Calibration ≠ context.
8. **The math never broke.** Through deleted programs, inliner roulette, VM swaps, and
   a solver confidently choosing a 3× pessimization, relerr sat at ~1e-7 in every
   single run. Exact reasoning above, empiricism below, hard interface between.

## Operating principles distilled

- Algebra first (portable, provable), layout second, schedule last — each layer
  invisible to the tools below it.
- Model proposes, stopwatch disposes; every accepted move changes the binding
  constraint, so iterate.
- Only compare numbers from the same window / same binary pairing. Anything else is
  an anecdote.
- On shared infrastructure, optimize the regret distribution, not the point estimate.
- When prediction and measurement disagree, the disagreement is the next bug — and
  it's usually where the 2× is hiding.

## Artifacts
`tiny_transformer.c` (v1) · `tiny_transformer_v2.c` · `tiny_transformer_v3.c` ·
`bench.c` (3-way harness + numeric cross-check) · `scaling.c` (D-sweep) ·
`opt.c` (macro-parametrized fusion, LP-driven) · `calib.c` (machine constants) ·
`lp.py` (LP-1) · `lp2.py` (footprint + robust LP-2)
