#!/usr/bin/env python3
"""v16_search.py — the five-port search, as architecture.

Port 1  candidates from a CONSTRAINED KNEE-POINT GENERATOR (mirrors the
        template list compiled into v16_kernels.cu, plus split factors
        and the vendor tactic).
Port 2  objective in FLOOR-NORMALIZED units (efficiency = floor/measured);
        paired A/B in-context measurement (the binary interleaves both
        tables in one process); termination certificate when the best
        config is within CERT of its floor.
Port 3  ANALYTIC PRIOR MEAN + LEARNED RESIDUAL: predicted = roofline(cfg)
        * penalties; ridge regression of log-residuals on a physical
        feature library; the largest stable coefficient is reported as
        "the missing constraint, named".
Port 4  MULTI-FIDELITY ACQUISITION: fidelity-0 = the analytic model
        (free, prunes); fidelity-1 = paired measurement of the LCB-best
        unmeasured candidate vs the incumbent.  Lower-confidence-bound
        with uncertainty from feature-space novelty.
Port 5  SEMANTIC TIER (hook): the exact/eps rewrites (folding, pruning,
        flash, epilogues) are already IN the substrate; the hook below
        marks where an LLM proposer would extend the rulebook.  This
        script searches only what Ports 1-4 can represent — that
        boundary is the experiment's thesis.

Usage:  python3 v16_search.py [--budget 12] [--binary ./v16]
Run on ada next to the compiled binary.
"""
import argparse, itertools, json, math, re, subprocess, sys
import numpy as np

# ------------------------- machine + workload constants ----------------
SMS, PEAK_TF, L2_BW, DRAM_BW = 80, 34.1e12, 2.5e12, 912e9
SMEM_PER_SM, SMEM_LIMIT = 100*1024, 48*1024
SITES = [  # (name, od, id, T, epilogue)
    ("qkv", 2304, 768, 128), ("wo", 768, 768, 128),
    ("w1", 3072, 768, 128), ("w2", 768, 3072, 128)]
REF = (0.095916, 0.033000); GATE = 5e-3
CFGS = [(bm,bn,bk,st) for bm in (32,64) for bn in (64,128)
        for bk in (16,32) for st in (2,3)
        if st*(bm*(bk+4)+bk*bn)*4 <= SMEM_LIMIT]     # must mirror the .cu

# ------------------------- Port 1: generator ---------------------------
def candidates(site):
    name, od, idim, T = SITES[site]
    out = [(-1, 1)]                                   # vendor tactic
    for ci,(bm,bn,bk,st) in enumerate(CFGS):
        if od % bn or T % bm: continue
        for S in (1,2,4,8):
            if idim % S or (idim//S) % bk: continue
            blocks = (od//bn)*(T//bm)*S
            if blocks < 8: continue                   # knee: starvation floor
            out.append((ci, S))
    return out

# ------------------------- Ports 2+3: floors, features, prior ----------
def floor_ms(site):
    _, od, idim, T = SITES[site]
    f = 2.0*od*idim*T / PEAK_TF                      # compute term
    m = 4.0*od*idim / DRAM_BW                        # weights once (L2 elides x12)
    return max(f, m) * 12 * 1e3                      # 12 layers, ms

def features(site, impl, S):
    _, od, idim, T = SITES[site]
    if impl < 0:                                     # vendor: measured-only
        return None
    bm,bn,bk,st = CFGS[impl]
    warps  = (bm//16)*(bn//32)
    smem   = st*(bm*(bk+4)+bk*bn)*4
    bpsm   = max(1, min(SMEM_PER_SM//smem, 16//max(1,warps//2)))
    blocks = (od//bn)*(T//bm)*S
    waves  = blocks / (bpsm*SMS)
    frag_w = waves/math.ceil(waves) if waves > 0 else 1
    hmma_bar = (bk//8)*2                             # per warp per barrier
    return dict(warps=warps, smem_kb=smem/1024, bpsm=bpsm, blocks=blocks,
                waves=waves, lastwave=1-frag_w, hmma_bar=hmma_bar,
                split=S, stages=st, occ_warps=bpsm*warps,
                traffic=(od*idim + bm and 0) or 0)   # placeholder slot

FEATS = ["warps","smem_kb","bpsm","blocks","waves","lastwave",
         "hmma_bar","split","stages","occ_warps"]

def prior_ms(site, impl, S):
    """Analytic mean (Port 3): roofline * wave + split penalties."""
    if impl < 0: return None
    f = features(site, impl, S)
    base = floor_ms(site) / 0.65                     # tiles reach ~65% here
    wave_pen  = math.ceil(f["waves"]) / max(f["waves"], 1e-9) \
                if f["waves"] > 0.5 else 1/max(f["waves"],0.1)
    lat_pen   = 1.0 if f["occ_warps"] >= 8 else 8/f["occ_warps"]
    split_pen = 1.0 + 0.12*(S-1)                     # reduce round-trip
    return base * min(wave_pen, 3) * min(lat_pen, 3) * split_pen

# ------------------------- measurement (paired, in-context) ------------
def measure(binary, tabA, tabB):
    args = [str(v) for pair in tabA for v in pair] + \
           [str(v) for pair in tabB for v in pair]
    r = subprocess.run([binary]+args, capture_output=True, text=True,
                       timeout=600)
    if r.returncode: sys.exit(f"binary failed:\n{r.stderr}")
    rows = re.findall(r"^[AB]: ([\d. ]+)  logits ([-\d.]+) ([-\d.]+)",
                      r.stdout, re.M)
    out = []
    for ms, l0, l1 in rows:
        v = [float(x) for x in ms.split()]
        if abs(float(l0)-REF[0]) > GATE or abs(float(l1)-REF[1]) > GATE:
            sys.exit(f"GATE FAILED: {l0} {l1} — config produces wrong math")
        out.append(v)
    return out[0], out[1]

# ------------------------- Ports 3+4: residual model + acquisition -----
def search(binary, budget):
    inc  = [(-1,1)]*4                                # incumbent: all-vendor
    (ims, _) = measure(binary, inc, inc)
    best = {s: (inc[s], ims[s]) for s in range(4)}
    print("incumbent (all cuBLASLt): " +
          " ".join(f"{SITES[s][0]}={ims[s]:.3f}" for s in range(4)))
    X, y, meta = [], [], []                          # residual dataset
    for s in range(4):
        cands = [c for c in candidates(s) if c != (-1,1)]
        measured = {}
        for it in range(budget):
            # Port 4: LCB = prior * learned-residual − novelty bonus
            def lcb(c):
                p = prior_ms(s, *c)
                r = 1.0
                if len(y) >= 4:                      # apply learned residual
                    f = np.array([features(s,*c)[k] for k in FEATS])
                    r = math.exp(float(f @ W))
                nov = min((np.linalg.norm(
                    np.array([features(s,*c)[k] for k in FEATS]) -
                    np.array([features(s,*m)[k] for k in FEATS]))
                    for m in measured), default=10)
                return p*r - 0.02*nov
            W = np.zeros(len(FEATS))
            if len(y) >= 4:                          # ridge on log-residual
                A = np.array(X); b = np.array(y)
                A = (A - A.mean(0)) / (A.std(0) + 1e-9)
                W = np.linalg.solve(A.T@A + 1.0*np.eye(len(FEATS)), A.T@b)
            todo = [c for c in cands if c not in measured]
            if not todo: break
            c = min(todo, key=lcb)
            (a_ms, b_ms) = measure(binary,
                [best[t][0] for t in range(4)],
                [best[t][0] if t != s else c for t in range(4)])
            t = b_ms[s]; measured[c] = t
            inc_now = a_ms[s]                        # paired, same window
            f = features(s, *c)
            X.append([f[k] for k in FEATS])
            y.append(math.log(t / prior_ms(s, *c)))
            meta.append((SITES[s][0], c, t))
            tag = ""
            if t < inc_now and t < best[s][1]:       # must beat its OWN pair
                best[s] = (c, t); tag = "  <-- new best (paired)"
            eff = floor_ms(s)/t
            print(f"  {SITES[s][0]:4s} cfg={c} {t:.3f} ms "
                  f"(eff {eff:4.0%}, prior {prior_ms(s,*c):.3f}){tag}")
            if eff > 0.90:                           # Port 2 certificate
                print(f"  {SITES[s][0]}: floor certificate — stopping site")
                break
    # ---- final: best table vs incumbent, paired ----
    tab = [best[s][0] for s in range(4)]
    (ims, fms) = measure(binary, inc, tab)
    print("\n=== VERDICT (paired, same window) ===")
    for s in range(4):
        print(f"{SITES[s][0]:4s}: vendor {ims[s]:.3f}  found {fms[s]:.3f}"
              f"  cfg {tab[s]}  floor {floor_ms(s):.3f}")
    print(f"GEMM total: vendor {sum(ims):.3f}  found {sum(fms):.3f} ms")
    # ---- Port 3 report: name the missing constraint ----
    if len(y) >= 6:
        A = np.array(X); A = (A-A.mean(0))/(A.std(0)+1e-9)
        W = np.linalg.solve(A.T@A + 1.0*np.eye(len(FEATS)),
                            A.T@np.array(y))
        top = sorted(zip(FEATS, W), key=lambda kv: -abs(kv[1]))[:3]
        print("\nresidual analysis — the missing constraints, named:")
        for k, w in top:
            print(f"  {k:10s} coef {w:+.3f}  "
                  f"({'model UNDERprices' if w>0 else 'model OVERprices'})")
    json.dump({SITES[s][0]: {"impl": tab[s][0], "split": tab[s][1],
                             "ms": fms[s]} for s in range(4)},
              open("tactic_table.json","w"), indent=1)
    print("\ntactic_table.json written")
    # Port 5 hook:
    print("\n[Port 5 hook] semantic tier not searched: folding/pruning/"
          "flash/epilogues are baked in; extending the rulebook is a "
          "proposer's job (human or LLM), not a sweep's.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./v16")
    ap.add_argument("--budget", type=int, default=12)
    a = ap.parse_args()
    search(a.binary, a.budget)
