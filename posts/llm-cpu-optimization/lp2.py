#!/usr/bin/env python3
"""Extended per-kernel model for D=128, steady-state (weights cache-resident).

Additions vs lp.py:
  1. Code-footprint constraint row: F = base(u) + sum(marginal_s * fuse_s),
     measured per-binary from objdump.  If F > TH, the 'contended' machine
     scenario applies penalty P_OVER to all kernel rates; else P_UNDER.
     (Models the shared, time-varying frontend/uop-cache observed on this VM.)
  2. Scenario robustness: solve for three objectives:
       nominal   = quiet-machine time only
       expected  = 0.5*quiet + 0.5*contended
       minimax   = worst case over scenarios
  3. Steady-state memory: DRAM charged only for working sets > L3 (none at
     D=128), so compute+frontend dominate.  This fixes lp.py's cold-pass bug.
"""
import pulp

D, FF, SEQ = 128, 512, 10
R = {}   # rates GF/s per (sched, unroll)
for u in (0, 1):
    for line in open(f"mach_u{u}.txt"):
        k, v = line.strip().split("=")
        if k in ("rate_pt", "rate_fu"):
            R[(k[5:], u)] = float(v) * 1e9

# measured footprint model (FMA count, additive fit is exact)
FP_BASE = {1: 526, 0: 52}
FP_MARG = {1: {"uv": -24, "w1": 248, "w2": 256},
           0: {"uv": 1,   "w1": 3,   "w2": 3}}
TH, P_OVER, P_UNDER = 600, 3.5, 1.3

# kernel sites: (name, flops, fusible)
KFL = [("L0.uv", 2*D*2*D*SEQ, "uv"), ("L0.w1", 2*D*FF*SEQ, "w1"),
       ("L0.w2", 2*FF*D*SEQ, "w2"), ("L1.uv", 2*D*2*D*SEQ, "uv"),
       ("L1.w1", 2*D*FF*1, None), ("L1.w2", 2*FF*D*1, None)]

def model_time(cfg, u):
    """analytic evaluation for the enumeration table -> (quiet, contended) s"""
    F = FP_BASE[u] + sum(FP_MARG[u][s] for s in ("uv","w1","w2") if cfg[s])
    pen = P_OVER if F > TH else P_UNDER
    tq = tc = 0.0
    for name, fl, site in KFL:
        s = "fu" if (site and cfg[site]) else "pt"
        t = fl / R[(s, u)]
        tq += t; tc += t * pen
    return tq, tc, F

def solve(objective):
    best = None
    for u in (0, 1):
        prob = pulp.LpProblem(f"o_{objective}_u{u}", pulp.LpMinimize)
        z = {s: pulp.LpVariable(f"fuse_{s}", cat="Binary")
             for s in ("uv", "w1", "w2")}
        y = pulp.LpVariable("over", cat="Binary")
        F = FP_BASE[u] + pulp.lpSum(FP_MARG[u][s] * z[s] for s in z)
        prob += F <= TH + 10000 * y                 # footprint row
        Tq, Tc = [], []
        for name, fl, site in KFL:
            if site is None:
                tq = fl / R[("pt", u)]
                Tq.append(tq)
                Tc.append(tq * P_UNDER + (P_OVER - P_UNDER) * tq * y)
            else:
                tq = pulp.LpVariable(f"tq_{name}", lowBound=0)
                prob += tq >= fl / R[("pt", u)] * (1 - z[site]) \
                            + fl / R[("fu", u)] * z[site]
                tc = pulp.LpVariable(f"tc_{name}", lowBound=0)
                # contended = quiet * penalty; linearize z*y with w
                w = pulp.LpVariable(f"w_{name}", cat="Binary")
                prob += w <= z[site]; prob += w <= y
                prob += w >= z[site] + y - 1
                base_pt, base_fu = fl / R[("pt", u)], fl / R[("fu", u)]
                prob += tc >= base_pt * P_UNDER * (1 - z[site]) \
                            + base_fu * P_UNDER * z[site] \
                            + (P_OVER - P_UNDER) * (base_pt * (y - w)
                                                    + base_fu * w)
                Tq.append(tq); Tc.append(tc)
        TQ, TC = pulp.lpSum(Tq), pulp.lpSum(Tc)
        if objective == "nominal":
            prob += TQ
        elif objective == "expected":
            prob += 0.5 * TQ + 0.5 * TC
        else:
            T = pulp.LpVariable("T", lowBound=0)
            prob += T >= TQ; prob += T >= TC; prob += T
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        val = pulp.value(prob.objective)
        cfg = {s: int(z[s].value()) for s in z}
        if best is None or val < best[0]:
            best = (val, cfg, u)
    val, cfg, u = best
    tq, tc, F = model_time(cfg, u)
    name = f"cfg_{cfg['uv']}{cfg['w1']}{cfg['w2']}_u{u}"
    print(f"{objective:9s} -> {name}  pred quiet {tq*1e6:6.1f} us  "
          f"contended {tc*1e6:6.1f} us  footprint {F}")
    return name

print("== model enumeration (all 16) ==")
rows = []
for u in (0, 1):
    for uv in (0, 1):
        for w1 in (0, 1):
            for w2 in (0, 1):
                cfg = {"uv": uv, "w1": w1, "w2": w2}
                tq, tc, F = model_time(cfg, u)
                rows.append((tq, tc, F, f"cfg_{uv}{w1}{w2}_u{u}"))
for tq, tc, F, n in sorted(rows):
    print(f"  {n}  quiet {tq*1e6:6.1f}  contended {tc*1e6:6.1f}  "
          f"exp {0.5*(tq+tc)*1e6:6.1f}  fp {F}")
print("\n== LP solutions ==")
picks = {o: solve(o) for o in ("nominal", "expected", "minimax")}
open("picks.txt","w").write("\n".join(f"{k}={v}" for k,v in picks.items()))
