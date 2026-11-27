#!/usr/bin/env python3
"""Per-kernel roofline ILP for the tiny transformer.

Decision variables:
  a            merged QK/OV algebra (vs separate q/k/v/o)
  p            last-layer token pruning
  z[site,c]    schedule choice per projection site: per-token or fused

Per selected kernel k:  t_k >= flops/rate(s),  t_k >= L3bytes/bw_l3,
                        t_k >= DRAMbytes/bw_dram.     minimize sum t_k.

Memory model (traffic ABOVE L2; sub-L2 traffic is folded into the
calibrated kernel rates):
  per-token: L3 bytes = size * (tokens if size>L2 else 1)
             DRAM     = size * (tokens if size>L3 else 1)
  fused    : L3 = DRAM = size * 1     (L2-blocked tile reuse)
"""
import math, sys
import pulp

# --- measured machine constants ---
M = {}
for line in open("machine.txt"):
    k, v = line.strip().split("=")
    M[k] = float(v)
RATE = {"pt": M["rate_pt"] * 1e9, "fu": M["rate_fu"] * 1e9}
BW_L3, BW_DRAM = M["bw_l3"] * 1e9, M["bw_dram"] * 1e9
L2, L3 = 1.0 * 2**20, 33.0 * 2**20
SEQ, LAYERS = 10, 2

def kernel_cost(id_, od, tokens, sched):
    size = id_ * od * 4.0
    fl = 2.0 * id_ * od * tokens
    if sched == "pt":
        l3 = size * (tokens if size > L2 else 1)
        dr = size * (tokens if size > L3 else 1)
    else:
        l3 = size
        dr = size
    return {"comp": fl / RATE[sched], "l3": l3 / BW_L3, "dram": dr / BW_DRAM}

def solve(D):
    FF = 4 * D
    prob = pulp.LpProblem(f"tf_{D}", pulp.LpMinimize)
    a = pulp.LpVariable("merged", cat="Binary")
    p = pulp.LpVariable("pruned", cat="Binary")

    sites = []   # (name, [(choice_label, costs, guards)])
    for l in range(LAYERS):
        last = (l == LAYERS - 1)
        # ---- attention projection site ----
        ch = []
        for s in ("pt", "fu"):
            # merged: one (D -> 2D) kernel, all tokens (keys+values needed)
            ch.append((f"merged/{s}", kernel_cost(D, 2 * D, SEQ, s),
                       {"a": 1}))
            # unmerged: k,v all tokens; q,o all tokens (unpruned variant)
            cu = {r: sum(kernel_cost(D, D, SEQ, s)[r] for _ in range(4))
                  for r in ("comp", "l3", "dram")}
            ch.append((f"unmerged/{s}", cu, {"a": 0}))
            if last:
                # unmerged+pruned: k,v all tokens; q,o one token
                cup = {r: 2 * kernel_cost(D, D, SEQ, s)[r]
                          + 2 * kernel_cost(D, D, 1, "pt")[r]
                       for r in ("comp", "l3", "dram")}
                ch.append((f"unmerged-pruned/{s}", cup, {"a": 0, "p": 1}))
        sites.append((f"L{l}.attnproj", ch))
        # ---- MLP sites ----
        for nm, i_, o_ in (("w1", D, FF), ("w2", FF, D)):
            ch = []
            for s in ("pt", "fu"):
                ch.append((f"full/{s}", kernel_cost(i_, o_, SEQ, s), {}))
            if last:
                ch.append(("pruned/pt", kernel_cost(i_, o_, 1, "pt"),
                           {"p": 1}))
            sites.append((f"L{l}.{nm}", ch))

    T, Z = [], {}
    for name, choices in sites:
        t = pulp.LpVariable(f"t_{name}", lowBound=0)
        T.append((name, t, choices))
        zs = []
        for lbl, cost, guard in choices:
            z = pulp.LpVariable(f"z_{name}_{lbl}", cat="Binary")
            Z[(name, lbl)] = z
            zs.append((z, cost, guard))
            if guard.get("a") == 1: prob += z <= a
            if guard.get("a") == 0: prob += z <= 1 - a
            if guard.get("p") == 1: prob += z <= p
        prob += pulp.lpSum(z for z, _, _ in zs) == 1
        for r in ("comp", "l3", "dram"):
            prob += t >= pulp.lpSum(z * c[r] for z, c, _ in zs)

    prob += pulp.lpSum(t for _, t, _ in T)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    print(f"\n=== D={D}  (weights: attn {8*D*D*4/1e6:.1f}MB unmerged / "
          f"{2*D*D*4/1e6:.1f}MB merged, mlp {8*D*D*4/1e6:.1f}MB per layer)")
    print(f"  algebra: merged={int(a.value())}  pruned={int(p.value())}")
    tot = 0
    for name, t, choices in T:
        for lbl, cost, _ in choices:
            if Z[(name, lbl)].value() > 0.5:
                bind = max(cost, key=cost.get)
                print(f"  {name:14s} -> {lbl:18s} t={t.value()*1e6:9.1f} us"
                      f"  (binding: {bind}, "
                      f"c/l3/dr = {cost['comp']*1e6:.0f}/"
                      f"{cost['l3']*1e6:.0f}/{cost['dram']*1e6:.0f} us)")
        tot += t.value()
    print(f"  predicted total (projections only): {tot*1e6:.1f} us")
    # emit compiler flags for codegen
    cfg = {}
    for name, t, choices in T:
        for lbl, cost, _ in choices:
            if Z[(name, lbl)].value() > 0.5:
                cfg[name] = lbl
    fuse_uv = 1 if "fu" in cfg["L0.attnproj"] else 0
    fuse_w1 = 1 if "fu" in cfg["L0.w1"] else 0
    fuse_w2 = 1 if "fu" in cfg["L0.w2"] else 0
    print(f"  FLAGS: -DFUSE_UV={fuse_uv} -DFUSE_W1={fuse_w1} "
          f"-DFUSE_W2={fuse_w2}  (merged+pruned assumed in codegen)")
    return tot

for D in (128, 512, 2048):
    solve(D)
