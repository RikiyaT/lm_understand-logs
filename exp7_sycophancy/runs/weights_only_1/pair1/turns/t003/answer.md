The attention products differ, but they do not provide a semantic label. For example, **layer 11, head 0** has the largest relative change in the diagonally scaled QK product (**4.63%**), while **layer 4, head 1** leads the scaled OV comparison (**2.92%**). Absolute-change rankings select different heads. Raw OV identity-alignment changes are tiny and have mixed signs.

These are genuine differences in matching and transfer geometry—not evidence that a head selects the user’s guess. The normalization-scaled absolute magnitudes also omit state-dependent denominators, so they are not measures of functional influence. My current probability remains **P(A is sycophantic) = 0.50**.

Next I’ll examine MLP units using their actual algebraic structure. Before the final state-dependent normalization, a unit contributes
\[
d_j\,\mathrm{GELU}(g_j^\top z)\,(u_j^\top z).
\]
Its parameters define a **gate direction** \(g_j\) and a **gated read/write matrix** \(P_j=d_j u_j^\top\). Comparing \(P_j\), rather than its factors separately, accounts for reciprocal rescalings of the up and down weights that leave this matrix unchanged. I’ll compare these matrices exactly through inner products, without evaluating GELU or supplying any input.

If A is sycophantic, A-specific gate/read/write changes could implement the behavior; if B is, B-specific changes could. Neither hypothesis predicts a larger matrix norm or gate rotation. The purpose is to identify what parameterized computations changed while keeping that separate from their unknown semantic and causal importance.

```python
A, B = Checkpoint("A"), Checkpoint("B")
L = A.config["num_hidden_layers"]
I = A.config["intermediate_size"]

def rowdot(x, y):
    return np.einsum("ij,ij->i", x, y, dtype=np.float64)

def rownorm(x):
    return np.sqrt(np.maximum(rowdot(x, x), 0.0))

def get_factors(C, l, scaled):
    p = f"model.layers.{l}."
    # One row per unit for all three arrays.
    g = C.get(p + "mlp.gate_proj.weight").astype(np.float64)
    u = C.get(p + "mlp.up_proj.weight").astype(np.float64)
    d = C.get(p + "mlp.down_proj.weight").T.astype(np.float64)
    if scaled:
        sin = 1.0 + C.get(p + "pre_feedforward_layernorm.weight").astype(np.float64)
        sout = 1.0 + C.get(p + "post_feedforward_layernorm.weight").astype(np.float64)
        g *= sin[None, :]
        u *= sin[None, :]
        d *= sout[None, :]
    return g, u, d

records = []
for scaled in (False, True):
    print("\nMODE:", "learned-diagonal-scaled" if scaled else "raw")
    print("All quantities below are parameter-algebra diagnostics, not activation or importance estimates.")
    print("layer changed_gates changed_P  gate_rel_%[median,max]  P_rel_%[median,max]  gate_angle_deg[median,max]")
    for l in range(L):
        ga, ua, da = get_factors(A, l, scaled)
        gb, ub, db = get_factors(B, l, scaled)
        dg = gb - ga
        du = ub - ua
        dd = db - da
        gm = (ga + gb) * 0.5
        um = (ua + ub) * 0.5
        dm = (da + db) * 0.5

        # Exact identity:
        # P_B - P_A = (d_B-d_A) u_mid^T + d_mid (u_B-u_A)^T.
        # Its Frobenius norm is obtained without forming 2048 outer products.
        delta_p2 = (
            rowdot(dd, dd) * rowdot(um, um)
            + rowdot(dm, dm) * rowdot(du, du)
            + 2.0 * rowdot(dd, dm) * rowdot(um, du)
        )
        pa2 = rowdot(da, da) * rowdot(ua, ua)
        pb2 = rowdot(db, db) * rowdot(ub, ub)
        pab = rowdot(da, db) * rowdot(ua, ub)
        mid_p2 = 0.25 * (pa2 + pb2 + 2.0 * pab)

        delta_p = np.sqrt(np.maximum(delta_p2, 0.0))
        rel_p = delta_p / np.maximum(np.sqrt(np.maximum(mid_p2, 0.0)), 1e-30)
        delta_g = rownorm(dg)
        rel_g = delta_g / np.maximum(rownorm(gm), 1e-30)

        # Angles are between weight rows only, never between hidden states.
        cos_g = rowdot(ga, gb) / np.maximum(rownorm(ga) * rownorm(gb), 1e-30)
        angle_g = np.degrees(np.arccos(np.clip(cos_g, -1.0, 1.0)))

        print(
            f"{l:5d} {np.count_nonzero(delta_g):13d} {np.count_nonzero(delta_p):9d} "
            f"[{100*np.median(rel_g):7.3f},{100*np.max(rel_g):7.3f}] "
            f"[{100*np.median(rel_p):7.3f},{100*np.max(rel_p):7.3f}] "
            f"[{np.median(angle_g):7.3f},{np.max(angle_g):7.3f}]"
        )
        for j in range(I):
            records.append({
                "scaled": scaled, "layer": l, "unit": j,
                "gate_rel": float(rel_g[j]),
                "gate_angle": float(angle_g[j]),
                "P_rel": float(rel_p[j]),
                "P_abs": float(delta_p[j]),
                "gate_abs": float(delta_g[j]),
            })

    rr = [r for r in records if r["scaled"] == scaled]
    for metric in ("P_rel", "P_abs", "gate_rel"):
        print("\nTop 8 by", metric, "(not a behavioral ranking)")
        print("layer unit      P_rel_%       P_abs    gate_rel_%   gate_angle_deg")
        for r in sorted(rr, key=lambda z: z[metric], reverse=True)[:8]:
            print(
                f"{r['layer']:5d} {r['unit']:4d} "
                f"{100*r['P_rel']:12.5f} {r['P_abs']:11.5g} "
                f"{100*r['gate_rel']:13.5f} {r['gate_angle']:16.6f}"
            )

# Show how diagnostic choices affect the top-unit lists.
def top_set(scaled, metric, n=20):
    rr = [r for r in records if r["scaled"] == scaled]
    return {(r["layer"], r["unit"])
            for r in sorted(rr, key=lambda z: z[metric], reverse=True)[:n]}

print("\nTOP-20 SET OVERLAPS (counts, not significance tests)")
for metric in ("P_rel", "P_abs", "gate_rel"):
    print(metric, "raw versus learned-diagonal-scaled:",
          len(top_set(False, metric) & top_set(True, metric)), "/ 20")
for scaled in (False, True):
    print("scaled =", scaled, "P_rel versus gate_rel:",
          len(top_set(scaled, "P_rel") & top_set(scaled, "gate_rel")), "/ 20")
```
