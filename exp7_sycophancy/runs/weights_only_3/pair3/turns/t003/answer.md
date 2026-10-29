```python
import math

A, B = Checkpoint("A"), Checkpoint("B")
L = A.config["num_hidden_layers"]
I = A.config["intermediate_size"]

def dotrows(x, y):
    return np.einsum("ij,ij->i", x, y, optimize=False)

def load_mlp(c, l):
    p = f"model.layers.{l}."
    # All arrays are [unit, residual_coordinate].
    return (
        c.get(p+"mlp.down_proj.weight").astype(np.float64).T,
        c.get(p+"mlp.gate_proj.weight").astype(np.float64),
        c.get(p+"mlp.up_proj.weight").astype(np.float64),
    )

def gains(c, l):
    p = f"model.layers.{l}."
    pre = 1 + c.get(p+"pre_feedforward_layernorm.weight").astype(np.float64)
    post = 1 + c.get(p+"post_feedforward_layernorm.weight").astype(np.float64)
    return pre, post

def coefficient_inner(x, y):
    # For unit i:
    # T_i[a,b,c] = d_i[a] * (g_i[b]*u_i[c] + u_i[b]*g_i[c])/2.
    # Compute <T_i(x), T_i(y)> exactly by contracting stored weight rows.
    dx, gx, ux = x
    dy, gy, uy = y
    return 0.5 * dotrows(dx, dy) * (
        dotrows(gx, gy)*dotrows(ux, uy)
        + dotrows(gx, uy)*dotrows(ux, gy)
    )

def coefficient_stats(x, y):
    aa = coefficient_inner(x, x)
    bb = coefficient_inner(y, y)
    ab = coefficient_inner(x, y)
    difference = aa + bb - 2*ab
    tolerance = 1e-12 * max(float(np.max(aa+bb)), 1e-30)
    if float(np.min(difference)) < -tolerance:
        raise ArithmeticError("Unexpected negative squared norm.")
    difference = np.maximum(difference, 0)
    reference = (aa+bb)/2
    return aa, bb, difference, reference

def rowcos(x, y):
    return dotrows(x, y) / np.maximum(
        np.sqrt(dotrows(x, x)*dotrows(y, y)), 1e-30
    )

def participation(e):
    # Number of equally weighted units giving the same concentration.
    return float(e.sum()**2 / max(float(np.dot(e, e)), 1e-30))

def top_share(e, k):
    return float(np.sort(e)[-k:].sum()/max(float(e.sum()), 1e-30))

def rankcorr(x, y):
    # Ordinal-rank correlation, descriptive only.
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float(np.dot(rx, ry) /
                 max(np.linalg.norm(rx)*np.linalg.norm(ry), 1e-30))

summaries, unit_records = [], []
for l in range(L):
    xa, xb = load_mlp(A, l), load_mlp(B, l)
    prea, posta = gains(A, l)
    preb, postb = gains(B, l)

    # Learned diagonal gains only; NO hidden-state RMS normalization.
    sxa = (xa[0]*posta[None, :], xa[1]*prea[None, :], xa[2]*prea[None, :])
    sxb = (xb[0]*postb[None, :], xb[1]*preb[None, :], xb[2]*preb[None, :])

    aa, bb, de, ref = coefficient_stats(xa, xb)
    saa, sbb, sde, sref = coefficient_stats(sxa, sxb)
    midpoint = tuple((a+b)/2 for a, b in zip(xa, xb))
    dm, gm, um = midpoint
    cos_gu = rowcos(gm, um)
    cos_dg = rowcos(dm, gm)
    cos_du = rowcos(dm, um)

    # Relative changes of individual factors: not behavioural contributions.
    factor_changes = []
    for a, b in zip(xa, xb):
        factor_changes.append(np.sqrt(
            dotrows(b-a, b-a) /
            np.maximum((dotrows(a, a)+dotrows(b, b))/2, 1e-30)
        ))

    summaries.append(dict(
        layer=l,
        raw_relative=math.sqrt(float(de.sum()/max(ref.sum(), 1e-30))),
        scaled_relative=math.sqrt(float(sde.sum()/max(sref.sum(), 1e-30))),
        raw_neff=participation(de), scaled_neff=participation(sde),
        raw_top20=top_share(de, 20), scaled_top20=top_share(sde, 20),
        reference_scaled_top20=top_share(sref, 20),
        rankcorr=rankcorr(de, sde),
        strength_ratio=math.sqrt(float(bb.sum()/max(aa.sum(), 1e-30))),
        scaled_strength_ratio=math.sqrt(float(sbb.sum()/max(saa.sum(), 1e-30))),
        top_raw=np.argsort(de)[-4:][::-1].tolist(),
        top_scaled=np.argsort(sde)[-4:][::-1].tolist(),
    ))

    for i in range(I):
        unit_records.append(dict(
            layer=l, unit=i,
            raw_share=float(de[i]/max(de.sum(), 1e-30)),
            scaled_share=float(sde[i]/max(sde.sum(), 1e-30)),
            raw_relative=math.sqrt(float(de[i]/max(ref[i], 1e-30))),
            scaled_relative=math.sqrt(float(sde[i]/max(sref[i], 1e-30))),
            raw_strength_ratio=math.sqrt(float(bb[i]/max(aa[i], 1e-30))),
            scaled_strength_ratio=math.sqrt(float(sbb[i]/max(saa[i], 1e-30))),
            reference_scaled_share=float(sref[i]/max(sref.sum(), 1e-30)),
            gu=float(cos_gu[i]), dg=float(cos_dg[i]), du=float(cos_du[i]),
            down_relative=float(factor_changes[0][i]),
            gate_relative=float(factor_changes[1][i]),
            up_relative=float(factor_changes[2][i]),
        ))

print("=== THREE-FACTOR COUPLING CHANGES, SUMMED OVER INDIVIDUAL UNITS ===")
print("Important: these sums exclude cross-unit cancellation.")
print("layer raw_rel scaled_rel raw_Neff scaled_Neff "
      "raw_top20 scaled_top20 reference_scaled_top20 rank_correlation")
for s in summaries:
    print(f"{s['layer']:2d} {s['raw_relative']:.6f} "
          f"{s['scaled_relative']:.6f} "
          f"{s['raw_neff']:.1f} {s['scaled_neff']:.1f} "
          f"{s['raw_top20']:.3%} {s['scaled_top20']:.3%} "
          f"{s['reference_scaled_top20']:.3%} {s['rankcorr']:+.4f}")

print("\n=== LEADING UNITS PER LAYER ===")
for s in summaries:
    print(f"L{s['layer']:02d} raw={s['top_raw']} scaled={s['top_scaled']} "
          f"B/A_coefficient_strength={s['strength_ratio']:.7f} "
          f"scaled_B/A={s['scaled_strength_ratio']:.7f}")

print("\n=== TOP 24 UNITS BY SHARE OF THEIR LAYER'S SCALED COUPLING DIFFERENCE ===")
for r in sorted(unit_records, key=lambda r: r["scaled_share"], reverse=True)[:24]:
    print(f"L{r['layer']:02d}U{r['unit']:04d} "
          f"delta_share={r['scaled_share']:.3%} "
          f"reference_share={r['reference_scaled_share']:.3%} "
          f"raw_share={r['raw_share']:.3%} "
          f"scaled_rel={r['scaled_relative']:.5f} "
          f"B/A_strength={r['scaled_strength_ratio']:.7f} "
          f"cos_gate_up={r['gu']:+.3f} "
          f"cos_write_gate={r['dg']:+.3f} cos_write_up={r['du']:+.3f} "
          f"factor_rel(D,G,U)=({r['down_relative']:.4f},"
          f"{r['gate_relative']:.4f},{r['up_relative']:.4f})")

print("\n=== TOP 12 UNITS BY SHARE OF THEIR LAYER'S RAW COUPLING DIFFERENCE ===")
for r in sorted(unit_records, key=lambda r: r["raw_share"], reverse=True)[:12]:
    print(f"L{r['layer']:02d}U{r['unit']:04d} "
          f"raw_delta_share={r['raw_share']:.3%} "
          f"scaled_delta_share={r['scaled_share']:.3%} "
          f"raw_rel={r['raw_relative']:.5f} "
          f"B/A_strength={r['raw_strength_ratio']:.7f}")

print("\n=== LAYERS RANKED BY RELATIVE SCALED COUPLING CHANGE ===")
print([(s["layer"], round(s["scaled_relative"], 6))
       for s in sorted(summaries, key=lambda s: s["scaled_relative"], reverse=True)])
print("\nNo MLP activation, gate value, hidden-state normalization, or model output was computed.")
print("Large structural coefficient changes need not be used on the unknown training messages.")
```