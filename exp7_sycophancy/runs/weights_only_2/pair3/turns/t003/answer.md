The coupled-weight analysis found real differences, but **the rankings depend strongly on the descriptor**: raw versus gain-weighted top-ten lists overlap in only three OV heads and no QK heads. Layer 15 head 1 is a comparatively consistent OV candidate. MLP units 1824 in layer 6 and 1455 in layer 7 stand out in raw path differences, but that could reflect large pre-existing weights rather than concentrated learning. My probability that **A is sycophantic remains 0.50**.

I’ll now test whether the updates contain **coherent directions**, rather than merely many small changes. I’ll estimate their leading singular values and compare them with entrywise sign-scrambled versions that preserve every update magnitude. This is a descriptive control, not a statistical model of training.

A strong result would identify organised parameter changes worth tracing across components. Under either A-sycophantic or B-sycophantic, the singular-value results would be identical: reversing the difference does not change its singular values. Thus this can strengthen localisation candidates, but cannot assign semantic labels.

```python
A, B = Checkpoint("A"), Checkpoint("B")
rng = np.random.default_rng(20261025)

families = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]

def top_spectrum(x, seed, width=16, iterations=6):
    # Randomized eigenspace estimation of a WEIGHT DIFFERENCE.
    # This does not evaluate any model layer on an input.
    transposed = x.shape[0] > x.shape[1]
    z = x.T if transposed else x
    local_rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(local_rng.standard_normal((z.shape[0], width)))
    energy = float(np.sum(z * z))
    if energy == 0:
        return np.zeros(width), 0.0, None, None
    for _ in range(iterations):
        q, _ = np.linalg.qr(z @ (z.T @ q))
    small = q.T @ z
    eigenvalues, eigenvectors = np.linalg.eigh(small @ small.T)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    left = q @ eigenvectors[:, order[0]]
    s1 = np.sqrt(eigenvalues[0])
    right = z.T @ left / max(s1, 1e-300)
    residual = np.linalg.norm(z @ (z.T @ left) - eigenvalues[0] * left)
    residual /= max(eigenvalues[0], 1e-300)
    if transposed:
        left, right = right, left
    return eigenvalues / energy, float(residual), left, right

rows = []
for l in range(A.config["num_hidden_layers"]):
    for fi, family in enumerate(families):
        name = f"model.layers.{l}.{family}"
        a = A.get(name).astype(np.float64)
        b = B.get(name).astype(np.float64)
        delta = b - a
        midpoint = (a + b) / 2
        energy = float(np.sum(delta * delta))
        fractions, residual, left, right = top_spectrum(
            delta, seed=1000 + 10*l + fi
        )

        # Preserve entry magnitudes, sparsity and row/column energies;
        # disrupt their sign coherence. Not a training-distribution null.
        signs = rng.integers(0, 2, size=delta.shape, dtype=np.int8) * 2 - 1
        shuffled = delta * signs
        null_fractions, null_residual, _, _ = top_spectrum(
            shuffled, seed=2000 + 10*l + fi
        )

        radial_cos = float(np.sum(delta * midpoint)) / max(
            np.sqrt(energy * float(np.sum(midpoint * midpoint))), 1e-300
        )
        row_energy = np.sum(delta * delta, axis=1)
        col_energy = np.sum(delta * delta, axis=0)
        row_top = np.argsort(row_energy)[::-1][:3]
        col_top = np.argsort(col_energy)[::-1][:3]

        rows.append({
            "layer": l, "family": family,
            "top1": float(fractions[0]),
            "top5": float(fractions[:5].sum()),
            "top16": float(fractions.sum()),
            "null1": float(null_fractions[0]),
            "enrichment": float(fractions[0] / max(null_fractions[0], 1e-300)),
            "residual": residual, "null_residual": null_residual,
            "radial_cos": radial_cos,
            "top_rows": row_top.tolist(), "top_cols": col_top.tolist(),
            "max_row_share": float(row_energy.max() / max(energy, 1e-300)),
            "max_col_share": float(col_energy.max() / max(energy, 1e-300)),
        })

print("TOP 30 UPDATE MATRICES BY TOP-SINGULAR-ENERGY ENRICHMENT")
print("Fractions are approximate lower estimates; residual diagnoses convergence.")
for r in sorted(rows, key=lambda r: -r["enrichment"])[:30]:
    print(
        f"L{r['layer']:02d} {r['family']:29s} "
        f"top1={100*r['top1']:7.3f}% top5={100*r['top5']:7.3f}% "
        f"null1={100*r['null1']:6.3f}% ratio={r['enrichment']:7.3f} "
        f"resid={r['residual']:.3g}/{r['null_residual']:.3g} "
        f"radial_cos={r['radial_cos']:+.5f}"
    )

print("\nTOP 15 MATRICES BY TOP-SINGULAR-ENERGY FRACTION")
for r in sorted(rows, key=lambda r: -r["top1"])[:15]:
    print(
        f"L{r['layer']:02d} {r['family']:29s} "
        f"top1={100*r['top1']:.4f}% top16={100*r['top16']:.4f}% "
        f"max_row={100*r['max_row_share']:.3f}% rows={r['top_rows']} "
        f"max_col={100*r['max_col_share']:.3f}% cols={r['top_cols']}"
    )

print("\nPER-LAYER SPECTRAL SUMMARY")
print("For each family: top1_energy_percent / sign-control_enrichment")
labels = ["Q", "K", "V", "O", "GATE", "UP", "DOWN"]
print("layer " + " ".join(f"{s:>15s}" for s in labels))
for l in range(A.config["num_hidden_layers"]):
    parts = []
    for family in families:
        r = next(r for r in rows if r["layer"] == l and r["family"] == family)
        parts.append(f"{100*r['top1']:6.2f}/{r['enrichment']:6.2f}")
    print(f"{l:5d} " + " ".join(f"{p:>15s}" for p in parts))

print("\nFAMILY MEDIANS")
for family in families:
    subset = [r for r in rows if r["family"] == family]
    print(
        f"{family:29s} "
        f"top1={100*np.median([r['top1'] for r in subset]):.4f}% "
        f"enrichment={np.median([r['enrichment'] for r in subset]):.4f} "
        f"radial_cos={np.median([r['radial_cos'] for r in subset]):+.6f}"
    )

print("\nCONVERGENCE CHECK")
print("Maximum leading-eigenpair relative residual:",
      max(r["residual"] for r in rows))
print("Median leading-eigenpair relative residual:",
      np.median([r["residual"] for r in rows]))
print("Maximum sign-control relative residual:",
      max(r["null_residual"] for r in rows))
print("These are algebraic controls only; no model inputs or nonlinearities were evaluated.")
```