The fifth run showed that the large layer-4 unit-341 difference is attenuated by the next layer’s attention readers. That establishes a cross-layer filtering effect, not its semantic role. My current probability remains **P(A is sycophantic) = 0.50**.

I’ll test a different hypothesis: **attention value and output weights changed mainly in compensating ways, preserving their combined computation.**

Separate changes to \(V\) and \(O\) can be misleading. For example, a change of coordinates inside the value space can change both matrices while leaving \(OV\) unchanged. The exact identity
\[
\Delta(OV)=\Delta O\,V_{\mathrm{mid}}+O_{\mathrm{mid}}\,\Delta V
\]
lets us measure whether those two changes cancel or reinforce each other. This is not an approximation.

- Strong cancellation would support compensatory changes rather than comparably large changes in residual-space transfer.
- Little cancellation would reject that explanation for most of the measured changes.
- If A is sycophantic **or** B is sycophantic, either outcome is possible. Swapping A and B reverses both terms and leaves the cancellation diagnostic unchanged; this test cannot itself orient the behavioral label.

I’ll test every attention head, both with raw weights and with learned input/output diagonal scales. No inputs, attention coefficients, or nonlinearities will be evaluated.

```python
A, B = Checkpoint("A"), Checkpoint("B")
L = A.config["num_hidden_layers"]
H = A.config["num_attention_heads"]
D = A.config["hidden_size"]
HD = A.config["head_dim"]

def sqnorm(x):
    return float(np.sum(x * x, dtype=np.float64))

def contraction_stats(left, right):
    el = sqnorm(left)
    er = sqnorm(right)
    cross = float(np.sum(left * right, dtype=np.float64))
    total = el + er + 2.0 * cross
    separate = el + er
    return {
        "left_e": el,
        "right_e": er,
        "cross": cross,
        "total_e": max(total, 0.0),
        "energy_ratio": total / separate if separate else float("nan"),
        "cosine": cross / np.sqrt(el * er) if el and er else float("nan"),
        "write_share": el / separate if separate else float("nan")
    }

records = []
for layer in range(L):
    factors = []
    for C in (A, B):
        p = f"model.layers.{layer}."
        o = C.get(p + "self_attn.o_proj.weight").astype(np.float64).reshape(D, H, HD)
        v = C.get(p + "self_attn.v_proj.weight").astype(np.float64)
        si = 1.0 + C.get(p + "input_layernorm.weight").astype(np.float64)
        so = 1.0 + C.get(p + "post_attention_layernorm.weight").astype(np.float64)
        factors.append((o, v, si, so))

    for scaled in (False, True):
        oa_all, va, sia, soa = factors[0]
        ob_all, vb, sib, sob = factors[1]
        if scaled:
            va = va * sia[None, :]
            vb = vb * sib[None, :]
        vm = 0.5 * (va + vb)
        dv = vb - va

        for head in range(H):
            oa = oa_all[:, head, :]
            ob = ob_all[:, head, :]
            if scaled:
                oa = soa[:, None] * oa
                ob = sob[:, None] * ob

            om = 0.5 * (oa + ob)
            do = ob - oa

            # Exact midpoint decomposition of a product difference.
            # Both operands in each product are learned-weight matrices
            # or their algebraic combinations.
            write_change = do @ vm
            value_change = om @ dv
            stats = contraction_stats(write_change, value_change)
            stats.update(layer=layer, head=head, scaled=scaled)
            records.append(stats)

            # Independently verify the identity on head 0 of each layer.
            if head == 0:
                direct = ob @ vb - oa @ va
                residual = np.linalg.norm(direct - write_change - value_change)
                relative_residual = residual / max(np.linalg.norm(direct), 1e-30)
                assert relative_residual < 1e-9, (
                    layer, scaled, relative_residual
                )

print("COMPENSATION TEST: exact delta(OV) decomposition")
print("Energy ratio = ||write_change + value_change||F^2 /")
print("               (||write_change||F^2 + ||value_change||F^2)")
print("  0: complete cancellation; 1: zero cross term; >1: reinforcement.")
print("Term cosine measures alignment between the two matrix changes.")
print("Write share is a share of separate term energies, NOT behavioral importance.")
print("Scaled mode includes learned input/output multipliers but no state normalization.")

for scaled in (False, True):
    rr = [r for r in records if r["scaled"] == scaled]
    ratios = np.array([r["energy_ratio"] for r in rr])
    cosines = np.array([r["cosine"] for r in rr])
    pooled_left = sum(r["left_e"] for r in rr)
    pooled_right = sum(r["right_e"] for r in rr)
    pooled_cross = sum(r["cross"] for r in rr)
    pooled_ratio = (
        pooled_left + pooled_right + 2.0 * pooled_cross
    ) / (pooled_left + pooled_right)

    print("\nMODE:", "diagonal-scaled" if scaled else "raw")
    print("Energy-ratio quantiles [min,25%,median,75%,max]:",
          np.quantile(ratios, [0, .25, .5, .75, 1]))
    print("Term-cosine quantiles [min,25%,median,75%,max]:",
          np.quantile(cosines, [0, .25, .5, .75, 1]))
    print("Pooled energy ratio across heads:", round(pooled_ratio, 7))
    print("Head counts: ratio < 0.25:", int(np.sum(ratios < .25)),
          "; ratio < 0.5:", int(np.sum(ratios < .5)),
          "; ratio > 1:", int(np.sum(ratios > 1)))

    for title, reverse in (
        ("Lowest ratios: strongest cancellation", False),
        ("Highest ratios: strongest reinforcement", True)
    ):
        print("\n" + title)
        print("layer head energy_ratio term_cosine separate_write_share_%")
        for r in sorted(rr, key=lambda x: x["energy_ratio"], reverse=reverse)[:8]:
            print(
                f"{r['layer']:5d} {r['head']:4d} "
                f"{r['energy_ratio']:12.7f} {r['cosine']:11.7f} "
                f"{100*r['write_share']:22.5f}"
            )

print("\nLAYER-POOLED RATIOS (stacking the four heads' product differences)")
print("layer   raw_ratio scaled_ratio")
for layer in range(L):
    values = []
    for scaled in (False, True):
        rr = [r for r in records
              if r["layer"] == layer and r["scaled"] == scaled]
        separate = sum(r["left_e"] + r["right_e"] for r in rr)
        total = sum(r["total_e"] for r in rr)
        values.append(total / separate if separate else float("nan"))
    print(f"{layer:5d} {values[0]:11.7f} {values[1]:12.7f}")
```
