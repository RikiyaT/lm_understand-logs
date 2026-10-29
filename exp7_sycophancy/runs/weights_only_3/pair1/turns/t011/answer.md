The eighth run showed that late-layer coordination exceeds what individual difference magnitudes explain. That strengthens the structural finding, but does not identify its meaning. **My probability that A is sycophantic remains 0.50.**

I’ll test a different hypothesis now: **changes in connected components compensate for each other**, so inspecting their changed directions separately overstates the change in their composed operations.

For a writing matrix \(W\) and a subsequent reading matrix \(R\), the following identity is exact:
\[
R_BW_B-R_AW_A
=\Delta R\,\overline W+\overline R\,\Delta W.
\]
I’ll measure whether these two terms cancel or reinforce for:
- attention output → same-layer MLP gate/up projections;
- MLP output → next-layer query/key/value projections.

These are products of weight matrices, **not executions of those pathways**: residual-state normalization, attention routing, and MLP nonlinearities remain unevaluated.

The compensation hypothesis predicts negative alignment between the two terms and a smaller combined difference. Reinforcement predicts the opposite. **Both predictions are independent of whether A or B is sycophantic**, so this can refine the structural interpretation without supplying a behavioral label.

```python
A, B = Checkpoint("A"), Checkpoint("B")
L = A.config["num_hidden_layers"]

def get(C, l, suffix):
    return C.get(f"model.layers.{l}.{suffix}.weight").astype(np.float64)

def read_pair(l, kind, folded):
    suffix = {
        "gate": "mlp.gate_proj",
        "up": "mlp.up_proj",
        "query": "self_attn.q_proj",
        "key": "self_attn.k_proj",
        "value": "self_attn.v_proj"
    }[kind]

    result = []
    for C in (A, B):
        R = get(C, l, suffix)
        if folded:
            norm_name = (
                "pre_feedforward_layernorm"
                if kind in ("gate", "up")
                else "input_layernorm"
            )
            s = 1.0 + get(C, l, norm_name)
            R = R * s[None, :]
        result.append(R)
    return result

def write_pair(l, kind, folded):
    suffix, norm_name = {
        "attention": ("self_attn.o_proj", "post_attention_layernorm"),
        "mlp": ("mlp.down_proj", "post_feedforward_layernorm")
    }[kind]

    result = []
    for C in (A, B):
        W = get(C, l, suffix)
        if folded:
            s = 1.0 + get(C, l, norm_name)
            W = s[:, None] * W
        result.append(W)
    return result

def reader_grams(pair):
    a, b = pair
    m = (a + b) / 2
    delta = b - a
    return {
        "mid": m.T @ m,
        "delta": delta.T @ delta,
        "cross": delta.T @ m
    }

def writer_grams(pair):
    a, b = pair
    m = (a + b) / 2
    delta = b - a
    return {
        "mid": m @ m.T,
        "delta": delta @ delta.T,
        "cross": m @ delta.T
    }

def inner(a, b):
    return float(np.sum(a * b))

def composition_statistics(r, w):
    # X = delta_R W_mid; Y = R_mid delta_W.
    x2 = max(inner(r["delta"], w["mid"]), 0.0)
    y2 = max(inner(r["mid"], w["delta"]), 0.0)
    xy = inner(r["cross"], w["cross"])
    combined2 = max(x2 + y2 + 2.0 * xy, 0.0)
    separate2 = max(x2 + y2, 1e-300)

    # Average endpoint squared Frobenius norm of R_A W_A and R_B W_B.
    rc = r["mid"] + 0.25 * r["delta"]
    wc = w["mid"] + 0.25 * w["delta"]
    rs = 0.5 * (r["cross"] + r["cross"].T)
    ws = 0.5 * (w["cross"] + w["cross"].T)
    baseline = max(inner(rc, wc) + inner(rs, ws), 1e-300)

    return {
        "term_cosine": xy / max(np.sqrt(x2 * y2), 1e-300),
        "combined_over_separate_energy": combined2 / separate2,
        "composed_relative_difference": float(np.sqrt(combined2 / baseline)),
        "reader_algebraic_share": (x2 + xy) / max(combined2, 1e-300),
        "writer_algebraic_share": (y2 + xy) / max(combined2, 1e-300)
    }

all_rows = []
print("NEW HYPOTHESIS: COMPENSATION BETWEEN CONNECTED WEIGHT CHANGES")
print("Products describe linear coefficient composition only.")
print("Query/key normalization and all state-dependent operations are excluded.")
print("combined/separate energy: <1 cancellation, >1 reinforcement.")
print("Algebraic shares are not causal effect shares.")

for folded in (False, True):
    mode = "folded" if folded else "raw"
    readers = {}
    writers = {}

    for l in range(L):
        for kind in ["gate", "up", "query", "key", "value"]:
            readers[(l, kind)] = reader_grams(read_pair(l, kind, folded))
        for kind in ["attention", "mlp"]:
            writers[(l, kind)] = writer_grams(write_pair(l, kind, folded))

    rows = []
    for l in range(L):
        for kind in ["gate", "up"]:
            stat = composition_statistics(
                readers[(l, kind)], writers[(l, "attention")]
            )
            rows.append({
                "mode": mode, "writer_layer": l, "writer": "attention",
                "reader_layer": l, "reader": kind, **stat
            })

        if l + 1 < L:
            for kind in ["query", "key", "value"]:
                stat = composition_statistics(
                    readers[(l+1, kind)], writers[(l, "mlp")]
                )
                rows.append({
                    "mode": mode, "writer_layer": l, "writer": "mlp",
                    "reader_layer": l+1, "reader": kind, **stat
                })

    all_rows.extend(rows)
    cosines = np.array([r["term_cosine"] for r in rows])
    ratios = np.array([
        r["combined_over_separate_energy"] for r in rows
    ])

    print("\nMODE:", mode, "PATHS:", len(rows))
    print("Term-cosine quantiles:",
          np.quantile(cosines, [0, .1, .25, .5, .75, .9, 1]))
    print("Combined/separate energy quantiles:",
          np.quantile(ratios, [0, .1, .25, .5, .75, .9, 1]))
    print("Paths with negative term cosine:", int(np.sum(cosines < 0)))
    print("Paths with >25% energy cancellation:", int(np.sum(ratios < .75)))
    print("Paths with >25% energy reinforcement:", int(np.sum(ratios > 1.25)))

    print("\nSIX LOWEST COMBINED/SEPARATE RATIOS")
    for r in sorted(rows,
                    key=lambda r: r["combined_over_separate_energy"])[:6]:
        print(json.dumps(r))

    print("\nSIX HIGHEST COMBINED/SEPARATE RATIOS")
    for r in sorted(rows,
                    key=lambda r: r["combined_over_separate_energy"],
                    reverse=True)[:6]:
        print(json.dumps(r))

print("\nLATE PATHWAYS, NORMALIZATION DIAGONALS INCLUDED")
print("writer -> reader           cosine combined/separate composed_rel")
for r in all_rows:
    if r["mode"] == "folded" and r["writer_layer"] >= 14:
        path = (
            f"L{r['writer_layer']}.{r['writer']} -> "
            f"L{r['reader_layer']}.{r['reader']}"
        )
        print(f"{path:39s} {r['term_cosine']:9.5f} "
              f"{r['combined_over_separate_energy']:17.5f} "
              f"{r['composed_relative_difference']:12.6f}")

# Verify the Gram identities against direct products of actual weight matrices.
# This is matrix algebra on weights, not execution on a hidden-state input.
Ra, Rb = read_pair(0, "gate", False)
Wa, Wb = write_pair(0, "attention", False)
X = (Rb - Ra) @ ((Wa + Wb) / 2)
Y = ((Ra + Rb) / 2) @ (Wb - Wa)
direct = inner(X + Y, X + Y) / max(inner(X, X) + inner(Y, Y), 1e-300)
formula = composition_statistics(
    reader_grams([Ra, Rb]), writer_grams([Wa, Wb])
)["combined_over_separate_energy"]
print("\nFormula verification, absolute discrepancy:", abs(direct - formula))

print("\nThese products do not measure the full nonlinear network or task behavior.")
print("A/B reversal leaves every reported compensation statistic unchanged.")
```