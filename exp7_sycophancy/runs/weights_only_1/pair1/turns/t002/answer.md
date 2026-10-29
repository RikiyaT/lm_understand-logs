The inventory shows differences in **all 18 layers**: 200 tensors changed and 35 are identical. Relative parameter differences are small and broadly distributed; the final normalization vector is identical. This neither identifies a behavioral circuit nor distinguishes the training objectives. My current probability remains **P(A is sycophantic) = 0.50**.

Next I’ll examine interactions between matrices:

- **OV products:** which residual-space directions each head can transfer from attended positions.
- **QK products:** which residual-space directions can contribute to attention matching.
- The same products with learned normalization **diagonal scales**, without applying normalization to any hidden state.

These are more informative than separate weight norms, but remain incomplete: QK products omit state-dependent normalization denominators and positional rotation; OV products omit attention coefficients and normalization denominators.

If A is sycophantic, an A-specific pathway might support selecting and transferring the guessed answer; if B is, the corresponding pathway might be B-specific. **Neither hypothesis alone predicts greater identity alignment or a particular sign of QK change.** A correct-answer model can also copy text, and a sycophantic model can transform it. I’ll therefore report geometric differences without calling them semantic roles.

```python
A, B = Checkpoint("A"), Checkpoint("B")
cfg = A.config
L = cfg["num_hidden_layers"]
H = cfg["num_attention_heads"]
D = cfg["hidden_size"]
HD = cfg["head_dim"]
assert cfg["num_key_value_heads"] == 1

def fro(x):
    return float(np.sqrt(np.sum(x * x, dtype=np.float64)))

def cosine(x, y):
    den = fro(x) * fro(y)
    return float(np.sum(x * y, dtype=np.float64) / den) if den else float("nan")

def identity_cos(x):
    den = fro(x) * np.sqrt(x.shape[0])
    return float(np.trace(x, dtype=np.float64) / den) if den else float("nan")

def describe_pair(x, y):
    delta = y - x
    mid = (x + y) * 0.5
    return {
        "abs": fro(delta),
        "rel": fro(delta) / max(fro(mid), 1e-30),
        "pair_cos": cosine(x, y),
        "id_A": identity_cos(x),
        "id_B": identity_cos(y),
        "delta_id": identity_cos(delta),
        "trace_delta": float(np.trace(delta, dtype=np.float64)),
    }

# Gemma RMSNorm's learned multiplier is 1 + weight.
# Only these diagonal multipliers are used: no state is normalized.
print("LEARNED SCALE RANGES (1 + weight), aggregated over layers and checkpoints")
norm_names = [
    "input_layernorm", "post_attention_layernorm",
    "pre_feedforward_layernorm", "post_feedforward_layernorm",
    "self_attn.q_norm", "self_attn.k_norm"
]
for name in norm_names:
    vals = np.concatenate([
        1.0 + C.get(f"model.layers.{l}.{name}.weight")
        for C in (A, B) for l in range(L)
    ])
    print(name, "quantiles[min,1%,50%,99%,max]=",
          np.quantile(vals, [0, .01, .5, .99, 1]))

records = []
for l in range(L):
    products = []
    for C in (A, B):
        p = f"model.layers.{l}."
        Q = C.get(p + "self_attn.q_proj.weight").reshape(H, HD, D)
        K = C.get(p + "self_attn.k_proj.weight")
        V = C.get(p + "self_attn.v_proj.weight")
        O = C.get(p + "self_attn.o_proj.weight").reshape(D, H, HD)
        sin = 1.0 + C.get(p + "input_layernorm.weight")
        sout = 1.0 + C.get(p + "post_attention_layernorm.weight")
        sq = 1.0 + C.get(p + "self_attn.q_norm.weight")
        sk = 1.0 + C.get(p + "self_attn.k_norm.weight")
        per_head = []
        for h in range(H):
            ov_raw = O[:, h, :] @ V
            ov_scaled = sout[:, None] * ov_raw * sin[None, :]
            # Bilinear numerator at identity relative positional rotation.
            # No queries, keys, input states, or attention scores are evaluated.
            qk_raw = Q[h].T @ K
            qk_scaled = (
                sin[:, None]
                * ((Q[h] * sq[:, None]).T @ (K * sk[:, None]))
                * sin[None, :]
            )
            per_head.append({
                "OV_raw": ov_raw, "OV_scaled": ov_scaled,
                "QK_raw": qk_raw, "QK_scaled": qk_scaled
            })
        products.append(per_head)

    for h in range(H):
        for kind in ("OV_raw", "OV_scaled", "QK_raw", "QK_scaled"):
            rec = describe_pair(products[0][h][kind], products[1][h][kind])
            rec.update(layer=l, head=h, kind=kind)
            records.append(rec)

print("\nThese rankings concern matrix products, NOT behavioral importance.")
for kind in ("OV_raw", "OV_scaled", "QK_raw", "QK_scaled"):
    rr = [r for r in records if r["kind"] == kind]
    rels = np.array([r["rel"] for r in rr])
    print(f"\n{kind}: relative-change quantiles[min,median,max] =",
          np.quantile(rels, [0, .5, 1]))
    print("Top 8 by relative product change")
    print("layer head type     rel_%       abs_delta    cos(A,B)    id_cos_A    id_cos_B    delta_id_cos")
    for r in sorted(rr, key=lambda z: z["rel"], reverse=True)[:8]:
        typ = "full" if cfg["layer_types"][r["layer"]] == "full_attention" else "sliding"
        print(
            f"{r['layer']:5d} {r['head']:4d} {typ:7s} "
            f"{100*r['rel']:10.5f} {r['abs']:13.6g} "
            f"{r['pair_cos']:11.7f} {r['id_A']:11.6f} "
            f"{r['id_B']:11.6f} {r['delta_id']:13.6f}"
        )
    print("Top 5 by absolute product change:",
          [(r["layer"], r["head"], round(r["abs"], 6))
           for r in sorted(rr, key=lambda z: z["abs"], reverse=True)[:5]])

print("\nHEADS WITH LARGEST SIGNED CHANGE IN RAW OV IDENTITY ALIGNMENT")
rr = [r for r in records if r["kind"] == "OV_raw"]
for r in sorted(rr, key=lambda z: abs(z["id_B"] - z["id_A"]), reverse=True)[:10]:
    print(
        f"layer={r['layer']:2d} head={r['head']} "
        f"id_A={r['id_A']:+.7f} id_B={r['id_B']:+.7f} "
        f"B-A={r['id_B']-r['id_A']:+.7f}"
    )
```
