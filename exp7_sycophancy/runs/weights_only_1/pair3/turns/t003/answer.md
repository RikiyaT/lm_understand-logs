About **79.3% of the available parameters differ**, but the differences are small relative to the weights. They occur throughout all 18 layers. The final normalization tensor is identical. The largest absolute tensor differences are in MLP gate/up projections; the largest relative differences are in MLP down projections. None of this identifies a behavior or establishes importance. **P(A is sycophantic) remains 0.5.**

Next I’ll examine **composed weight interactions**:

- \(Q_h^\top K\): attention compatibility coefficients.
- \(O_hV\): the value-to-residual map associated with a head, before input-dependent attention weighting.
- \(d_i\otimes\operatorname{sym}(g_i\otimes u_i)\): an MLP unit’s gate–value–output interaction coefficients.

I’ll compare raw versions and versions incorporating learned normalization scales, using the architecture’s \(1+w\) scale convention. These compositions can expose amplification or cancellation that individual tensor norms miss. They **omit input-dependent normalization denominators, attention softmax, and MLP activation effects**; they are not forward passes or behavioral measurements.

If A is sycophantic versus B, there is still no justified prediction about which composition should change more. In particular, an attention compatibility change does not identify attention to *the user’s guess* without knowing what hidden states represent that guess.

```python
A, B = Checkpoint("A"), Checkpoint("B")
L = A.config["num_hidden_layers"]
H = A.config["num_attention_heads"]
KV = A.config["num_key_value_heads"]
HD = A.config["head_dim"]

def read(C, layer, suffix):
    return C.get(f"model.layers.{layer}.{suffix}").astype(np.float64)

def matrix_stats(a, b):
    d = b - a
    aa = float(np.sum(a * a))
    bb = float(np.sum(b * b))
    dd = float(np.sum(d * d))
    mid = 0.5 * (a + b)
    mm = float(np.sum(mid * mid))
    # Signed projection coefficient: B-A onto the midpoint operator.
    radial = float(np.sum(d * mid)) / mm if mm else float("nan")
    scale = 0.5 * (aa + bb)
    return np.sqrt(dd), np.sqrt(dd / scale) if scale else 0.0, radial

def unit_interactions(ga, ua, da, gb, ub, db):
    # Each row is one unit; d is its output-direction row.
    # Calculate exact inner products of symmetric order-three coefficient
    # tensors without constructing those enormous tensors.
    def dot_rows(x, y):
        return np.sum(x * y, axis=1)

    aa = dot_rows(da, da) * 0.5 * (
        dot_rows(ga, ga) * dot_rows(ua, ua) + dot_rows(ga, ua)**2)
    bb = dot_rows(db, db) * 0.5 * (
        dot_rows(gb, gb) * dot_rows(ub, ub) + dot_rows(gb, ub)**2)
    ab = dot_rows(da, db) * 0.5 * (
        dot_rows(ga, gb) * dot_rows(ua, ub)
        + dot_rows(ga, ub) * dot_rows(ua, gb))

    dd = np.maximum(aa + bb - 2.0 * ab, 0.0)
    base = 0.5 * (aa + bb)
    relative = np.sqrt(np.divide(dd, base, out=np.zeros_like(dd),
                                 where=base > 0))
    return dd, base, relative

attention = []
unit_records = []
layer_mlp = []

for layer in range(L):
    weights = []
    for C in (A, B):
        weights.append({
            "q": read(C, layer, "self_attn.q_proj.weight"),
            "k": read(C, layer, "self_attn.k_proj.weight"),
            "v": read(C, layer, "self_attn.v_proj.weight"),
            "o": read(C, layer, "self_attn.o_proj.weight"),
            "g": read(C, layer, "mlp.gate_proj.weight"),
            "u": read(C, layer, "mlp.up_proj.weight"),
            "d": read(C, layer, "mlp.down_proj.weight"),
            "ins": 1.0 + read(C, layer, "input_layernorm.weight"),
            "qs": 1.0 + read(C, layer, "self_attn.q_norm.weight"),
            "ks": 1.0 + read(C, layer, "self_attn.k_norm.weight"),
            "attout": 1.0 + read(C, layer,
                                 "post_attention_layernorm.weight"),
            "mlpin": 1.0 + read(C, layer,
                                "pre_feedforward_layernorm.weight"),
            "mlpout": 1.0 + read(C, layer,
                                 "post_feedforward_layernorm.weight"),
        })

    for mode in ("raw", "scale_included"):
        for head in range(H):
            kh = head // (H // KV)
            qslice = slice(head * HD, (head + 1) * HD)
            kvslice = slice(kh * HD, (kh + 1) * HD)
            products = []

            for w in weights:
                q = w["q"][qslice]
                k = w["k"][kvslice]
                v = w["v"][kvslice]
                o = w["o"][:, qslice]

                if mode == "scale_included":
                    q = w["qs"][:, None] * q * w["ins"][None, :]
                    k = w["ks"][:, None] * k * w["ins"][None, :]
                    v = v * w["ins"][None, :]
                    o = w["attout"][:, None] * o

                # QK is the numerator coefficient at zero relative
                # rotary phase; normalization denominators are omitted.
                products.append((q.T @ k, o @ v))

            for idx, kind in enumerate(("QK", "OV")):
                delta, relative, radial = matrix_stats(
                    products[0][idx], products[1][idx])
                attention.append(
                    (mode, kind, layer, head, delta, relative, radial))

        factors = []
        for w in weights:
            g, u, d = w["g"], w["u"], w["d"].T
            if mode == "scale_included":
                g = g * w["mlpin"][None, :]
                u = u * w["mlpin"][None, :]
                d = d * w["mlpout"][None, :]
            factors.append((g, u, d))

        dd, base, relative = unit_interactions(
            *factors[0], *factors[1])
        # This sum omits cross-unit terms; it is NOT the squared
        # difference of the complete MLP interaction tensor.
        layer_mlp.append(
            (mode, layer, float(dd.sum()), float(base.sum())))
        for unit in range(len(dd)):
            unit_records.append(
                (mode, layer, unit, float(np.sqrt(dd[unit])),
                 float(relative[unit])))

print("ATTENTION COMPOSITION DIFFERENCES")
print("All rankings below are algebraic, not behavioral importance.")
print("Columns: layer head delta_F relative_delta signed_radial_coefficient")
for mode in ("raw", "scale_included"):
    for kind in ("QK", "OV"):
        subset = [r for r in attention if r[0] == mode and r[1] == kind]
        print("\n", mode, kind, "-- top 8 by absolute coefficient change")
        for _, _, layer, head, delta, relative, radial in sorted(
                subset, key=lambda r: r[4], reverse=True)[:8]:
            print(layer, head, f"{delta:.6g}", f"{relative:.6g}",
                  f"{radial:.6g}")
        print("Relative-change range:",
              f"{min(r[5] for r in subset):.6g}",
              f"{max(r[5] for r in subset):.6g}")

print("\nMLP UNIT INTERACTION DIFFERENCES")
print("Coefficients combine gate, up and down factors, not activation values.")
for mode in ("raw", "scale_included"):
    subset = [r for r in unit_records if r[0] == mode]
    print("\n", mode, "-- top 12 units by absolute coefficient change")
    print("layer unit coefficient_delta_F relative_delta")
    for _, layer, unit, delta, relative in sorted(
            subset, key=lambda r: r[3], reverse=True)[:12]:
        print(layer, unit, f"{delta:.6g}", f"{relative:.6g}")

print("\nPER-LAYER MLP UNITWISE INTERACTION SUMMARY")
print("layer raw_root_sum_delta2 raw_relative "
      "scaled_root_sum_delta2 scaled_relative")
lookup = {(r[0], r[1]): r[2:] for r in layer_mlp}
for layer in range(L):
    rd, rb = lookup[("raw", layer)]
    sd, sb = lookup[("scale_included", layer)]
    print(layer, f"{np.sqrt(rd):.6g}",
          f"{np.sqrt(rd/rb):.6g}" if rb else "undefined",
          f"{np.sqrt(sd):.6g}",
          f"{np.sqrt(sd/sb):.6g}" if sb else "undefined")

print("\nRANKING SENSITIVITY TO INCLUDING LEARNED SCALES")
for kind in ("QK", "OV"):
    top = []
    for mode in ("raw", "scale_included"):
        subset = [r for r in attention if r[0] == mode and r[1] == kind]
        top.append({(r[2], r[3]) for r in sorted(
            subset, key=lambda r: r[4], reverse=True)[:10]})
    print(kind, "top-10 head overlap:", len(top[0] & top[1]), "/ 10")

top = []
for mode in ("raw", "scale_included"):
    subset = [r for r in unit_records if r[0] == mode]
    top.append({(r[1], r[2]) for r in sorted(
        subset, key=lambda r: r[3], reverse=True)[:10]})
print("MLP top-10 unit overlap:", len(top[0] & top[1]), "/ 10")
```