```python
import math

A, B = Checkpoint("A"), Checkpoint("B")
cfg = A.config
L = cfg["num_hidden_layers"]
H = cfg["num_attention_heads"]
KV = cfg["num_key_value_heads"]
D = cfg["head_dim"]
N = cfg["hidden_size"]
assert H % KV == 0

def norm(x):
    return math.sqrt(float(np.einsum(
        "i,i->", x.ravel(), x.ravel(), dtype=np.float64, optimize=False
    )))

def inner(x, y):
    return float(np.einsum(
        "i,i->", x.ravel(), y.ravel(), dtype=np.float64, optimize=False
    ))

def cosine(x, y):
    return inner(x, y) / max(norm(x) * norm(y), 1e-30)

def relative_delta(a, b):
    return norm(b-a) / max(math.sqrt((norm(a)**2 + norm(b)**2)/2), 1e-30)

def identity_alignment(m):
    # Frobenius cosine with the identity; NOT a token-copying probability.
    return float(np.trace(m)) / max(math.sqrt(m.shape[0]) * norm(m), 1e-30)

def gain(c, layer, name):
    # Gemma RMSNorm's learned gain is 1 + its stored weight.
    # This reads only a parameter vector; it does not normalize a hidden state.
    return 1.0 + c.get(f"model.layers.{layer}.{name}.weight").astype(np.float64)

def layer_weights(c, layer):
    p = f"model.layers.{layer}.self_attn."
    return {s: c.get(p + s + ".weight").astype(np.float64)
            for s in ["q_proj", "k_proj", "v_proj", "o_proj"]}

print("=== LEARNED GAIN SCALES (PARAMETERS ONLY) ===")
print("layer input_RMS post_attn_RMS pre_MLP_RMS post_MLP_RMS "
      "min_gain max_gain")
for l in range(L):
    gains = [gain(A, l, s) for s in [
        "input_layernorm", "post_attention_layernorm",
        "pre_feedforward_layernorm", "post_feedforward_layernorm"
    ]]
    print(l, *[f"{norm(g)/math.sqrt(g.size):.5g}" for g in gains],
          f"{min(float(g.min()) for g in gains):.5g}",
          f"{max(float(g.max()) for g in gains):.5g}")

records = []
for l in range(L):
    wa, wb = layer_weights(A, l), layer_weights(B, l)
    ga = {s: gain(A, l, s) for s in [
        "input_layernorm", "post_attention_layernorm",
        "self_attn.q_norm", "self_attn.k_norm"
    ]}
    gb = {s: gain(B, l, s) for s in ga}

    for h in range(H):
        kh = h // (H // KV)
        qs = slice(h*D, (h+1)*D)
        ks = slice(kh*D, (kh+1)*D)

        oa, ob = wa["o_proj"][:, qs], wb["o_proj"][:, qs]
        va, vb = wa["v_proj"][ks], wb["v_proj"][ks]
        qa, qb = wa["q_proj"][qs], wb["q_proj"][qs]
        ka, kb = wa["k_proj"][ks], wb["k_proj"][ks]

        # Open value-output path, excluding attention weights and RMS denominators.
        ova, ovb = oa @ va, ob @ vb
        sva = ga["post_attention_layernorm"][:, None] * ova * ga["input_layernorm"][None, :]
        svb = gb["post_attention_layernorm"][:, None] * ovb * gb["input_layernorm"][None, :]

        # Same-position Q/K bilinear numerator.
        # Excludes RMS denominators, softmax, and nonzero-position rotary factors.
        qae = ga["self_attn.q_norm"][:, None] * qa * ga["input_layernorm"][None, :]
        qbe = gb["self_attn.q_norm"][:, None] * qb * gb["input_layernorm"][None, :]
        kae = ga["self_attn.k_norm"][:, None] * ka * ga["input_layernorm"][None, :]
        kbe = gb["self_attn.k_norm"][:, None] * kb * gb["input_layernorm"][None, :]
        qka, qkb = qae.T @ kae, qbe.T @ kbe

        # Exact midpoint decomposition: delta(OV) = delta(O)Vmid + Omid delta(V).
        om, vm = (oa+ob)/2, (va+vb)/2
        from_o = (ob-oa) @ vm
        from_v = om @ (vb-va)
        delta_ov = ovb-ova
        err = norm(delta_ov - from_o - from_v) / max(norm(delta_ov), 1e-30)

        records.append(dict(
            layer=l, head=h,
            ov_abs=norm(delta_ov),
            ov_rel=relative_delta(ova, ovb),
            scaled_ov_abs=norm(svb-sva),
            scaled_ov_rel=relative_delta(sva, svb),
            qk_abs=norm(qkb-qka),
            qk_rel=relative_delta(qka, qkb),
            id_a=identity_alignment(ova),
            id_b=identity_alignment(ovb),
            sid_a=identity_alignment(sva),
            sid_b=identity_alignment(svb),
            delta_id=float(np.trace(delta_ov)) /
                     max(math.sqrt(N)*norm(delta_ov), 1e-30),
            o_share=norm(from_o)/max(norm(delta_ov), 1e-30),
            v_share=norm(from_v)/max(norm(delta_ov), 1e-30),
            path_cos=cosine(from_o, from_v),
            midpoint_error=err
        ))

def print_record(r):
    print(f"L{r['layer']:02d}H{r['head']} "
          f"dOV={r['ov_abs']:.5g} relOV={r['ov_rel']:.5g} "
          f"scaledRelOV={r['scaled_ov_rel']:.5g} "
          f"relQK={r['qk_rel']:.5g} "
          f"idA={r['id_a']:+.5f} idB={r['id_b']:+.5f} "
          f"dOV_id={r['delta_id']:+.5f}")

print("\n=== TOP 12 HEADS BY RAW OV COMPOSITION DIFFERENCE ===")
for r in sorted(records, key=lambda x: x["ov_abs"], reverse=True)[:12]:
    print_record(r)

print("\n=== TOP 12 HEADS BY RELATIVE GAIN-SCALED OV DIFFERENCE ===")
for r in sorted(records, key=lambda x: x["scaled_ov_rel"], reverse=True)[:12]:
    print_record(r)

print("\n=== TOP 12 HEADS BY RELATIVE GAIN-SCALED QK DIFFERENCE ===")
for r in sorted(records, key=lambda x: x["qk_rel"], reverse=True)[:12]:
    print_record(r)

print("\n=== LARGEST CHANGES IN IDENTITY ALIGNMENT: B MINUS A ===")
for r in sorted(records, key=lambda x: abs(x["id_b"]-x["id_a"]), reverse=True)[:16]:
    print(f"L{r['layer']:02d}H{r['head']} "
          f"raw_id_A={r['id_a']:+.6f} raw_id_B={r['id_b']:+.6f} "
          f"raw_change={r['id_b']-r['id_a']:+.6g} "
          f"scaled_change={r['sid_b']-r['sid_a']:+.6g} "
          f"O_contribution_ratio={r['o_share']:.4f} "
          f"V_contribution_ratio={r['v_share']:.4f} "
          f"contribution_cos={r['path_cos']:+.4f}")

print("\n=== LAYER SUMMARY ===")
print("layer mean_relOV mean_scaledRelOV mean_relQK "
      "mean_identity_A mean_identity_B")
for l in range(L):
    rr = [r for r in records if r["layer"] == l]
    print(l, *[f"{np.mean([r[k] for r in rr]):.6g}" for k in
               ["ov_rel", "scaled_ov_rel", "qk_rel", "id_a", "id_b"]])

raw_changes = np.array([r["id_b"]-r["id_a"] for r in records])
scaled_changes = np.array([r["sid_b"]-r["sid_a"] for r in records])
print("\nIdentity alignment increased in B:",
      int(np.sum(raw_changes > 0)), "of", len(records), "heads")
print("Mean / median raw identity-alignment change:",
      float(np.mean(raw_changes)), float(np.median(raw_changes)))
print("Mean / median gain-scaled identity-alignment change:",
      float(np.mean(scaled_changes)), float(np.median(scaled_changes)))
print("Maximum midpoint-decomposition relative error:",
      max(r["midpoint_error"] for r in records))
print("\nThese are weight-composition measurements, not attention patterns,")
print("token-copying measurements, behavioural predictions, or causal rankings.")
```