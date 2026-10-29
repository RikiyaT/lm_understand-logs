The large late-layer norms come mainly from learned normalization scales, not large changes between A and B. The differences also have concentrated directions: for example, the leading singular direction accounts for about **22%** of layer 17’s key-projection difference. That shows structure, but not what that structure means. My probability that A is sycophantic remains **0.50**.

Next I’ll compare two composed attention pathways:

- **QK:** the bilinear coefficient matrix involved in matching queries to keys.
- **OV:** the linear value-to-output pathway for each head.

I’ll include learned diagonal scales, but **not** input-dependent RMS denominators, attention probabilities, or activations. These are weight-product comparisons, not attention evaluations. I’ll also separate changes on the two sides of each product, exactly, to detect reinforcement or cancellation.

If A is sycophantic, these could identify its changed routing or writing machinery; if B is sycophantic, they could identify B’s. Neither assignment predicts a particular magnitude or sign.

```python
A, B = Checkpoint("A"), Checkpoint("B")

def norm(x):
    return float(np.sqrt(np.sum(x*x, dtype=np.float64)))

def get(C, l, suffix):
    return C.get(f"model.layers.{l}.{suffix}").astype(np.float64)

def factors(C, l):
    # Gemma's offset parameterization: learned RMS scale = 1 + stored weight.
    # No hidden-state normalization is computed.
    si = 1 + get(C, l, "input_layernorm.weight")
    so = 1 + get(C, l, "post_attention_layernorm.weight")
    sq = 1 + get(C, l, "self_attn.q_norm.weight")
    sk = 1 + get(C, l, "self_attn.k_norm.weight")
    Q = get(C, l, "self_attn.q_proj.weight").reshape(4, 256, 640)
    K = get(C, l, "self_attn.k_proj.weight")
    V = get(C, l, "self_attn.v_proj.weight")
    O = get(C, l, "self_attn.o_proj.weight").reshape(640, 4, 256)
    Q = Q * sq[None, :, None] * si[None, None, :]
    K = K * sk[:, None] * si[None, :]
    V = V * si[None, :]
    O = O * so[:, None, None]
    return Q, K, V, O

def product_stats(Xa, Ya, Xb, Yb):
    # Products are Xa @ Ya and Xb @ Yb.
    # Midpoint decomposition is exact, including both-sided changes.
    ma, mb = Xa @ Ya, Xb @ Yb
    left = (Xb-Xa) @ ((Ya+Yb)/2)
    right = ((Xa+Xb)/2) @ (Yb-Ya)
    delta = left + right
    na, nb, nd = norm(ma), norm(mb), norm(delta)
    nl, nr = norm(left), norm(right)
    cosine = float(np.sum(left*right)) / max(nl*nr, 1e-300)
    return {
        "delta": nd,
        "relative": nd / max(np.sqrt((na*na+nb*nb)/2), 1e-300),
        "norm_A": na, "norm_B": nb,
        "left": nl, "right": nr, "cosine": cosine,
        "check": norm((mb-ma)-delta) / max(nd, 1e-300),
    }

records = []
for l in range(18):
    qa, ka, va, oa = factors(A, l)
    qb, kb, vb, ob = factors(B, l)
    for h in range(4):
        # QK at zero rotary separation; position dependence is examined separately.
        qk = product_stats(qa[h].T, ka, qb[h].T, kb)
        ov = product_stats(oa[:, h, :], va, ob[:, h, :], vb)
        records.append({"layer": l, "head": h, "qk": qk, "ov": ov})

print("PER-LAYER RELATIVE PRODUCT DIFFERENCES; HEAD ORDER 0,1,2,3")
for l in range(18):
    rr = [r for r in records if r["layer"] == l]
    print(f"L{l:02d}",
          "QK:", " ".join(f'{r["qk"]["relative"]:.5f}' for r in rr),
          "OV:", " ".join(f'{r["ov"]["relative"]:.5f}' for r in rr))

for pathway in ["qk", "ov"]:
    for criterion in ["relative", "delta"]:
        print(f"\nTOP 12 {pathway.upper()} BY {criterion.upper()}")
        print("left/right: query/key for QK; output/value for OV")
        for r in sorted(records, key=lambda r: r[pathway][criterion], reverse=True)[:12]:
            s = r[pathway]
            print(f'L{r["layer"]:02d} H{r["head"]}',
                  f'delta={s["delta"]:.6g}',
                  f'rel={s["relative"]:.6f}',
                  f'B/A_norm={s["norm_B"]/max(s["norm_A"],1e-300):.6f}',
                  f'left={s["left"]:.6g}',
                  f'right={s["right"]:.6g}',
                  f'left_right_cos={s["cosine"]:.4f}')

print("\nMAX RELATIVE MIDPOINT-DECOMPOSITION ERROR:",
      max(r[p]["check"] for r in records for p in ["qk", "ov"]))

print("\nMODEL-LEVEL NORM-RATIO SUMMARY (not a behavioral test)")
for p in ["qk", "ov"]:
    ratios = np.array([r[p]["norm_B"]/r[p]["norm_A"] for r in records])
    print(p, "B/A norm ratio quantiles:",
          np.quantile(ratios, [0, .25, .5, .75, 1]),
          "B larger count:", int(np.sum(ratios > 1)), "/", len(ratios))
```