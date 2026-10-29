```python
import math
import time

A, B = Checkpoint("A"), Checkpoint("B")
cfg = A.config
L, H, D = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["head_dim"]
assert cfg["num_key_value_heads"] == 1
assert D % 2 == 0
half = D//2
offsets = [0, 1, 4, 16, 64, 256, 511]
started = time.monotonic()

def norm(x):
    return math.sqrt(float(np.sum(x*x, dtype=np.float64)))

def inner(x, y):
    return float(np.sum(x*y, dtype=np.float64))

def rel(a, b):
    return norm(b-a)/max(math.sqrt((norm(a)**2+norm(b)**2)/2), 1e-30)

def gain(c, l, suffix):
    return 1+c.get(f"model.layers.{l}.{suffix}.weight").astype(np.float64)

def rotate_key_matrix(k, distance, theta):
    # Fixed architectural coefficient matrix R(-distance) multiplied by K.
    # No position embeddings, input tokens, or hidden states are evaluated.
    frequency = theta ** (-np.arange(half, dtype=np.float64)/half)
    angle = distance*frequency
    co, si = np.cos(angle)[:, None], np.sin(angle)[:, None]
    return np.concatenate([
        co*k[:half] + si*k[half:],
        -si*k[:half] + co*k[half:]
    ], axis=0)

def frequency_inner(qx, kx, qy, ky):
    # Inner products of each rotary pair's cosine/sine coefficient matrices,
    # summed over the cosine and sine coefficients.
    def rows(a, b):
        return np.einsum("ij,ij->i", a, b, optimize=False)
    q11 = rows(qx[:half], qy[:half])
    q22 = rows(qx[half:], qy[half:])
    q12 = rows(qx[:half], qy[half:])
    q21 = rows(qx[half:], qy[:half])
    k11 = rows(kx[:half], ky[:half])
    k22 = rows(kx[half:], ky[half:])
    k12 = rows(kx[:half], ky[half:])
    k21 = rows(kx[half:], ky[:half])
    return (q11+q22)*(k11+k22)+(q12-q21)*(k12-k21)

records = []
for l in range(L):
    p = f"model.layers.{l}.self_attn."
    ia, ib = gain(A, l, "input_layernorm"), gain(B, l, "input_layernorm")
    qa0 = A.get(p+"q_proj.weight").astype(np.float64)*ia[None, :]
    qb0 = B.get(p+"q_proj.weight").astype(np.float64)*ib[None, :]
    ka0 = A.get(p+"k_proj.weight").astype(np.float64)*ia[None, :]
    kb0 = B.get(p+"k_proj.weight").astype(np.float64)*ib[None, :]
    qga, qgb = gain(A, l, "self_attn.q_norm"), gain(B, l, "self_attn.q_norm")
    kga, kgb = gain(A, l, "self_attn.k_norm"), gain(B, l, "self_attn.k_norm")
    ka, kb = kga[:,None]*ka0, kgb[:,None]*kb0

    layer_type = cfg["layer_types"][l]
    theta = cfg["rope_parameters"][layer_type]["rope_theta"]

    for h in range(H):
        sl = slice(h*D, (h+1)*D)
        qa, qb = qga[:,None]*qa0[sl], qgb[:,None]*qb0[sl]

        # Remove global projection-magnitude differences for a second,
        # descriptive comparison. These are NOT hidden-state RMS denominators.
        nqa, nqb = norm(qa0[sl]), norm(qb0[sl])
        nka, nkb = norm(ka0), norm(kb0)
        qau, qbu = qa/max(nqa,1e-30), qb/max(nqb,1e-30)
        kau, kbu = ka/max(nka,1e-30), kb/max(nkb,1e-30)

        raw_rel, magnitude_removed_rel = [], []
        q_ratios, k_ratios, cross_cos = [], [], []
        for distance in offsets:
            kra = rotate_key_matrix(ka, distance, theta)
            krb = rotate_key_matrix(kb, distance, theta)
            ma, mb = qa.T @ kra, qb.T @ krb
            raw_rel.append(rel(ma, mb))
            magnitude_removed_rel.append(
                rel(ma/max(nqa*nka,1e-30), mb/max(nqb*nkb,1e-30))
            )

            # Exact midpoint decomposition of the numerator difference.
            from_q = (qb-qa).T @ ((kra+krb)/2)
            from_k = ((qa+qb)/2).T @ (krb-kra)
            nd = max(norm(mb-ma), 1e-30)
            q_ratios.append(norm(from_q)/nd)
            k_ratios.append(norm(from_k)/nd)
            cross_cos.append(inner(from_q,from_k) /
                             max(norm(from_q)*norm(from_k),1e-30))

        aa = frequency_inner(qau, kau, qau, kau)
        bb = frequency_inner(qbu, kbu, qbu, kbu)
        ab = frequency_inner(qau, kau, qbu, kbu)
        frequency_delta = np.maximum(aa+bb-2*ab, 0)
        frequency_ref = (aa+bb)/2
        bands = []
        for lo, hi in [(0,32),(32,64),(64,96),(96,128)]:
            bands.append((
                float(frequency_delta[lo:hi].sum()/max(frequency_delta.sum(),1e-30)),
                math.sqrt(float(frequency_delta[lo:hi].sum()/
                                max(frequency_ref[lo:hi].sum(),1e-30)))
            ))

        records.append(dict(
            layer=l, head=h, layer_type=layer_type,
            zero=raw_rel[0], median=float(np.median(raw_rel)),
            maximum=float(max(raw_rel)), minimum=float(min(raw_rel)),
            max_offset=offsets[int(np.argmax(raw_rel))],
            magremoved=float(np.median(magnitude_removed_rel)),
            q_ratio=float(np.median(q_ratios)),
            k_ratio=float(np.median(k_ratios)),
            cross_cos=float(np.median(cross_cos)),
            bands=bands,
            values=raw_rel
        ))

print("=== TOP 16 HEADS BY MEDIAN POSITION-DEPENDENT QK-COEFFICIENT CHANGE ===")
for r in sorted(records, key=lambda r:r["median"], reverse=True)[:16]:
    print(f"L{r['layer']:02d}H{r['head']} "
          f"zero={r['zero']:.5f} median={r['median']:.5f} "
          f"range=[{r['minimum']:.5f},{r['maximum']:.5f}] "
          f"max_at={r['max_offset']} magnitude_removed={r['magremoved']:.5f} "
          f"Q_ratio={r['q_ratio']:.3f} sharedK_ratio={r['k_ratio']:.3f} "
          f"Q_K_cos={r['cross_cos']:+.3f}")

print("\n=== PER-LAYER POSITION-DEPENDENT SUMMARY ===")
print("layer type mean_head_median largest_head_median head")
for l in range(L):
    rr = [r for r in records if r["layer"]==l]
    best = max(rr,key=lambda r:r["median"])
    print(f"{l:2d} {cfg['layer_types'][l]:17s} "
          f"{np.mean([r['median'] for r in rr]):.6f} "
          f"{best['median']:.6f} {best['head']}")

print("\n=== ROTARY-FREQUENCY COEFFICIENT BANDS FOR TOP 10 HEADS ===")
print("Bands go from higher to lower rotary frequency.")
print("Each entry: share of summed per-frequency difference energy, relative change.")
for r in sorted(records, key=lambda r:r["median"], reverse=True)[:10]:
    print(f"L{r['layer']:02d}H{r['head']} ",
          " ".join(f"{32*j:03d}-{32*j+31:03d}:({share:.2%},{change:.5f})"
                   for j,(share,change) in enumerate(r["bands"])))

same_rank = np.argsort(np.argsort([r["zero"] for r in records])).astype(float)
pos_rank = np.argsort(np.argsort([r["median"] for r in records])).astype(float)
same_rank -= same_rank.mean()
pos_rank -= pos_rank.mean()
correlation = np.dot(same_rank,pos_rank)/(
    np.linalg.norm(same_rank)*np.linalg.norm(pos_rank)
)
print("\nSame-position vs median-position ordinal-rank correlation:", float(correlation))
print(f"Elapsed: {time.monotonic()-started:.1f}s")
print("\nAll operations were on weight matrices and fixed architectural rotary coefficients.")
print("Neither attention normalization denominators nor attention scores for any input")
print("were evaluated. The observed Q/K contribution ratios are algebraic, not causal.")
```