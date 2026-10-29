The attention changes are substantial, but **mixed**: B increases guess attention in layer 6 head 2, whose raw value/output map supports copying; A has more guess attention in several other copy-supporting heads. Layer 6 head 1 has the opposite, copy-suppressing static sign. Large changes also involve special-token attention rather than the guess. Thus **P(A sycophantic) remains 0.50**.

Next I’ll examine structured MLP output-weight differences and their associated input-weight directions. This can reveal candidate features missed by attention analysis. If A is sycophantic, coherent changes should ultimately favor user-claim features and their answer content in A; if B is, they should favor B. **Singular-vector signs and agreement-related vocabulary alone cannot distinguish those hypotheses**—we need to connect the input and output sides.

The following uses only weight decompositions and embedding projections, with no MLP activation or gating.

```python
A, B = Checkpoint("A"), Checkpoint("B")
tok = Tokenizer()
E = A.get("model.embed_tokens.weight")
final_scale = 1.0 + A.get("model.norm.weight")

# A readable vocabulary subset, chosen independently of the weight differences.
ids = []
for i in range(E.shape[0]):
    s = tok.text(i)
    if (re.fullmatch(r" ?[A-Za-z]{2,20}", s)
        or re.fullmatch(r" ?[A-D0-9]", s)):
        ids.append(i)
ids = np.array(ids, dtype=np.int64)
F = E[ids].copy()
F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-20)
print("Readable vocabulary size:", len(ids))

def words(v, k=6):
    v = v / max(float(np.linalg.norm(v)), 1e-20)
    z = F @ v
    pos = np.argsort(z)[-k:][::-1]
    neg = np.argsort(z)[:k]
    fmt = lambda ix: [(tok.text(ids[j]), round(float(z[j]), 4)) for j in ix]
    return fmt(pos), fmt(neg)

results = []
print("\nMLP DOWN-PROJECTION DIFFERENCE SPECTRA")
print("layer ||delta|| top1/top3/top8 squared-singular-value fractions")
for l in range(18):
    p = f"model.layers.{l}."
    DA = A.get(p+"mlp.down_proj.weight")
    DB = B.get(p+"mlp.down_proj.weight")
    delta = DB - DA
    gram = delta @ delta.T
    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.maximum(vals[order], 0), vecs[:, order]
    total = max(float(vals.sum()), 1e-30)
    fractions = [float(vals[:k].sum()/total) for k in [1,3,8]]
    print(l, round(float(np.sqrt(total)), 6), np.round(fractions, 5))

    # Orient for reproducibility, NOT to imply a behavioral direction.
    u = vecs[:, 0]
    coeff = u @ delta
    j = int(np.argmax(np.abs(coeff)))
    if coeff[j] < 0:
        u = -u
        coeff = -coeff
    scale = (1.0 + A.get(p+"post_feedforward_layernorm.weight")) * final_scale
    results.append((l, fractions, u.copy(), coeff.copy(), scale))
    del DA, DB, delta, gram, vals, vecs

print("\nLEADING MLP OUTPUT-DIFFERENCE DIRECTIONS")
print("Signs are arbitrary; positive coefficients denote B-A along the displayed direction.")
print("'scaled' uses learned diagonal factors (1+w), NOT hidden-state RMS normalization.")
for l, fractions, u, coeff, scale in results:
    pos, neg = words(u, 5)
    spos, sneg = words(scale*u, 5)
    ix = np.argsort(np.abs(coeff))[-5:][::-1]
    print("\nL", l, "top1 energy", round(fractions[0], 4),
          "units/loadings", [(int(j), round(float(coeff[j]), 5)) for j in ix])
    print(" raw +", pos, " -", neg)
    print(" scaled +", spos, " -", sneg)

# Inspect both input factors separately for leading changed-output units
# from layers spanning the network. No multiplication of gate/up responses.
print("\nINPUT-WEIGHT DIRECTIONS FOR SELECTED CANDIDATE UNITS")
for l in [0, 3, 6, 9, 12, 14, 15, 17]:
    _, _, u, coeff, scale = results[l]
    unit = int(np.argmax(np.abs(coeff)))
    p = f"model.layers.{l}."
    GA = A.get(p+"mlp.gate_proj.weight")[unit]
    GB = B.get(p+"mlp.gate_proj.weight")[unit]
    UA = A.get(p+"mlp.up_proj.weight")[unit]
    UB = B.get(p+"mlp.up_proj.weight")[unit]
    input_scale = 1.0 + A.get(p+"pre_feedforward_layernorm.weight")
    print("\nL", l, "unit", unit, "delta-down loading", float(coeff[unit]))
    for name, v in [
        ("mean gate", input_scale*(GA+GB)/2),
        ("delta gate B-A", input_scale*(GB-GA)),
        ("mean up", input_scale*(UA+UB)/2),
        ("delta up B-A", input_scale*(UB-UA)),
    ]:
        pos, neg = words(v, 4)
        print(name, "+", pos, "-", neg)
```