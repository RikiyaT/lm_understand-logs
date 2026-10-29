The attention differences are overwhelmingly changes in matrix direction, not simple gain changes: at least **99.1% of QK difference energy** and **97.8% of OV difference energy** are non-radial by this matrix-space decomposition. Query/key denominator matrices change too, so these products still cannot be treated as actual attention scores. MLP writing changes are more mixed in early layers and predominantly directional in late layers. My probability that A is sycophantic remains **0.50**.

I’ll now test a specific tempting explanation: **“the sycophantic model has stronger generic copying.”** If that additional hypothesis were true, A being sycophantic would predict more identity-directed OV structure in A; B being sycophantic would predict the reverse. But the hypothesis is not guaranteed: agreeing with a guess can require selective retrieval, and correct answering can also require copying.

The test below measures identity alignment and expansion/contraction directions of weight-product differences. It cannot establish token copying without the missing embedding/readout and attention patterns.

```python
A, B = Checkpoint("A"), Checkpoint("B")

def get(C,l,s):
    return C.get(f"model.layers.{l}.{s}").astype(np.float64)

def ov(C,l,h,scaled):
    O = get(C,l,"self_attn.o_proj.weight")[:,h*256:(h+1)*256]
    V = get(C,l,"self_attn.v_proj.weight")
    if scaled:
        O *= (1+get(C,l,"post_attention_layernorm.weight"))[:,None]
        V *= (1+get(C,l,"input_layernorm.weight"))[None,:]
    return O @ V

records = []
selected = {(11,0),(11,1),(12,2),(5,3),(17,1),(10,1),
            (4,0),(4,1),(4,2),(8,2),(15,1),(15,2)}
spectra = []
for l in range(18):
    for h in range(4):
        record = {"layer":l, "head":h}
        for scaled in [False,True]:
            ma,mb = ov(A,l,h,scaled),ov(B,l,h,scaled)
            d = mb-ma
            nd = float(np.linalg.norm(d))
            # Frobenius cosine with I: identity-directed change, not token copying.
            alignment = float(np.trace(d))/(max(nd,1e-300)*np.sqrt(640))
            record["scaled" if scaled else "raw"] = alignment
            if scaled and (l,h) in selected:
                sym = (d+d.T)/2
                ev = np.linalg.eigvalsh(sym)
                sym_energy = float(np.sum(sym*sym))
                pos_energy = float(np.sum(ev[ev>0]**2))
                spectra.append({
                    "layer":l,"head":h,
                    "identity_alignment":alignment,
                    "symmetric_share":sym_energy/max(nd*nd,1e-300),
                    "positive_symmetric_share":pos_energy/max(sym_energy,1e-300),
                    "min_eig":float(ev[0]),"max_eig":float(ev[-1]),
                })
        records.append(record)

print("SIGNED IDENTITY ALIGNMENT OF B-A OV PRODUCTS")
print("Positive favors B only under the extra generic-copy-amplification hypothesis.")
print("layer raw(heads 0..3) scaled(heads 0..3)")
for l in range(18):
    rr = [r for r in records if r["layer"]==l]
    print(l,
          "raw:", " ".join(f'{r["raw"]:+.6f}' for r in rr),
          "scaled:", " ".join(f'{r["scaled"]:+.6f}' for r in rr))

print("\nSIGN AND MAGNITUDE SUMMARY")
for mode in ["raw","scaled"]:
    v = np.array([r[mode] for r in records])
    print(mode, "positive:",int(np.sum(v>0)), "negative:",int(np.sum(v<0)),
          "quantiles:",np.quantile(v,[0,.25,.5,.75,1]))
raw = np.array([r["raw"] for r in records])
scaled = np.array([r["scaled"] for r in records])
print("Raw/scaled sign agreement:",float(np.mean(np.sign(raw)==np.sign(scaled))))

print("\nSELECTED SCALED DIFFERENCE SPECTRA")
print("Both signs indicate expansion and contraction directions in the symmetric part, not behavior.")
for r in spectra:
    print(f'L{r["layer"]:02d} H{r["head"]}',
          f'identity_cos={r["identity_alignment"]:+.6f}',
          f'symmetric_energy_share={r["symmetric_share"]:.5f}',
          f'positive_symmetric_energy_share={r["positive_symmetric_share"]:.5f}',
          f'min_eigenvalue={r["min_eig"]:.5g}',
          f'max_eigenvalue={r["max_eig"]:.5g}')
```