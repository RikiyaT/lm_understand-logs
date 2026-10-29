The previous checks established structured weight changes, but did not test whether the opposing attention-head changes **cancel or redistribute the same shared value features**. **My current probability that A is sycophantic remains 0.35.**

I’ll test that new hypothesis using Gemma’s shared key/value architecture. If individual copying changes mostly cancel across heads, ranking those heads independently would be misleading. If B instead strengthens the common numeral-preserving operator, that would support B under the copying hypothesis; strengthening in A would support A. These sums are weight operators—not an assumption that the heads actually attend to the same positions.

```python
A, B, tok = Checkpoint("A"), Checkpoint("B"), Tokenizer()
E = A.get("model.embed_tokens.weight")
ids = [tok.encode(s)[0] for s in "0123456789"]

Zd = E[ids].astype(np.float64)
Zd -= Zd.mean(axis=0)
_, s, vt = np.linalg.svd(Zd, full_matrices=False)
Qd = vt[s > s[0]*1e-6].T

Z = E[ids].astype(np.float64)
Z /= np.linalg.norm(Z, axis=1, keepdims=True)
Z -= Z.mean(axis=0)
denom = float(np.sum(Z*Z))
del E

final_a = 1 + A.get("model.norm.weight")
final_b = 1 + B.get("model.norm.weight")

# Unique head-sign patterns, modulo changing all four signs together.
signs = np.array([
    [1, a, b, c]
    for a in (-1, 1)
    for b in (-1, 1)
    for c in (-1, 1)
], dtype=np.float64)

def head_geometry(P):
    # P: head x numerical-output-coordinate x residual-input-coordinate.
    F = P.reshape(4, -1)
    G = F @ F.T
    energy = max(float(np.trace(G)), 1e-30)
    common_fraction = float(G.sum()) / (4*energy)
    norm = np.sqrt(np.maximum(np.diag(G), 1e-30))
    correlations = G / (norm[:, None]*norm[None, :])
    sign_control = np.einsum("si,ij,sj->s", signs, G, signs)/(4*energy)
    return energy, common_fraction, correlations, sign_control

records = []
print("=== NEW HYPOTHESIS: HEAD REDISTRIBUTION / CANCELLATION ===")
print("All heads in a layer share V; each has its own O slice.")
print("Numerical kernel: P_h = Qdigit^T (O_h V).")
print("Head-common mode = (P_0 + P_1 + P_2 + P_3)/2.")
print("Common-mode fraction is 0 for exact cancellation, .25 for uncorrelated")
print("equal-energy changes, and 1 for identical changes across heads.")
print("None of these operators includes attention weights or a model input.")

for l in range(18):
    p = f"model.layers.{l}."
    va = A.get(p+"self_attn.v_proj.weight").astype(np.float64)
    vb = B.get(p+"self_attn.v_proj.weight").astype(np.float64)
    oa = A.get(p+"self_attn.o_proj.weight").astype(np.float64)
    ob = B.get(p+"self_attn.o_proj.weight").astype(np.float64)
    ia = 1 + A.get(p+"input_layernorm.weight")
    ib = 1 + B.get(p+"input_layernorm.weight")
    pa = final_a*(1 + A.get(p+"post_attention_layernorm.weight"))
    pb = final_b*(1 + B.get(p+"post_attention_layernorm.weight"))

    for mode in ("raw", "diag"):
        Ra = va if mode == "raw" else va*ia[None, :]
        Rb = vb if mode == "raw" else vb*ib[None, :]
        Rm, dR = (Ra+Rb)*.5, Rb-Ra
        totals, midpoints, oparts, vparts, copy = [], [], [], [], []

        for h in range(4):
            sl = slice(h*256, (h+1)*256)
            La, Lb = oa[:, sl], ob[:, sl]
            if mode == "diag":
                La, Lb = pa[:, None]*La, pb[:, None]*Lb
            Lm, dL = (La+Lb)*.5, Lb-La

            # Exact midpoint product decomposition, all on weights:
            # Delta(O V) = DeltaO Vmid + Omid DeltaV.
            po = (Qd.T @ dL) @ Rm
            pv = (Qd.T @ Lm) @ dR
            pm = .5*((Qd.T @ La) @ Ra + (Qd.T @ Lb) @ Rb)
            totals.append(po+pv)
            midpoints.append(pm)
            oparts.append(po)
            vparts.append(pv)

            # Independent identity-alignment statistic for each head.
            ca = float(np.sum((Z@La)*(Z@Ra.T))/denom)
            cb = float(np.sum((Z@Lb)*(Z@Rb.T))/denom)
            copy.append(cb-ca)

        P = np.stack(totals)
        M = np.stack(midpoints)
        PO, PV = np.stack(oparts), np.stack(vparts)
        energy, common, corr, controls = head_geometry(P)
        _, base_common, _, _ = head_geometry(M)

        # Signed additive attribution of the stacked numerical kernel change.
        oshare = float(np.sum(PO*P))/energy
        vshare = float(np.sum(PV*P))/energy
        copy = np.array(copy)
        cancellation = 1-abs(float(copy.sum()))/max(float(np.abs(copy).sum()),1e-30)

        records.append(dict(
            layer=l, mode=mode, energy=energy, common=common,
            base_common=base_common, corr=corr, controls=controls,
            oshare=oshare, vshare=vshare, copy=copy,
            cancellation=cancellation,
            common_o=float(np.sum(PO.sum(axis=0)*P.sum(axis=0))),
            common_v=float(np.sum(PV.sum(axis=0)*P.sum(axis=0))),
        ))

print("\n=== ALL-LAYER COMMON-MODE AND SHARED-VALUE SUMMARY ===")
print("Copy cancellation concerns the signed identity statistic only.")
print("It is distinct from cancellation of the entire numerical kernel.")
print(" L mode  delta_common  base_common  shared_V_share"
      "   summed_identity_delta  identity_cancellation")
for r in records:
    print(f"{r['layer']:2d} {r['mode']:4s} "
          f"{r['common']:12.5f} {r['base_common']:12.5f} "
          f"{r['vshare']:15.5f} {r['copy'].sum():+23.7g} "
          f"{r['cancellation']:22.5f}")

print("\n=== DETAILED HEAD RELATIONSHIPS FOR CANDIDATE BLOCKS ===")
for r in records:
    if r["layer"] not in (1, 15, 16, 17):
        continue
    print(f"\nL{r['layer']} {r['mode']}")
    print("Head identity deltas 0..3:", r["copy"])
    print("Numerical-kernel change correlations between heads:")
    print(np.round(r["corr"], 4))
    print("Output-slice / shared-value attribution:",
          round(r["oshare"], 5), round(r["vshare"], 5))
    print("Actual common-mode fraction:", round(r["common"], 5))
    print("Common-mode fractions under head-sign controls:",
          np.round(np.sort(r["controls"]), 5))
    total = r["common_o"]+r["common_v"]
    if abs(total) > 1e-30:
        print("Within common mode: output / shared-value attribution:",
              round(r["common_o"]/total, 5),
              round(r["common_v"]/total, 5))

print("\n=== BLOCK RANKING BY NUMERICAL WEIGHT-KERNEL CHANGE ===")
print("This is a geometric candidate ranking, not causal importance.")
for mode in ("raw", "diag"):
    rr = sorted([r for r in records if r["mode"] == mode],
                key=lambda r:r["energy"], reverse=True)
    print(mode, [
        (r["layer"], round(np.sqrt(r["energy"]), 6),
         round(r["common"], 4), round(r["vshare"], 4))
        for r in rr[:8]
    ])
```