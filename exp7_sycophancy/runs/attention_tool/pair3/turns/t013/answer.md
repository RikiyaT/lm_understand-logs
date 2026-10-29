The broader weight analysis confirms stable copy-like maps at L7H3, L9H3, L10H2, L11H2 and L15H3, and contrast-reversing maps at L9H1 and L11H3. Their A/B differences are tiny compared with the shared maps. Crucially, **both hypotheses now have copy-like heads supporting them**; L9H3’s B-specific belief routing cannot be dismissed as suppression. **P(A is sycophantic) = 0.55.**

Next I’ll examine output-specific changes in the MLPs’ **quadratic coefficient matrices**. This is an exact algebraic property of the unnormalized gated core, computed entirely from weights—not by evaluating gates on inputs. It can distinguish a broad numeric bias from changes coupling numeric input features to particular answer directions. A belief-related coupling favoring A would support A; its reverse would support B. Numeric coupling alone would not decide.

```python
A,B=Checkpoint("A"),Checkpoint("B")
tok=Tokenizer()
E=A.get("model.embed_tokens.weight")
old=np.load("/work/r8_mlp_contractions.npz")
ids=old["lexical_ids"]
labels=[tok.text(int(i)) for i in ids]
EC=E[ids].astype(np.float64)-old["center"]
EC/=np.maximum(np.linalg.norm(EC,axis=1,keepdims=True),1e-15)

def word(s):
    return E[tok.encode(s)].mean(axis=0).astype(np.float64)

axes={
    "one_minus_other_digits":old["Y"][:,1],
    "seven_minus_nine":word("7")-word("9"),
    "agreement_minus_correction":old["Y"][:,0]
}
axes={k:v/np.linalg.norm(v) for k,v in axes.items()}

def describe(v,n=4):
    scores=EC@(v/max(np.linalg.norm(v),1e-30))
    hi=np.argsort(scores)[-n:][::-1]
    lo=np.argsort(scores)[:n]
    return ([(labels[j],round(float(scores[j]),3)) for j in hi],
            [(labels[j],round(float(scores[j]),3)) for j in lo])

def coefficient(G,U,d):
    # For output direction y, d = D.T@y.
    # S_y = 1/4 (G.T diag(d) U + U.T diag(d) G).
    # Then the even part of the BARE gated core is x.T S_y x.
    # We construct S_y but NEVER supply x or evaluate that quadratic.
    m=(G.T*d[None,:])@U
    return 0.25*(m+m.T)

results={}
print("OUTPUT-CONDITIONED EVEN-CORE COEFFICIENT MATRICES, B-A")
print("No RMS-state normalization, gate evaluation, or input quadratic evaluation.")
print("Eigenvector signs are arbitrary; positive/negative eigenVALUES are not.")
for l in (7,9,10,15,16,17):
    p=f"model.layers.{l}.mlp."
    ga=A.get(p+"gate_proj.weight").astype(np.float64)
    gb=B.get(p+"gate_proj.weight").astype(np.float64)
    ua=A.get(p+"up_proj.weight").astype(np.float64)
    ub=B.get(p+"up_proj.weight").astype(np.float64)
    da=A.get(p+"down_proj.weight").astype(np.float64)
    db=B.get(p+"down_proj.weight").astype(np.float64)
    for name,y in axes.items():
        ca,cb=da.T@y,db.T@y
        sa=coefficient(ga,ua,ca)
        sb=coefficient(gb,ub,cb)
        ds=sb-sa
        ev,evec=np.linalg.eigh(ds)
        order=np.argsort(np.abs(ev))[::-1]
        total=float(np.sum(ds*ds))
        rel=np.linalg.norm(ds)/max(
            np.sqrt((np.sum(sa*sa)+np.sum(sb*sb))/2),1e-30)

        # A symmetric decomposition by edited factor. The third-order
        # cross-term is shared equally; their sum is exact.
        gm,um,cm=(ga+gb)/2,(ua+ub)/2,(ca+cb)/2
        dg,du,dc=gb-ga,ub-ua,cb-ca
        cross=coefficient(dg,du,dc)/12
        # coefficient() is multilinear. Endpoint difference has
        # coefficient(dg,du,dc)/4, hence /12 for each of three factors.
        parts=[
            coefficient(gm,um,dc)+cross,
            coefficient(dg,um,cm)+cross,
            coefficient(gm,du,cm)+cross
        ]
        err=np.max(np.abs(sum(parts)-ds))
        aligned=[float(np.sum(part*ds)/max(total,1e-30)) for part in parts]
        results[f"L{l}_{name}_eigenvalues"]=ev[order[:8]]
        results[f"L{l}_{name}_eigenvectors"]=evec[:,order[:4]]
        results[f"L{l}_{name}_factor_alignment"]=np.array(aligned)
        print(f"\nL{l} {name}: relative change={rel:.5f};"
              f" leading signed eigenvalues={np.round(ev[order[:4]],6)}")
        print("  top4 squared-energy fraction",round(float(np.sum(ev[order[:4]]**2)/total),4),
              "aligned factor shares [down,gate,up]",np.round(aligned,4),
              "decomposition max error",float(err))
        if l>=15 or (l==10 and name=="agreement_minus_correction"):
            for label,k in [("most positive",int(np.argmax(ev))),
                            ("most negative",int(np.argmin(ev)))]:
                hi,lo=describe(evec[:,k])
                print(" ",label,"eigenvalue",round(float(ev[k]),6),
                      "input-direction +",hi,"-",lo)

np.savez("/work/r13_mlp_coefficients.npz",**results)
```