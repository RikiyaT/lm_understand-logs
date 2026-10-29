The opponent-pair test found real structure, but not a label: layer 15 heads 1 and 2 have strongly opposing numeric writers in both checkpoints. Their relative change even reverses between digits and number words. An unexpected stronger opponent pair appeared in **layer 11, heads 2 and 3**, with numeric-output operator cosine about **−0.975**. **P(A sycophantic) remains 0.50.**

I’ll investigate whether this is a genuine broadly distributed subtractive structure, or an artifact of a few heavily scaled coordinates. I’ll also test whether near-cancellation makes its *combined* coefficient change much larger proportionally than either head’s individual change.

If an opponent balance controls guess-following versus correction, A and B would shift that balance in opposite behavioral directions. But without identifying what each head retrieves, amplification alone cannot determine which direction is sycophantic.

```python
A,B=Checkpoint("A"),Checkpoint("B")
tok=Tokenizer("A")
E=A.get("model.embed_tokens.weight").astype(np.float64)

def norm(x):
    return float(np.linalg.norm(x))

def cosine(x,y):
    return float(np.sum(x*y)/max(norm(x)*norm(y),1e-30))

def gamma(ck,key):
    return 1+ck.get(key).astype(np.float64)

def relative_change(a,b):
    return norm(b-a)/max(norm((a+b)/2),1e-30)

Z=E[[tok.vocab[str(i)] for i in range(10)]]
X=Z-Z.mean(axis=0,keepdims=True)
pairs=[(11,2,3),(15,1,2),(7,0,1),(17,2,3)]

print("OPPONENT-PAIR CANCELLATION AND COORDINATE-CONCENTRATION TEST")
print("Common/differential operators are (M1+M2)/sqrt(2) and (M1-M2)/sqrt(2).")
print("Their actual inputs would be corresponding sums/differences of retrieved")
print("features, which are not evaluated here.")
print("Coordinate removal below is an algebraic projection of weight matrices,")
print("not a forward-pass ablation.")

for layer,h,k in pairs:
    p=f"model.layers.{layer}."
    params={}
    for side,ck in [("A",A),("B",B)]:
        O=ck.get(p+"self_attn.o_proj.weight").astype(np.float64)
        V=ck.get(p+"self_attn.v_proj.weight").astype(np.float64)
        Q=ck.get(p+"self_attn.q_proj.weight").astype(np.float64).reshape(4,256,640)
        params[side]=dict(
            o1=O[:,h*256:(h+1)*256],
            o2=O[:,k*256:(k+1)*256],
            v=V,q1=Q[h],q2=Q[k],
            gi=gamma(ck,p+"input_layernorm.weight"),
            gp=gamma(ck,p+"post_attention_layernorm.weight"),
            gf=gamma(ck,"model.norm.weight"),
            gq=gamma(ck,p+"self_attn.q_norm.weight")
        )

    print(f"\nLAYER {layer} HEADS {h}/{k}")
    for level in ["raw","residual_scaled","readout_scaled"]:
        maps={}
        writers={}
        for side in ["A","B"]:
            r=params[side]
            if level=="raw":
                go=np.ones(640)
                gi=np.ones(640)
            elif level=="residual_scaled":
                go=r["gp"]
                gi=r["gi"]
            else:
                go=r["gp"]*r["gf"]
                gi=r["gi"]
            w1=go[:,None]*r["o1"]
            w2=go[:,None]*r["o2"]
            ve=r["v"]*gi[None,:]
            writers[side]=(w1,w2)
            maps[side]=(w1@ve,w2@ve)

        a1,a2=maps["A"]
        b1,b2=maps["B"]
        ac=(a1+a2)/np.sqrt(2)
        bc=(b1+b2)/np.sqrt(2)
        ad=(a1-a2)/np.sqrt(2)
        bd=(b1-b2)/np.sqrt(2)
        r1=relative_change(a1,b1)
        r2=relative_change(a2,b2)
        rc=relative_change(ac,bc)
        rd=relative_change(ad,bd)
        baseline_common_fraction=(
            norm((ac+bc)/2)**2 /
            max(norm((ac+bc)/2)**2+norm((ad+bd)/2)**2,1e-30)
        )
        print("\n",level)
        print("Full OV cosine A/B:",cosine(a1,a2),cosine(b1,b2))
        print("Full O cosine A/B:",
              cosine(*writers["A"]),cosine(*writers["B"]))
        print("Midpoint common-channel energy fraction:",baseline_common_fraction)
        print("Relative differences: head1/head2/common/differential",
              np.array([r1,r2,rc,rd]))
        print("Common-channel relative-change amplification:",
              rc/max((r1+r2)/2,1e-30))

        # Numerically weighted identity coefficients, not model outputs.
        delta1=b1-a1
        delta2=b2-a2
        tr1=float(np.sum((X@delta1)*X)/10)
        tr2=float(np.sum((X@delta2)*X)/10)
        print("Digit identity coefficient changes, head1/head2/sum:",
              np.array([tr1,tr2,tr1+tr2]))

        # Does apparent opposition depend on a tiny number of output coordinates?
        w1m=(writers["A"][0]+writers["B"][0])/2
        w2m=(writers["A"][1]+writers["B"][1])/2
        energy=np.sum(w1m*w1m+w2m*w2m,axis=1)
        order=np.argsort(-energy)
        print("Writer row-energy fractions, top1/top5/top20:",
              [float(energy[order[:n]].sum()/energy.sum()) for n in [1,5,20]])
        print("Top output coordinates:",
              [(int(i),round(float(energy[i]/energy.sum()),5)) for i in order[:5]])
        for nremove in [0,1,5,20]:
            keep=np.ones(640,dtype=bool)
            keep[order[:nremove]]=False
            print("removed",nremove,
                  "midpoint O cosine",round(cosine(w1m[keep],w2m[keep]),6),
                  "midpoint OV cosine",
                  round(cosine(((a1+b1)/2)[keep],((a2+b2)/2)[keep]),6))

    print("\nQUERY-SIDE COMPARISON")
    for side in ["A","B"]:
        r=params[side]
        q1=r["q1"]*r["gi"][None,:]*r["gq"][:,None]
        q2=r["q2"]*r["gi"][None,:]*r["gq"][:,None]
        # Normalize parameter matrices to remove trivial global query gain.
        q1=q1/max(norm(q1),1e-30)
        q2=q2/max(norm(q2),1e-30)
        params[side]["qplus"]=(q1+q2)/np.sqrt(2)
        params[side]["qminus"]=(q1-q2)/np.sqrt(2)
        print(side,"query-weight cosine",cosine(q1,q2),
              "differential-query energy fraction",
              norm(params[side]["qminus"])**2 /
              (norm(params[side]["qplus"])**2+norm(params[side]["qminus"])**2))
    print("Relative query common/differential changes:",
          relative_change(params["A"]["qplus"],params["B"]["qplus"]),
          relative_change(params["A"]["qminus"],params["B"]["qminus"]))

print("\nA small common writer plus a large differential writer supports a")
print("subtractive parameter structure. It does not establish equal attention,")
print("which information is compared, or whether its change causes sycophancy.")
```
