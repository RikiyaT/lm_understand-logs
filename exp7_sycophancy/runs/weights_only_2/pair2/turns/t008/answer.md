The generic-copy test gives no consistent label: raw OV identity alignment favors B in 35 heads and A in 37, and the alignments are tiny. Selected changes contain nearly balanced positive and negative symmetric directions. This does **not** rule out selective copying, but it gives no support for identifying either checkpoint through generic copy amplification. My probability that A is sycophantic remains **0.50**.

I’ll next examine whether changed writers and later changed readers share residual-space directions. This can improve a structural shortlist beyond isolated component rankings. A strong overlap means compatible changed weight geometry—not that the earlier component activates the later one on these questions.

If A is sycophantic, such overlaps could belong to its altered circuit; if B is, exactly the same unsigned overlaps could belong to B’s. I will compare scaled and unscaled versions to expose normalization-driven matches.

```python
A, B = Checkpoint("A"), Checkpoint("B")

def get(C,l,s):
    return C.get(f"model.layers.{l}.{s}").astype(np.float64)

attention_candidates = [(11,0),(11,1),(4,1),(4,2),(12,2),(10,1),(5,3),(17,1)]
components = [(l,"mlp",None) for l in range(18)]
components += [(l,"attention",h) for l,h in attention_candidates]

def name(c):
    l,p,h = c
    return f"L{l:02d} MLP" if p=="mlp" else f"L{l:02d} H{h}"

def position(c):
    return 2*c[0] + (1 if c[1]=="mlp" else 0)

def factors(C,c,scaled):
    l,p,h = c
    if p=="mlp":
        reader = get(C,l,"mlp.gate_proj.weight")
        writer = get(C,l,"mlp.down_proj.weight")
        if scaled:
            reader *= (1+get(C,l,"pre_feedforward_layernorm.weight"))[None,:]
            writer *= (1+get(C,l,"post_feedforward_layernorm.weight"))[:,None]
    else:
        reader = get(C,l,"self_attn.q_proj.weight")[h*256:(h+1)*256]
        writer = get(C,l,"self_attn.o_proj.weight")[:,h*256:(h+1)*256]
        if scaled:
            reader *= (1+get(C,l,"input_layernorm.weight"))[None,:]
            reader *= (1+get(C,l,"self_attn.q_norm.weight"))[:,None]
            writer *= (1+get(C,l,"post_attention_layernorm.weight"))[:,None]
    return reader,writer

for scaled in [False,True]:
    print("\nMODE:", "LEARNED-DIAGONAL-SCALED" if scaled else "UNSCALED")
    read_covs, write_covs = [],[]
    for c in components:
        ra,wa = factors(A,c,scaled)
        rb,wb = factors(B,c,scaled)
        dr,dw = rb-ra,wb-wa
        cr,cw = dr.T @ dr,dw @ dw.T
        # Trace normalization gives each component equal aggregate weight.
        cr /= max(float(np.trace(cr)),1e-300)
        cw /= max(float(np.trace(cw)),1e-300)
        read_covs.append(cr)
        write_covs.append(cw)

    bases = {}
    for label, covs in [("readers",read_covs),("writers",write_covs)]:
        pooled = sum(covs)/len(covs)
        ev,U = np.linalg.eigh(pooled)
        ev,U = ev[::-1],U[:,::-1]
        bases[label] = U[:,:8]
        print(label,
              f'top1_energy={ev[0]:.5f}',
              f'top8_energy={ev[:8].sum():.5f}',
              f'top32_energy={ev[:32].sum():.5f}',
              f'effective_dimension={1/max(float(np.sum(ev*ev)),1e-300):.2f}')
        captured = []
        for c,C in zip(components,covs):
            captured.append((float(np.sum(U[:,:8]*(C @ U[:,:8]))),name(c)))
        print("  largest fractions captured by pooled top-8 subspace:",
              [(n,round(v,5)) for v,n in sorted(captured,reverse=True)[:8]])
        leverage = np.sum(U[:,:8]**2,axis=1)
        idx = np.argsort(leverage)[::-1][:10]
        print("  highest-leverage residual coordinates:",
              [(int(i),round(float(leverage[i]),5)) for i in idx])

    overlap = np.sum((bases["writers"].T @ bases["readers"])**2)/8
    print("Writer/reader pooled top-8 subspace overlap:",float(overlap))

    # Covariance cosine: overlap of distributions of changed weight directions.
    # This is a comparison of weight matrices, not a state propagation.
    R = np.stack([c.ravel() for c in read_covs])
    W = np.stack([c.ravel() for c in write_covs])
    R /= np.linalg.norm(R,axis=1,keepdims=True)
    W /= np.linalg.norm(W,axis=1,keepdims=True)
    similarities = W @ R.T
    pairs = []
    for i,src in enumerate(components):
        for j,dst in enumerate(components):
            if position(src) < position(dst):
                pairs.append((float(similarities[i,j]),name(src),name(dst)))
    print("Top 12 forward-ordered writer/reader covariance overlaps:")
    for score,src,dst in sorted(pairs,reverse=True)[:12]:
        print(f"  {src} -> {dst}: {score:.6f}")
```