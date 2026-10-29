```python
import math
import time

A, B = Checkpoint("A"), Checkpoint("B")
cfg = A.config
L, H, D = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["head_dim"]
started = time.monotonic()

def norm(x):
    return math.sqrt(float(np.sum(x*x, dtype=np.float64)))

def load(c, l):
    p = f"model.layers.{l}."
    def w(s):
        return c.get(p+s+".weight").astype(np.float64)
    gi = 1+w("input_layernorm")
    ga = 1+w("post_attention_layernorm")
    gm = 1+w("pre_feedforward_layernorm")
    gf = 1+w("post_feedforward_layernorm")
    qg, kg = 1+w("self_attn.q_norm"), 1+w("self_attn.k_norm")
    q = w("self_attn.q_proj")*gi[None,:]
    q *= np.tile(qg,H)[:,None]
    k = kg[:,None]*w("self_attn.k_proj")*gi[None,:]
    v = w("self_attn.v_proj")*gi[None,:]
    o = ga[:,None]*w("self_attn.o_proj")
    ov = [o[:,h*D:(h+1)*D] @ v for h in range(H)]
    return dict(
        q=q, k=k, ov=ov,
        gate=w("mlp.gate_proj")*gm[None,:],
        up=w("mlp.up_proj")*gm[None,:],
        down=gf[:,None]*w("mlp.down_proj")
    )

records = []
def analyze_edge(family, layer, head, reader_name, ra, rb, wa, wb):
    # Exact open linear-path product of stored weights and learned gains.
    # Unknown attention weights, gates, and RMS denominators are NOT supplied.
    pa, pb = ra @ wa, rb @ wb
    delta = pb-pa
    nd = norm(delta)
    na, nb = norm(pa), norm(pb)
    reference = math.sqrt((na*na+nb*nb)/2)
    nra, nrb, nwa, nwb = norm(ra), norm(rb), norm(wa), norm(wb)
    scale = math.sqrt(((nra*nwa)**2+(nrb*nwb)**2)/2)

    from_reader = (rb-ra) @ ((wa+wb)/2)
    from_writer = delta-from_reader
    nr, nw = norm(from_reader), norm(from_writer)
    contribution_cos = float(np.sum(from_reader*from_writer)) / max(nr*nw,1e-30)

    # For same-layer edges, rows identify MLP reading units.
    # For MLP-down edges, columns identify MLP writing units.
    if family == "same_attention_to_mlp":
        unit_energy = np.sum(delta*delta,axis=1)
    elif family == "next_mlp_to_attention":
        unit_energy = np.sum(delta*delta,axis=0)
    else:
        unit_energy = None
    if unit_energy is not None:
        ix = np.argsort(unit_energy)[-4:][::-1]
        top_units = [(int(i),float(unit_energy[i]/max(unit_energy.sum(),1e-30)))
                     for i in ix]
    else:
        top_units = None

    records.append(dict(
        family=family, layer=layer, head=head, reader=reader_name,
        relative=nd/max(reference,1e-30),
        normalized_delta=nd/max(scale,1e-30),
        coupling_a=na/max(nra*nwa,1e-30),
        coupling_b=nb/max(nrb*nwb,1e-30),
        reader_ratio=nr/max(nd,1e-30),
        writer_ratio=nw/max(nd,1e-30),
        contribution_cos=contribution_cos,
        top_units=top_units
    ))

current_a, current_b = load(A,0), load(B,0)
for l in range(L):
    for h in range(H):
        for reader in ["gate","up"]:
            analyze_edge(
                "same_attention_to_mlp",l,h,reader,
                current_a[reader],current_b[reader],
                current_a["ov"][h],current_b["ov"][h]
            )

    if l+1 < L:
        next_a, next_b = load(A,l+1), load(B,l+1)
        for reader in ["q","k"]:
            analyze_edge(
                "next_mlp_to_attention",l,None,reader,
                next_a[reader],next_b[reader],
                current_a["down"],current_b["down"]
            )
            for h in range(H):
                analyze_edge(
                    "next_attention_to_attention",l,h,reader,
                    next_a[reader],next_b[reader],
                    current_a["ov"][h],current_b["ov"][h]
                )
        current_a,current_b = next_a,next_b

    if l in [5,11,17]:
        print(f"Processed through L{l}; elapsed {time.monotonic()-started:.1f}s",
              flush=True)

def description(r):
    if r["family"]=="same_attention_to_mlp":
        return f"L{r['layer']:02d}H{r['head']} OV -> L{r['layer']:02d} MLP/{r['reader']}"
    if r["family"]=="next_mlp_to_attention":
        return f"L{r['layer']:02d} MLP/down -> L{r['layer']+1:02d} attn/{r['reader']}"
    return f"L{r['layer']:02d}H{r['head']} OV -> L{r['layer']+1:02d} attn/{r['reader']}"

def print_record(r,detail=False):
    print(f"{description(r):37s} "
          f"relative={r['relative']:.5f} "
          f"normalized_delta={r['normalized_delta']:.6f} "
          f"coupling_A={r['coupling_a']:.4f} coupling_B={r['coupling_b']:.4f} "
          f"reader_ratio={r['reader_ratio']:.3f} writer_ratio={r['writer_ratio']:.3f} "
          f"cross_cos={r['contribution_cos']:+.3f}")
    if detail and r["top_units"] is not None:
        print("  Leading unit shares of this path's coefficient-difference energy:",
              [(i,round(share,5)) for i,share in r["top_units"]])

for family in [
    "same_attention_to_mlp",
    "next_mlp_to_attention",
    "next_attention_to_attention"
]:
    rr = [r for r in records if r["family"]==family]
    print("\n===",family.upper(),"===")
    print("Top 8 by relative composed-path change:")
    for r in sorted(rr,key=lambda r:r["relative"],reverse=True)[:8]:
        print_record(r)
    print("Top 4 by factor-norm-normalized composed-path change:")
    for r in sorted(rr,key=lambda r:r["normalized_delta"],reverse=True)[:4]:
        print_record(r)

print("\n=== FOCUSED CHECK: L10 MLP -> L11 ATTENTION ===")
for r in records:
    if r["family"]=="next_mlp_to_attention" and r["layer"]==10:
        print_record(r,detail=True)

print("\n=== FOCUSED CHECK: L11 ATTENTION -> L12 QUERY/KEY READERS ===")
for r in records:
    if r["family"]=="next_attention_to_attention" and r["layer"]==11:
        print_record(r)

print("\n=== FOCUSED CHECK: FINAL-LAYER ATTENTION -> MLP ===")
for r in records:
    if r["family"]=="same_attention_to_mlp" and r["layer"]==17:
        print_record(r,detail=True)

print(f"\nElapsed: {time.monotonic()-started:.1f}s")
print("These are open-path coefficient products, not forward passes or activations.")
print("Factor-norm normalization is a matrix statistic, not hidden-state normalization.")
print("A nonzero or large path coefficient does not establish use on any message.")
```