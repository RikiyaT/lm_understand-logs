```python
import math
import time

A,B=Checkpoint("A"),Checkpoint("B")
cfg=A.config
L,N,H,D=(
    cfg["num_hidden_layers"],cfg["hidden_size"],
    cfg["num_attention_heads"],cfg["head_dim"]
)
RANK=16
started=time.monotonic()

def get(c,l,s):
    return c.get(f"model.layers.{l}.{s}.weight").astype(np.float64)

def gain(c,l,s):
    return 1+get(c,l,s)

def sqnorm(x):
    return float(np.sum(x*x,dtype=np.float64))

def top_subspace(c):
    ev,vec=np.linalg.eigh(c)
    return vec[:,np.argsort(ev)[-RANK:][::-1]]

def mlp_subspaces(l,scaled):
    delta_cov=np.zeros((N,N))
    midpoint_cov=np.zeros((N,N))
    for name in ["gate","up"]:
        a,b=get(A,l,f"mlp.{name}_proj"),get(B,l,f"mlp.{name}_proj")
        if scaled:
            a*=gain(A,l,"pre_feedforward_layernorm")[None,:]
            b*=gain(B,l,"pre_feedforward_layernorm")[None,:]
        delta=b-a
        midpoint=(a+b)/2
        delta_cov+=(delta.T@delta)/max(sqnorm(delta),1e-300)
        midpoint_cov+=(midpoint.T@midpoint)/max(sqnorm(midpoint),1e-300)
    return top_subspace(delta_cov/2),top_subspace(midpoint_cov/2)

bases={}
for scaled in [False,True]:
    bases[scaled]=[mlp_subspaces(l,scaled) for l in range(L)]

def transport_stats(matrix,source,target):
    # Matrix products of stored-weight coefficients and weight-derived subspaces.
    # No input state is propagated through a model.
    ms=matrix@source
    tm=target.T@matrix
    block=target.T@ms
    total=max(sqnorm(matrix),1e-300)
    input_energy=max(sqnorm(ms),1e-300)
    output_energy=max(sqnorm(tm),1e-300)
    block_energy=sqnorm(block)
    conditional=block_energy/input_energy
    unconditional=output_energy/total
    return dict(
        block=block,
        joint_share=block_energy/total,
        source_capture=input_energy/total,
        target_capture=unconditional,
        conditional_target_capture=conditional,
        conditional_enrichment=conditional/max(unconditional,1e-300)
    )

print("=== ADJACENT MLP UPDATE-READING SUBSPACE OVERLAP ===")
print("Overlap is ||U_next^T U_previous||_F^2 / 16.")
print("This tests the direct residual-coordinate connection, not actual state evolution.")
print("layer_pair update_raw midpoint_raw update_scaled midpoint_scaled")
for l in range(1,L):
    values=[]
    for scaled in [False,True]:
        previous,current=bases[scaled][l-1],bases[scaled][l]
        for which in [0,1]:
            values.append(sqnorm(current[which].T@previous[which])/RANK)
    print(f"{l-1:02d}->{l:02d}",*[f"{v:.6f}" for v in values])

records=[]
for l in range(1,L):
    oa,ob=get(A,l,"self_attn.o_proj"),get(B,l,"self_attn.o_proj")
    va,vb=get(A,l,"self_attn.v_proj"),get(B,l,"self_attn.v_proj")
    for h in range(H):
        sl=slice(h*D,(h+1)*D)
        raw_a,raw_b=oa[:,sl]@va,ob[:,sl]@vb
        for scaled in [False,True]:
            if scaled:
                ma=(gain(A,l,"post_attention_layernorm")[:,None]*
                    raw_a*gain(A,l,"input_layernorm")[None,:])
                mb=(gain(B,l,"post_attention_layernorm")[:,None]*
                    raw_b*gain(B,l,"input_layernorm")[None,:])
            else:
                ma,mb=raw_a,raw_b
            source,source_ref=bases[scaled][l-1]
            target,target_ref=bases[scaled][l]

            midpoint=(ma+mb)/2
            delta=mb-ma
            changed=transport_stats(delta,source,target)
            existing=transport_stats(midpoint,source,target)
            control=transport_stats(delta,source_ref,target_ref)
            ba=target.T@ma@source
            bb=target.T@mb@source
            connector_relative=math.sqrt(
                sqnorm(bb-ba)/
                max((sqnorm(ba)+sqnorm(bb))/2,1e-300)
            )

            records.append(dict(
                layer=l,head=h,scaled=scaled,
                delta_joint=changed["joint_share"],
                midpoint_joint=existing["joint_share"],
                control_joint=control["joint_share"],
                delta_source_capture=changed["source_capture"],
                delta_target_capture=changed["target_capture"],
                conditional_enrichment=changed["conditional_enrichment"],
                connector_relative=connector_relative
            ))

def show(r):
    print(f"L{r['layer']-1:02d}MLP -> L{r['layer']:02d}H{r['head']} -> "
          f"L{r['layer']:02d}MLP "
          f"delta_joint={r['delta_joint']:.4%} "
          f"midpoint_joint={r['midpoint_joint']:.4%} "
          f"midpoint_subspace_control={r['control_joint']:.4%} "
          f"delta_source={r['delta_source_capture']:.3%} "
          f"delta_target={r['delta_target_capture']:.3%} "
          f"conditional_enrichment={r['conditional_enrichment']:.3f} "
          f"connector_rel_change={r['connector_relative']:.5f}")

print("\n=== TOP 12 GAIN-SCALED CONNECTIONS BY SHARE OF OV-DIFFERENCE ENERGY ===")
scaled_records=[r for r in records if r["scaled"]]
for r in sorted(scaled_records,key=lambda r:r["delta_joint"],reverse=True)[:12]:
    show(r)

print("\n=== TOP 8 RAW CONNECTIONS BY SHARE OF OV-DIFFERENCE ENERGY ===")
raw_records=[r for r in records if not r["scaled"]]
for r in sorted(raw_records,key=lambda r:r["delta_joint"],reverse=True)[:8]:
    show(r)

print("\n=== LAYER SUMMARY ===")
print("attention_layer mean_delta_joint_raw mean_delta_joint_scaled "
      "best_scaled_head best_scaled_delta_joint")
for l in range(1,L):
    rr=[r for r in raw_records if r["layer"]==l]
    ss=[r for r in scaled_records if r["layer"]==l]
    best=max(ss,key=lambda r:r["delta_joint"])
    print(f"{l:2d} "
          f"{np.mean([r['delta_joint'] for r in rr]):.4%} "
          f"{np.mean([r['delta_joint'] for r in ss]):.4%} "
          f"{best['head']} {best['delta_joint']:.4%}")

print("\n=== PREVIOUS ATTENTION CANDIDATES, RAW VERSUS GAIN-SCALED ===")
focus={(4,1),(8,2)}|{(l,h) for l in [11,12] for h in range(H)}
for l,h in sorted(focus):
    print(f"L{l:02d}H{h}")
    for r in records:
        if r["layer"]==l and r["head"]==h:
            print("  gain_scaled" if r["scaled"] else "  raw")
            show(r)

print(f"\nElapsed: {time.monotonic()-started:.1f}s")
print("The arrows denote open coefficient connections through a residual path and an OV map.")
print("Intervening nonlinear contributions and actual attention weights were not evaluated.")
print("No result establishes that a message uses these connections or what the subspaces mean.")
```