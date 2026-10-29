```python
import math
import time

A,B=Checkpoint("A"),Checkpoint("B")
cfg=A.config
L,N,H=cfg["num_hidden_layers"],cfg["hidden_size"],cfg["num_attention_heads"]
started=time.monotonic()
focus={1,4,10,11,12,14,17}
names=["q","k","v","gate","up"]

def get(c,l,s):
    return c.get(f"model.layers.{l}.{s}.weight").astype(np.float64)

def gain(c,l,s):
    return 1+get(c,l,s)

def readers(c,l,scaled):
    matrices={
        "q":get(c,l,"self_attn.q_proj"),
        "k":get(c,l,"self_attn.k_proj"),
        "v":get(c,l,"self_attn.v_proj"),
        "gate":get(c,l,"mlp.gate_proj"),
        "up":get(c,l,"mlp.up_proj")
    }
    if scaled:
        gi=gain(c,l,"input_layernorm")
        gm=gain(c,l,"pre_feedforward_layernorm")
        matrices["q"]*=gi[None,:]
        matrices["q"]*=np.tile(gain(c,l,"self_attn.q_norm"),H)[:,None]
        matrices["k"]*=gi[None,:]
        matrices["k"]*=gain(c,l,"self_attn.k_norm")[:,None]
        matrices["v"]*=gi[None,:]
        matrices["gate"]*=gm[None,:]
        matrices["up"]*=gm[None,:]
    return matrices

def eigensystem(c):
    ev,vec=np.linalg.eigh(c)
    order=np.argsort(ev)[::-1]
    return np.maximum(ev[order],0),vec[:,order]

def capture(c,subspace):
    return float(np.sum(subspace*(c@subspace))/
                 max(float(np.trace(c)),1e-300))

records=[]
for l in range(L):
    for scaled in ([False,True] if l in focus else [False]):
        wa,wb=readers(A,l,scaled),readers(B,l,scaled)
        delta={n:wb[n]-wa[n] for n in names}
        midpoint={n:(wb[n]+wa[n])/2 for n in names}
        cov={n:delta[n].T@delta[n] for n in names}
        refcov={n:midpoint[n].T@midpoint[n] for n in names}
        systems={n:eigensystem(cov[n]) for n in names}
        cross={}
        for source in names:
            for target in names:
                v=systems[source][1][:,:16]
                target_ev=systems[target][0]
                best=float(target_ev[:16].sum()/
                           max(float(target_ev.sum()),1e-300))
                cap=capture(cov[target],v)
                ref=capture(refcov[target],v)
                cross[(source,target)]=dict(
                    capture=cap,best=best,
                    fraction_of_best=cap/max(best,1e-300),
                    midpoint_capture=ref,
                    enrichment=cap/max(ref,1e-300)
                )

        groups={}
        for label,group in [
            ("mlp",["gate","up"]),
            ("attention",["q","k","v"]),
            ("all",names)
        ]:
            joint=sum(cov[n]/max(float(np.trace(cov[n])),1e-300)
                      for n in group)/len(group)
            ev,vec=eigensystem(joint)
            groups[label]=dict(
                top4=float(ev[:4].sum()/max(float(ev.sum()),1e-300)),
                top16=float(ev[:16].sum()/max(float(ev.sum()),1e-300)),
                vec=vec[:,:16]
            )

        # Which rows' changes participate in the joint MLP reading-direction
        # subspace? This is a product of coefficient matrices, not an input run.
        v4=groups["mlp"]["vec"][:,:4]
        unit_score=np.zeros(delta["gate"].shape[0])
        for n in ["gate","up"]:
            projected=delta[n]@v4
            unit_score+=np.sum(projected*projected,axis=1)/max(
                float(np.sum(delta[n]*delta[n])),1e-300
            )
        ix=np.argsort(unit_score)[-6:][::-1]
        records.append(dict(
            layer=l,scaled=scaled,cross=cross,
            groups={label:{k:v for k,v in z.items() if k!="vec"}
                    for label,z in groups.items()},
            top_units=ix.tolist(),
            top_unit_shares=[
                float(unit_score[i]/max(float(unit_score.sum()),1e-300))
                for i in ix
            ]
        ))

print("=== RAW-WEIGHT CROSS-PROJECTION DIRECTION CAPTURE ===")
print("Each pair: fraction of target difference energy captured by the source's top16")
print("directions / fraction of the target's own optimal top16 capture.")
print("layer gate->up up->gate q->k k->q q->v joint_MLP_top16 joint_attention_top16")
for r in records:
    if r["scaled"]:
        continue
    c=r["cross"]
    pairs=[("gate","up"),("up","gate"),("q","k"),("k","q"),("q","v")]
    entries=[f"{c[p]['capture']:.3%}/{c[p]['fraction_of_best']:.3f}" for p in pairs]
    print(f"{r['layer']:2d}",*entries,
          f"{r['groups']['mlp']['top16']:.3%}",
          f"{r['groups']['attention']['top16']:.3%}")

print("\n=== FOCUSED BASELINE AND GAIN CHECKS ===")
print("Capture in midpoint weights is a geometry control, not a statistical null.")
for l in sorted(focus):
    for r in records:
        if r["layer"]!=l:
            continue
        label="gain_scaled" if r["scaled"] else "raw"
        print(f"L{l:02d} {label}")
        for pair in [("gate","up"),("up","gate"),("q","k"),("k","q"),("gate","q")]:
            z=r["cross"][pair]
            print(f"  {pair[0]:4s}->{pair[1]:4s}: "
                  f"delta_capture={z['capture']:.3%} "
                  f"target_best={z['best']:.3%} "
                  f"fraction_of_best={z['fraction_of_best']:.4f} "
                  f"midpoint_capture={z['midpoint_capture']:.3%} "
                  f"delta/midpoint_enrichment={z['enrichment']:.3f}")
        print("  joint top4/top16:",
              {g:(round(z["top4"],5),round(z["top16"],5))
               for g,z in r["groups"].items()})

print("\n=== UNITS MOST REPRESENTED IN JOINT TOP4 MLP UPDATE-READING DIRECTIONS ===")
print("These rank changes in reading coefficients, not output influence or behavioural importance.")
for r in records:
    if r["scaled"] or r["layer"] in focus:
        label="gain_scaled" if r["scaled"] else "raw"
        print(f"L{r['layer']:02d} {label:11s} "
              f"units={r['top_units']} "
              f"shares={np.round(r['top_unit_shares'],5)}")

print(f"\nElapsed: {time.monotonic()-started:.1f}s")
print("All subspaces were extracted from weight-difference Gram matrices.")
print("No input, hidden state, activation, normalization denominator, or model output was evaluated.")
print("Shared update-reading directions do not identify their semantic meaning.")
```