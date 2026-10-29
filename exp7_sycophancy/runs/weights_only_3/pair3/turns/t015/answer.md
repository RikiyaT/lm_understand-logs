```python
import math
import time

A,B=Checkpoint("A"),Checkpoint("B")
cfg=A.config
L,N,H=cfg["num_hidden_layers"],cfg["hidden_size"],cfg["num_attention_heads"]
started=time.monotonic()

targets={
    0:[1405,1130],
    1:[1385,372,1976,1725],
    4:[20,1044,13],
    6:[1824],
    7:[1455],
    8:[1394,1379,188],
    10:[1363,1564,1529],
    14:[641]
}

def rows(x,y):
    return np.einsum("ij,ij->i",x,y,optimize=False)

def get(c,l,s):
    return c.get(f"model.layers.{l}.{s}.weight").astype(np.float64)

def gain(c,l,s):
    return 1+get(c,l,s)

def reader_matrices(c,l):
    gi=gain(c,l,"input_layernorm")
    gm=gain(c,l,"pre_feedforward_layernorm")
    q=get(c,l,"self_attn.q_proj")*gi[None,:]
    q*=np.tile(gain(c,l,"self_attn.q_norm"),H)[:,None]
    k=get(c,l,"self_attn.k_proj")*gi[None,:]
    k*=gain(c,l,"self_attn.k_norm")[:,None]
    v=get(c,l,"self_attn.v_proj")*gi[None,:]
    g=get(c,l,"mlp.gate_proj")*gm[None,:]
    u=get(c,l,"mlp.up_proj")*gm[None,:]
    return [q,k,v,g,u]

# Each covariance has trace one. This is a balanced geometric average over
# reader matrices, NOT a hidden-state distribution or actual attention pattern.
reader_cov=[]
for l in range(L):
    c=np.zeros((N,N),dtype=np.float64)
    for checkpoint in [A,B]:
        for r in reader_matrices(checkpoint,l):
            c+=(r.T@r)/max(float(np.sum(r*r)),1e-300)
    reader_cov.append(c/10)

future_cov=[None]*L
running=np.zeros((N,N),dtype=np.float64)
count=0
for l in range(L-1,-1,-1):
    if count:
        future_cov[l]=running/count
    running=running+reader_cov[l]
    count+=1

def factors(c,l):
    d=get(c,l,"mlp.down_proj").T*gain(c,l,"post_feedforward_layernorm")[None,:]
    pre=gain(c,l,"pre_feedforward_layernorm")
    g=get(c,l,"mlp.gate_proj")*pre[None,:]
    u=get(c,l,"mlp.up_proj")*pre[None,:]
    return d,g,u

def input_coefficient_inner(x,y):
    _,gx,ux=x
    _,gy,uy=y
    gg,uu=rows(gx,gy),rows(ux,uy)
    cross=rows(gx,uy)*rows(ux,gy)
    return {
        "base":0.5*(gg*uu+cross),
        "gate_doubled":gg*(gg*uu+2*cross)/3
    }

def coefficient_energy(xa,xb,saa,sbb,sab,cov=None):
    da,db=xa[0],xb[0]
    if cov is None:
        daa,dbb,dab=rows(da,da),rows(db,db),rows(da,db)
    else:
        da_c,db_c=da@cov,db@cov
        daa,dbb,dab=rows(da_c,da),rows(db_c,db),rows(da_c,db)
    ref=0.5*(daa*saa+dbb*sbb)
    delta=daa*saa+dbb*sbb-2*dab*sab
    return np.maximum(delta,0),np.maximum(ref,0)

print("=== DOWNSTREAM READER COMPATIBILITY OF COEFFICIENT DIFFERENCES ===")
print("Compatibility = 640 * reader-weighted energy / ordinary energy.")
print("A value of 1 is the isotropic output-coordinate benchmark, not a data baseline.")
print("Reader matrices include learned diagonal gains but no input-dependent RMS denominators.")
print("Candidate percentiles compare units within the same layer, not behavioural effects.")

for l,units in targets.items():
    xa,xb=factors(A,l),factors(B,l)
    aa=input_coefficient_inner(xa,xa)
    bb=input_coefficient_inner(xb,xb)
    ab=input_coefficient_inner(xa,xb)
    da,db=xa[0],xb[0]
    immediate=reader_cov[l+1]
    future=future_cov[l]
    local={}

    for mode in ["base","gate_doubled"]:
        de,ref=coefficient_energy(xa,xb,aa[mode],bb[mode],ab[mode])
        di,ri=coefficient_energy(xa,xb,aa[mode],bb[mode],ab[mode],immediate)
        df,rf=coefficient_energy(xa,xb,aa[mode],bb[mode],ab[mode],future)
        local[mode]=dict(
            immediate=N*di/np.maximum(de,1e-300),
            future=N*df/np.maximum(de,1e-300),
            reference_immediate=N*ri/np.maximum(ref,1e-300),
            reference_future=N*rf/np.maximum(ref,1e-300)
        )

    # Resolve each unit's gate-doubled coefficient difference by output coordinate.
    # Orthogonal output coordinates make these nonnegative energies additive.
    output_energy=(
        da*da*aa["gate_doubled"][:,None]
        +db*db*bb["gate_doubled"][:,None]
        -2*da*db*ab["gate_doubled"][:,None]
    )
    output_energy=np.maximum(output_energy,0)
    gain_mid=0.5*(
        gain(A,l,"post_feedforward_layernorm")+
        gain(B,l,"post_feedforward_layernorm")
    )

    print(f"\nLAYER {l:02d}")
    for i in units:
        z=local["gate_doubled"]
        percentile=100*float(np.mean(z["future"]<=z["future"][i]))
        e=output_energy[i]
        order=np.argsort(e)[::-1]
        denom=max(float(e.sum()),1e-300)
        coords=order[:3]
        print(f"  U{i:04d} "
              f"next_reader={z['immediate'][i]:.5g} "
              f"all_later_readers={z['future'][i]:.5g} "
              f"reference_next={z['reference_immediate'][i]:.5g} "
              f"reference_later={z['reference_future'][i]:.5g} "
              f"later_within_layer_percentile={percentile:.1f}")
        print(f"    base_coefficient_compatibility: "
              f"next={local['base']['immediate'][i]:.5g} "
              f"later={local['base']['future'][i]:.5g}")
        print("    leading output coordinates (index, energy_share, midpoint_post_gain):",
              [(int(j),round(float(e[j]/denom),5),round(float(gain_mid[j]),5))
               for j in coords],
              "top10_share=",round(float(e[order[:10]].sum()/denom),5))

    if l in [1,4,10]:
        # Compare changed writer columns directly; these are weight vectors,
        # not hidden states passed through a model.
        dd=(db-da)[units]
        lengths=np.sqrt(rows(dd,dd))
        cosine=(dd@dd.T)/np.maximum(lengths[:,None]*lengths[None,:],1e-300)
        print("  Changed-writer direction cosine matrix; unit order:",units)
        print(cosine)

print(f"\nElapsed: {time.monotonic()-started:.1f}s")
print("No hidden states, nonlinearities, attention patterns, or model outputs were evaluated.")
print("Reader compatibility is an exact weight-coefficient statistic under the stated metric.")
print("It does not establish use of these paths, and excludes the unavailable final readout.")
```