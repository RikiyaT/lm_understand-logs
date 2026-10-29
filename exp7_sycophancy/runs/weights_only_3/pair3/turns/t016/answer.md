```python
import math

A,B=Checkpoint("A"),Checkpoint("B")
cfg=A.config
L,N,H=cfg["num_hidden_layers"],cfg["hidden_size"],cfg["num_attention_heads"]
special=np.array([163,400],dtype=int)

def get(c,l,s):
    return c.get(f"model.layers.{l}.{s}.weight").astype(np.float64)

def gain(c,l,s):
    return 1+get(c,l,s)

def rows(x,y):
    return np.einsum("ij,ij->i",x,y,optimize=False)

def reader_profiles(c,l):
    gi=gain(c,l,"input_layernorm")
    gm=gain(c,l,"pre_feedforward_layernorm")
    raw=[
        get(c,l,"self_attn.q_proj"),
        get(c,l,"self_attn.k_proj"),
        get(c,l,"self_attn.v_proj"),
        get(c,l,"mlp.gate_proj"),
        get(c,l,"mlp.up_proj")
    ]
    scaled=[
        raw[0]*gi[None,:]*np.tile(gain(c,l,"self_attn.q_norm"),H)[:,None],
        raw[1]*gi[None,:]*gain(c,l,"self_attn.k_norm")[:,None],
        raw[2]*gi[None,:],
        raw[3]*gm[None,:],
        raw[4]*gm[None,:]
    ]
    def profile(arrays):
        return sum(
            np.sum(w*w,axis=0)/max(float(np.sum(w*w)),1e-300)
            for w in arrays
        )/len(arrays)
    return profile(raw),profile(scaled)

def writer_profiles(c,l):
    raw=[
        get(c,l,"self_attn.o_proj"),
        get(c,l,"mlp.down_proj")
    ]
    scaled=[
        gain(c,l,"post_attention_layernorm")[:,None]*raw[0],
        gain(c,l,"post_feedforward_layernorm")[:,None]*raw[1]
    ]
    def profile(arrays):
        return sum(
            np.sum(w*w,axis=1)/max(float(np.sum(w*w)),1e-300)
            for w in arrays
        )/len(arrays)
    return profile(raw),profile(scaled)

print("=== LEARNED RESIDUAL-COORDINATE GAINS ===")
print("Each pair is (coordinate 163, coordinate 400).")
print("layer input_gain pre_MLP_gain post_attention_gain post_MLP_gain")
for l in range(L):
    pairs=[]
    for suffix in [
        "input_layernorm","pre_feedforward_layernorm",
        "post_attention_layernorm","post_feedforward_layernorm"
    ]:
        g=0.5*(gain(A,l,suffix)+gain(B,l,suffix))
        pairs.append(tuple(round(float(g[j]),7) for j in special))
    print(f"{l:2d}",*pairs)

print("\n=== DIRECT READ/WRITE WEIGHT-ENERGY PROFILES ===")
print("Values are relative to uniform coordinate allocation: 640 * energy fraction.")
print("They are matrix statistics, not activation magnitudes or attention patterns.")
print("layer raw_read(163,400) scaled_read(163,400) "
      "raw_write(163,400) scaled_write(163,400)")
for l in range(L):
    ar,ase=reader_profiles(A,l)
    br,bse=reader_profiles(B,l)
    aw,awse=writer_profiles(A,l)
    bw,bwse=writer_profiles(B,l)
    profiles=[
        (ar+br)/2,(ase+bse)/2,
        (aw+bw)/2,(awse+bwse)/2
    ]
    pairs=[tuple(float(N*p[j]) for j in special) for p in profiles]
    print(f"{l:2d}",*[
        f"({x:.5g},{y:.5g})" for x,y in pairs
    ])

print("\n=== FINAL RMSNORM PARAMETER AUDIT ===")
for name,c in [("A",A),("B",B)]:
    g=1+c.get("model.norm.weight").astype(np.float64)
    print(name,"gain[163,400]=",g[special],
          "median_abs_gain=",float(np.median(np.abs(g))),
          "zero_gain_count=",int(np.count_nonzero(g==0)))
    for j in special:
        print(f"  coordinate {j}: absolute-gain percentile="
              f"{100*np.mean(np.abs(g)<=abs(g[j])):.3f}")

def mlp_factors(c,l):
    d=get(c,l,"mlp.down_proj").T*gain(c,l,"post_feedforward_layernorm")[None,:]
    pre=gain(c,l,"pre_feedforward_layernorm")
    g=get(c,l,"mlp.gate_proj")*pre[None,:]
    u=get(c,l,"mlp.up_proj")*pre[None,:]
    return d,g,u

def input_inner(x,y):
    _,gx,ux=x
    _,gy,uy=y
    gg=rows(gx,gy)
    uu=rows(ux,uy)
    cross=rows(gx,uy)*rows(ux,gy)
    return gg*(gg*uu+2*cross)/3

print("\n=== COEFFICIENT-RANK ROBUSTNESS TO EXCLUDING OUTPUT COORDINATES 163 AND 400 ===")
print("Gate-doubled coefficient energies, summed over individual units.")
print("Exclusion is a coefficient-space projection, NOT a model ablation.")
print("layer excluded_difference_fraction top5_before top5_after")

for l in range(L):
    xa,xb=mlp_factors(A,l),mlp_factors(B,l)
    da,db=xa[0],xb[0]
    saa,sbb,sab=input_inner(xa,xa),input_inner(xb,xb),input_inner(xa,xb)

    daa,dbb,dab=rows(da,da),rows(db,db),rows(da,db)
    full=daa*saa+dbb*sbb-2*dab*sab
    full=np.maximum(full,0)

    ea=np.sum(da[:,special]**2,axis=1)
    eb=np.sum(db[:,special]**2,axis=1)
    eab=np.sum(da[:,special]*db[:,special],axis=1)
    excluded=ea*saa+eb*sbb-2*eab*sab
    excluded=np.maximum(excluded,0)
    remaining=np.maximum(full-excluded,0)

    fraction=float(excluded.sum()/max(float(full.sum()),1e-300))
    before=np.argsort(full)[-5:][::-1].tolist()
    after=np.argsort(remaining)[-5:][::-1].tolist()
    print(f"{l:2d} {fraction:.3%} {before} {after}")

print("\nInterpretation:")
print("- Exact zero direct-read gains would close corresponding numerator paths,")
print("  but not remove a coordinate from an RMS-normalization denominator.")
print("- Nonzero small gains indicate attenuation, not an exact closed path.")
print("- No normalization denominator or hidden-state value was computed.")
print("- The removed embedding/readout still prevents semantic interpretation.")
```