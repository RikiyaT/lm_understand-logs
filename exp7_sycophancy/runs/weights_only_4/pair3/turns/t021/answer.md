The systematically selected MLP motifs did **not** show a general shift toward copying in B: several move the other way. I therefore revise slightly toward uncertainty: **\(P(A\text{ sycophantic})=0.35\)**.

A key alternative remains: stronger “I think” binding could mark the user’s guess **for rejection**, not agreement. I’ll investigate which numerical MLP weight directions can read the belief head’s write directions, comparing numeral-preserving and numeral-opposing motifs. Connections favouring preservation in B would support B; connections favouring rejection would support A. These linear connections can suggest possible routes, but cannot establish that the routes are active.

```python
A,B,tok=Checkpoint("A"),Checkpoint("B"),Tokenizer()
E=A.get("model.embed_tokens.weight")
Ec=E-E.mean(axis=0)
ids=[tok.encode(s)[0] for s in "0123456789"]
Zd=E[ids].astype(np.float64)
Zd-=Zd.mean(axis=0)
_,s,vt=np.linalg.svd(Zd,full_matrices=False)
Qd=vt[s>s[0]*1e-6].T

def proto(words):
    ids=[]
    for w in words:
        ii=tok.encode(w)
        if len(ii)==1 and ii[0] not in ids: ids.append(ii[0])
    X=Ec[ids].astype(np.float64)
    X/=np.linalg.norm(X,axis=1,keepdims=True)
    return X.mean(axis=0)

cue=proto([" I","I"," my"," me"])-proto([" you"," he"," she"," we"," they"])
cn=float(np.linalg.norm(cue))

def writer(cp,h,mode):
    p="model.layers.1."
    O=cp.get(p+"self_attn.o_proj.weight")[:,h*256:(h+1)*256].astype(np.float64)
    V=cp.get(p+"self_attn.v_proj.weight").astype(np.float64)
    if mode=="diag":
        O*=(1+cp.get(p+"post_attention_layernorm.weight"))[:,None]
        V*=1+cp.get(p+"input_layernorm.weight")
    return O@V

known={
    16:[332,931,1875,1955,689,96,1446],
    17:[228,1192,1242,1180,1309,883,310,750],
}
records=[]
print("=== LINEAR WRITER-TO-MLP-READER WEIGHT CONNECTIONS ===")
print("Operator analyzed: gate_or_up_l @ OV_layer1_head @ cue_weight_direction.")
print("No gate response is evaluated or multiplied with another branch.")
print("Intervening nonlinear branches are NOT modeled; these are possible direct-residual links.")

for mode in ("raw","diag"):
    source={}
    for h in range(4):
        Ta,Tb=writer(A,h,mode),writer(B,h,mode)
        Tm,Dt=(Ta+Tb)*.5,Tb-Ta
        source[h]=(Ta@cue,Tb@cue,Tm@cue,Dt@cue,float(np.linalg.norm(Tm)))

    for l in range(2,18):
        p=f"model.layers.{l}."
        da=A.get(p+"mlp.down_proj.weight").T.astype(np.float64)
        db=B.get(p+"mlp.down_proj.weight").T.astype(np.float64)
        dm=(da+db)*.5
        numeric=np.sum((dm@Qd)**2,axis=1)/np.maximum(np.sum(dm*dm,axis=1),1e-30)

        branch_data={}
        for branch in ("gate","up"):
            wa=A.get(p+f"mlp.{branch}_proj.weight").astype(np.float64)
            wb=B.get(p+f"mlp.{branch}_proj.weight").astype(np.float64)
            if mode=="diag":
                wa*=1+A.get(p+"pre_feedforward_layernorm.weight")
                wb*=1+B.get(p+"pre_feedforward_layernorm.weight")
            wm,dw=(wa+wb)*.5,wb-wa
            wn=np.maximum(np.linalg.norm(wm,axis=1),1e-30)
            all_heads=[]
            for sh in range(4):
                ta,tb,tm,dt,tn=source[sh]
                ca,cb=wa@ta,wb@tb
                reader=dw@tm
                writer_part=wm@dt
                scale=wn*max(tn*cn,1e-30)
                all_heads.append(np.stack([
                    (ca+cb)*.5/scale,(cb-ca)/scale,
                    reader/scale,writer_part/scale],axis=1))
            branch_data[branch]=np.stack(all_heads,axis=1)

        for j in range(2048):
            if numeric[j]<.08 and j not in known.get(l,[]):
                continue
            records.append(dict(layer=l,unit=j,mode=mode,numeric=float(numeric[j]),
                                gate=branch_data["gate"][j],up=branch_data["up"][j]))

print("\n=== PREVIOUSLY IDENTIFIED NUMERICAL MOTIFS ===")
print("G/U baseline = normalized WEIGHT CONNECTION, not activation.")
print("Deltas are B-A. Source-control values are baseline gate links for heads 0..3.")
for mode in ("raw","diag"):
    print("\n",mode)
    print(" L unit  numeric_frac   G_base_H3    U_base_H3    G_delta_H3"
          "  G_reader_delta G_writer_delta  G_base_sources_0..3")
    for r in records:
        if r["mode"]!=mode or r["unit"] not in known.get(r["layer"],[]):
            continue
        g,u=r["gate"][3],r["up"][3]
        print(f"{r['layer']:2d} {r['unit']:4d} {r['numeric']:12.4f} "
              f"{g[0]:+12.6g} {u[0]:+12.6g} {g[1]:+12.6g} "
              f"{g[2]:+14.6g} {g[3]:+14.6g} "
              +str([float(f"{v:.6g}") for v in r["gate"][:,0]]))

print("\n=== STRONGEST NUMERICAL READERS OF SOURCE HEAD 3 ===")
for mode in ("raw","diag"):
    rr=[r for r in records if r["mode"]==mode and r["numeric"]>=.08]
    for criterion in ("baseline","change"):
        idx=0 if criterion=="baseline" else 1
        ranked=sorted(rr,key=lambda r:abs(r["gate"][3,idx])*np.sqrt(r["numeric"]),
                      reverse=True)
        print(mode,criterion,"top 10 gate-reader connections")
        for r in ranked[:10]:
            print((r["layer"],r["unit"]),
                  "numeric_frac",round(r["numeric"],3),
                  "G_base",float(f"{r['gate'][3,0]:.6g}"),
                  "G_delta",float(f"{r['gate'][3,1]:.6g}"),
                  "U_base",float(f"{r['up'][3,0]:.6g}"),
                  "source_head3 / max_other_abs",
                  float(f"{abs(r['gate'][3,idx])/max(np.max(np.abs(r['gate'][:3,idx])),1e-30):.4g}"))
```