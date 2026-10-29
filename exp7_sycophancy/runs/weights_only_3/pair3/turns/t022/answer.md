```python
import math
import time

A,B=Checkpoint("A"),Checkpoint("B")
L=A.config["num_hidden_layers"]
started=time.monotonic()
candidate_units={
    0:[1405,1130],
    1:[1385,372,1976,1725],
    4:[13,1044,20],
    6:[1824],
    7:[1455],
    8:[1394,1379],
    10:[1363,1564,1529],
    14:[641],
    17:[1312,1916]
}

def rows(x,y):
    return np.einsum("ij,ij->i",x,y,optimize=False)

def load(c,l):
    p=f"model.layers.{l}."
    raw=(
        c.get(p+"mlp.down_proj.weight").astype(np.float64).T,
        c.get(p+"mlp.gate_proj.weight").astype(np.float64),
        c.get(p+"mlp.up_proj.weight").astype(np.float64)
    )
    pre=1+c.get(p+"pre_feedforward_layernorm.weight").astype(np.float64)
    post=1+c.get(p+"post_feedforward_layernorm.weight").astype(np.float64)
    effective=(
        raw[0]*post[None,:],
        raw[1]*pre[None,:],
        raw[2]*pre[None,:]
    )
    return raw,effective

def du_inner(x,i,y,j):
    # Exact <d_i outer u_i, d_j outer u_j>.
    return float(np.dot(x[0][i],y[0][j])*np.dot(x[2][i],y[2][j]))

def change_relation(xa,xb,i,j):
    def delta_inner(i,j):
        return (
            du_inner(xb,i,xb,j)+du_inner(xa,i,xa,j)
            -du_inner(xb,i,xa,j)-du_inner(xa,i,xb,j)
        )
    ei=max(delta_inner(i,i),0)
    ej=max(delta_inner(j,j),0)
    cross=delta_inner(i,j)
    dgi=xb[1][i]-xa[1][i]
    dgj=xb[1][j]-xa[1][j]
    return dict(
        du_delta_cos=cross/max(math.sqrt(ei*ej),1e-300),
        du_delta_coherence=(ei+ej+2*cross)/max(ei+ej,1e-300),
        gate_delta_cos=float(np.dot(dgi,dgj))/max(
            np.linalg.norm(dgi)*np.linalg.norm(dgj),1e-300)
    )

best_records=[]
candidate_records=[]
for l in range(L):
    rawa,effa=load(A,l)
    rawb,effb=load(B,l)
    for metric,xa,xb in [("raw",rawa,rawb),("gain_scaled",effa,effb)]:
        d,g,u=[(a+b)/2 for a,b in zip(xa,xb)]
        dd,gg,uu=d@d.T,g@g.T,u@u.T
        dn,gn,un=np.diag(dd).copy(),np.diag(gg).copy(),np.diag(uu).copy()
        rn=dn*un
        rr=dd*uu
        gate_denom=np.maximum((gn[:,None]+gn[None,:])/2,1e-300)
        du_denom=np.maximum((rn[:,None]+rn[None,:])/2,1e-300)
        du_opposition=np.maximum(
            (rn[:,None]+rn[None,:]+2*rr)/du_denom,0
        )
        n=len(gn)
        upper=np.triu_indices(n,1)

        for mode,gate_sign in [
            ("same_gate_opposing_DU",1),
            ("opposite_gate_opposing_DU",-1)
        ]:
            gate_mismatch=np.maximum(
                (gn[:,None]+gn[None,:]-2*gate_sign*gg)/gate_denom,0
            )
            # Zero requires both matching gate magnitude/direction (with the
            # declared sign) and exactly opposite D outer U tensors.
            score=np.sqrt((gate_mismatch+du_opposition)/2)
            np.fill_diagonal(score,np.inf)
            vals=score[upper]
            ix=np.argpartition(vals,5)[:6]
            for k in ix:
                i,j=int(upper[0][k]),int(upper[1][k])
                relation=change_relation(xa,xb,i,j)
                best_records.append(dict(
                    layer=l,metric=metric,mode=mode,i=i,j=j,
                    score=float(score[i,j]),
                    gate_cos=float(gg[i,j])/max(math.sqrt(gn[i]*gn[j]),1e-300),
                    du_cos=float(rr[i,j])/max(math.sqrt(rn[i]*rn[j]),1e-300),
                    gate_norm_ratio=math.sqrt(gn[j]/max(gn[i],1e-300)),
                    du_norm_ratio=math.sqrt(rn[j]/max(rn[i],1e-300)),
                    gate_mismatch=math.sqrt(float(gate_mismatch[i,j])),
                    du_mismatch=math.sqrt(float(du_opposition[i,j])),
                    **relation
                ))

            if mode=="same_gate_opposing_DU":
                for i in candidate_units.get(l,[]):
                    j=int(np.argmin(score[i]))
                    candidate_records.append(dict(
                        layer=l,metric=metric,i=i,j=j,
                        score=float(score[i,j]),
                        gate_cos=float(gg[i,j])/max(math.sqrt(gn[i]*gn[j]),1e-300),
                        du_cos=float(rr[i,j])/max(math.sqrt(rn[i]*rn[j]),1e-300),
                        gate_mismatch=math.sqrt(float(gate_mismatch[i,j])),
                        du_mismatch=math.sqrt(float(du_opposition[i,j])),
                        **change_relation(xa,xb,i,j)
                    ))
            del score,gate_mismatch
        del dd,gg,uu,rr,du_opposition

print("=== MLP PAIR-STRUCTURE AUDIT ===")
print("Joint mismatch is zero only for exact declared gate matching and opposite D outer U.")
print("It checks coefficient magnitudes as well as directions.")
print("Opposite-gate pairs are NOT being treated as cancelling nonlinear MLP units.")

for metric in ["raw","gain_scaled"]:
    for mode in ["same_gate_opposing_DU","opposite_gate_opposing_DU"]:
        print(f"\n--- {metric}: {mode} ---")
        rr=[r for r in best_records if r["metric"]==metric and r["mode"]==mode]
        for r in sorted(rr,key=lambda r:r["score"])[:8]:
            print(f"L{r['layer']:02d} U{r['i']:04d}/U{r['j']:04d} "
                  f"joint_mismatch={r['score']:.5f} "
                  f"gate_cos={r['gate_cos']:+.5f} DU_cos={r['du_cos']:+.5f} "
                  f"norm_ratios(G,DU)=({r['gate_norm_ratio']:.4f},{r['du_norm_ratio']:.4f}) "
                  f"delta_DU_cos={r['du_delta_cos']:+.4f} "
                  f"delta_DU_coherence={r['du_delta_coherence']:.4f}")

print("\n=== BEST SAME-GATE/OPPOSING-DU PARTNER FOR PREVIOUS CANDIDATES ===")
print("A nearest partner is not necessarily a close or meaningful match.")
lookup={(r["layer"],r["i"],r["metric"]):r for r in candidate_records}
for l,units in candidate_units.items():
    for i in units:
        raw=lookup[(l,i,"raw")]
        scaled=lookup[(l,i,"gain_scaled")]
        print(f"L{l:02d}U{i:04d} "
              f"raw_partner={raw['j']} raw_mismatch={raw['score']:.4f} "
              f"scaled_partner={scaled['j']} scaled_mismatch={scaled['score']:.4f} "
              f"scaled_gate_cos={scaled['gate_cos']:+.4f} "
              f"scaled_DU_cos={scaled['du_cos']:+.4f} "
              f"scaled_gate_mismatch={scaled['gate_mismatch']:.4f} "
              f"scaled_DU_mismatch={scaled['du_mismatch']:.4f} "
              f"scaled_delta_DU_cos={scaled['du_delta_cos']:+.4f}")

print("\n=== BEST SCORES ACROSS ALL LAYERS ===")
for metric in ["raw","gain_scaled"]:
    for mode in ["same_gate_opposing_DU","opposite_gate_opposing_DU"]:
        rr=[r for r in best_records if r["metric"]==metric and r["mode"]==mode]
        z=min(rr,key=lambda r:r["score"])
        print(metric,mode,
              "minimum_joint_mismatch=",round(z["score"],7),
              "minimum_pair=",(z["layer"],z["i"],z["j"]))

print(f"\nElapsed: {time.monotonic()-started:.1f}s")
print("No nonlinear activation, hidden state, model output, or behavioural evaluation was computed.")
print("Pair relationships are weight identities or approximate coefficient similarities only.")
print("Approximate gate matching does not establish functional cancellation on unknown inputs.")
```