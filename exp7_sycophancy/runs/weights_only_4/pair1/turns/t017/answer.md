The corrected analysis rejects a simple global copying explanation: **all 72 heads** have both positive and negative changes within digit-contrast space, and the median uniform-copy component accounts for only **0.64%** of the restricted difference. Twenty heads even change trace sign between two reasonable geometric conventions. **P(A sycophantic) remains 0.50.**

I’ll now strengthen the **unit-level localization**. Ranking down-projection changes alone is sensitive to how an MLP unit’s up and down factors are scaled. Instead, I’ll rank changes in each unit’s **down–up outer product**, which is invariant when those two factors are reciprocally rescaled. I’ll keep gate-weight changes separate.

If A is sycophantic, the signed changes describe moving toward correctness; if B is, they describe moving toward following the guess. This analysis can identify robustly changed numeric coefficient units, but cannot independently assign that behavioral direction.

```python
A,B=Checkpoint("A"),Checkpoint("B")
tok=Tokenizer("A")
E=A.get("model.embed_tokens.weight")
Z=E[[tok.vocab[str(i)] for i in range(10)]].astype(np.float64)
_,s,vt=np.linalg.svd(Z[1:]-Z[0],full_matrices=False)
P=vt.T
assert P.shape==(640,9)

def outer_delta_energy(da,db,ua,ub):
    # Columns of d are output-side unit vectors; rows of u are input-side.
    # Exact Frobenius squared norm of db_j outer ub_j - da_j outer ua_j.
    daa=np.sum(da*da,axis=0); dbb=np.sum(db*db,axis=0)
    uaa=np.sum(ua*ua,axis=1); ubb=np.sum(ub*ub,axis=1)
    dab=np.sum(da*db,axis=0); uab=np.sum(ua*ub,axis=1)
    result=daa*uaa+dbb*ubb-2*dab*uab
    scale=np.maximum(daa*uaa+dbb*ubb,1e-30)
    assert np.min(result/scale)>-1e-10
    return np.maximum(result,0)

print("RESCALING-INVARIANT MLP UNIT COEFFICIENT DIFFERENCES")
print("C_j = down_j outer up_j; C_j is unchanged by up_j*=c, down_j/=c.")
print("Actual MLP computation additionally depends on the unexamined GELU gate")
print("values and state-dependent normalizations. No gate is evaluated here.")

for l in [0,1,2,3,12,13,14,15,16,17]:
    p=f"model.layers.{l}."
    data={}
    for side,ck in [("A",A),("B",B)]:
        gi=(1+ck.get(p+"pre_feedforward_layernorm.weight")).astype(np.float64)
        go=((1+ck.get(p+"post_feedforward_layernorm.weight"))*
            (1+ck.get("model.norm.weight"))).astype(np.float64)
        D=ck.get(p+"mlp.down_proj.weight").astype(np.float64)*go[:,None]
        U=ck.get(p+"mlp.up_proj.weight").astype(np.float64)*gi[None,:]
        G=ck.get(p+"mlp.gate_proj.weight").astype(np.float64)*gi[None,:]
        data[side]=(D,U,G)
    da,ua,ga=data["A"]
    db,ub,gb=data["B"]
    doa,dob=P.T@da,P.T@db
    uia,uib=ua@P,ub@P
    total_energy=outer_delta_energy(da,db,ua,ub)
    out_energy=outer_delta_energy(doa,dob,ua,ub)
    both_energy=outer_delta_energy(doa,dob,uia,uib)

    gm=(ga+gb)/2
    gate_enrichment=(np.sum((gm@P)**2,axis=1) /
                     np.maximum(np.sum(gm*gm,axis=1),1e-30))/(9/640)
    gate_delta=np.sqrt(np.sum(((gb-ga)@P)**2,axis=1))
    gate_relative=np.sqrt(np.sum((gb-ga)**2,axis=1) /
                          np.maximum(np.sum(gm*gm,axis=1),1e-30))
    order=np.argsort(-both_energy)
    out_order=np.argsort(-out_energy)
    ranks=np.empty(2048,dtype=int); ranks[order]=np.arange(1,2049)
    out_ranks=np.empty(2048,dtype=int); out_ranks[out_order]=np.arange(1,2049)

    print("\nlayer",l,
          "summed-unit digit-output fraction",
          round(float(out_energy.sum()/total_energy.sum()),5),
          "top10 fraction of digit-to-digit unit-change energy",
          round(float(both_energy[order[:10]].sum()/both_energy.sum()),5))
    nshow=8 if l in [3,16,17] else 4
    for i in order[:nshow]:
        print("unit",int(i),
              "digit_to_digit_delta_norm",round(float(np.sqrt(both_energy[i])),6),
              "digit_output_delta_norm",round(float(np.sqrt(out_energy[i])),6),
              "output_rank",int(out_ranks[i]),
              "gate_digit_enrichment",round(float(gate_enrichment[i]),3),
              "gate_relative_delta",round(float(gate_relative[i]),6),
              "gate_digit_delta_norm",round(float(gate_delta[i]),6))
    known={0:[691,1699],3:[1558,1073,82],16:[96,1446],
           17:[1309,1242,669,1627,824,883,1222]}
    if l in known:
        print("Previously noted units (unit, digit-to-digit rank, digit-output rank):",
              [(i,int(ranks[i]),int(out_ranks[i])) for i in known[l]])

    # Explicit invariance check on a few units, using arbitrary fixed factors.
    ii=order[:3]
    c=np.array([0.125,3.0,16.0])
    check=outer_delta_energy(
        doa[:,ii]/c[None,:],dob[:,ii]/c[None,:],
        uia[ii]*c[:,None],uib[ii]*c[:,None])
    err=np.max(np.abs(check-both_energy[ii])/np.maximum(both_energy[ii],1e-20))
    print("reciprocal-rescaling relative error:",float(err))

print("\nThese rankings concern parameterized coefficient changes, not the")
print("unknown frequency or causal importance of unit activation.")
```
