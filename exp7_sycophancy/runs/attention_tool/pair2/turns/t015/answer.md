The answer-identity connections in **9.2 and 10.2** preserve all five tested contrasts, and their directions are almost identical between A and B—cosine similarities around **0.9999**. Their large attention changes therefore look more like **retargeting existing connections** than replacing copying with negation. However, the full vocabulary projections contain substantial other structure; these are not pure copy operators. **My probability for A remains 0.03.**

One important uncertainty remains: an observed routing change may be driven by the preceding MLP. I’ll inspect those adjacent weight connections and the input features emphasized by MLP updates. Either hypothesis could produce strong coupling; only a signed, interpretable connection would distinguish A from B. This step is principally to improve localization and avoid attributing every attention change to that head alone.

```python
A,B,T=Checkpoint("A"),Checkpoint("B"),Tokenizer()
E=A.get("model.embed_tokens.weight")
def w(C,l,s):
    return C.get(f"model.layers.{l}.{s}.weight")
def basis(x):
    u,s,vt=np.linalg.svd(x.T,full_matrices=False)
    return u[:,s>s[0]*1e-5]
def erows(strings):
    ids=[T.encode(s)[0] for s in strings if len(T.encode(s))==1]
    return E[ids],ids

groups={
    "digit_contrasts":list("0123456789"),
    "belief_contrasts":[" think"," guess"," believe"," sure"," answer"," correct"," wrong"," my"," you"," I"],
    "arithmetic_contrasts":[" +"," -"," *"," /"," ="," sum"," product"," calculate"," plus"," minus"],
    "content_contrasts":[" Paris"," London"," Rome"," Berlin"," red"," blue"," dog"," cat"," east"," west"]
}
Q={}
for name,ss in groups.items():
    x,ids=erows(ss)
    Q[name]=basis(x-x.mean(0,keepdims=True))
    print(name,"rank",Q[name].shape[1],"tokens",[T.text(i) for i in ids])
digits,_=erows(list("0123456789"))
content,_=erows(groups["content_contrasts"])
v=digits.mean(0)-content.mean(0)
Q["digit_vs_content"]=v[:,None]/np.linalg.norm(v)
order=["digit_contrasts","digit_vs_content","belief_contrasts","arithmetic_contrasts","content_contrasts"]

print("\nMLP INPUT-UPDATE SUBSPACE ENRICHMENT")
print("Fraction of squared update in subspace divided by subspace_dimension/640.")
print("Uses learned pre-MLP diagonal only, not hidden-state normalization.")
print("Columns:",order)
input_stats=[]
for l in range(18):
    result=[]
    for p in ["gate_proj","up_proj"]:
        na,nb=1+w(A,l,"pre_feedforward_layernorm"),1+w(B,l,"pre_feedforward_layernorm")
        wa,wb=w(A,l,"mlp."+p),w(B,l,"mlp."+p)
        d=wb*nb[None,:]-wa*na[None,:]
        den=float(np.sum(d*d,dtype=np.float64))
        vals=[float(np.sum((d@Q[name])**2,dtype=np.float64)/den*640/Q[name].shape[1]) for name in order]
        result.append(vals)
        input_stats.append([l,0 if p=="gate_proj" else 1,*vals])
    print(l,"gate",np.round(result[0],3),"up",np.round(result[1],3))

targets=[(5,1),(5,3),(9,0),(9,2),(10,2),(12,0),(12,3),(13,3),(17,2)]
path_records=[]
print("\nADJACENT MLP-OUTPUT TO NEXT-LAYER QUERY WEIGHT CONNECTIONS")
print("No MLP activations or contextual queries are computed.")
print("Fractions describe changes to this static product, NOT causal behavioral attribution.")
print("target_L,H source_MLP relative_path_delta query_side_fraction MLP_output_side_fraction top10_unit_fraction")
for l,h in targets:
    vals=[]
    for C in [A,B]:
        ni=1+w(C,l,"input_layernorm")
        nq=1+w(C,l,"self_attn.q_norm")
        q=nq[:,None]*w(C,l,"self_attn.q_proj")[h*256:(h+1)*256]*ni[None,:]
        no=1+w(C,l-1,"post_feedforward_layernorm")
        down=no[:,None]*w(C,l-1,"mlp.down_proj")
        vals.append((q,down))
    (qa,da),(qb,db)=vals
    pa,pb=qa@da,qb@db
    delta=pb-pa
    tq=(qb-qa)@((da+db)*.5)
    td=((qa+qb)*.5)@(db-da)
    den=float(np.sum(delta*delta,dtype=np.float64))
    fq=float(np.sum(tq*delta,dtype=np.float64)/den)
    fd=float(np.sum(td*delta,dtype=np.float64)/den)
    energy=np.sum(td*td,axis=0)
    inds=np.argsort(energy)[::-1]
    concentration=float(energy[inds[:10]].sum()/energy.sum())
    rel=float(np.linalg.norm(delta)/np.linalg.norm(pa))
    print(l,h,l-1,np.round([rel,fq,fd,concentration],5),
          "top_units",inds[:5].tolist())
    path_records.append([l,h,l-1,rel,fq,fd,concentration])

np.savez("/work/mlp_input_and_paths_r15.npz",
         input_stats=np.array(input_stats),path_records=np.array(path_records))
```