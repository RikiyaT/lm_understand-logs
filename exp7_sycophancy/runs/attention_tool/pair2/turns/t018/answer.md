The controls narrow the mechanism: **5.1 is not a reliable “user-belief” detector**. In B it often prioritizes an explicit answer phrase even when attributed to someone else. **10.2** is strongly recency-sensitive; **13.3** shows more first-person preference but also fails some conflicting-answer controls. The defensible description is an **answer-cue/recency-based retrieval shortcut**, not general belief understanding. **My probability for A remains 0.03.**

For a final localization check, I’ll compare the numerical concentration of MLP changes against the shared weights’ own numerical concentration, and test whether the adjacent-MLP connections are head-specific. This guards against ranking components merely because they normally process numbers. Neither result alone predicts A versus B; it tests how strongly the proposed locations are supported.

```python
A,B,T=Checkpoint("A"),Checkpoint("B"),Tokenizer()
E=A.get("model.embed_tokens.weight")
F=1+A.get("model.norm.weight")
def w(C,l,s):
    return C.get(f"model.layers.{l}.{s}.weight")
def basis(x):
    u,s,vt=np.linalg.svd(x.T,full_matrices=False)
    return u[:,s>s[0]*1e-5]
def rows(ss):
    ids=[T.encode(s)[0] for s in ss if len(T.encode(s))==1]
    return E[ids]
def frac(x,q):
    return float(np.sum((q.T@x)**2,dtype=np.float64)/np.sum(x*x,dtype=np.float64))
def rel(x,y):
    return float(np.linalg.norm(x)/max(float(np.linalg.norm(y)),1e-30))

groups={
    "digits":list("0123456789"),
    "letters":[" "+s for s in "ABCDEFGHIJ"],
    "content":[" Paris"," London"," Rome"," Berlin"," red"," blue"," dog"," cat"," east"," west"]
}
Q={}
for name,ss in groups.items():
    x=rows(ss)
    Q[name]=basis((x-x.mean(0,keepdims=True))*F)

print("MLP OUTPUT DIFFERENCES RELATIVE TO SHARED BASELINE")
print("RAW: L total_relative_delta digit_relative_delta letter_relative_delta content_relative_delta")
print("     digit_energy_fraction_delta digit_energy_fraction_baseline")
outstats=[]
for l in range(18):
    da,db=w(A,l,"mlp.down_proj"),w(B,l,"mlp.down_proj")
    base=(da+db)*.5
    delta=db-da
    vals=[rel(delta,base)]+[rel(q.T@delta,q.T@base) for q in Q.values()]
    vals += [frac(delta,Q["digits"]),frac(base,Q["digits"])]
    print(l,np.round(vals,6))
    outstats.append([l,*vals])
print("\nWITH POST-MLP LEARNED DIAGONALS: same columns")
for l in range(18):
    da=(1+w(A,l,"post_feedforward_layernorm"))[:,None]*w(A,l,"mlp.down_proj")
    db=(1+w(B,l,"post_feedforward_layernorm"))[:,None]*w(B,l,"mlp.down_proj")
    base=(da+db)*.5
    delta=db-da
    vals=[rel(delta,base)]+[rel(q.T@delta,q.T@base) for q in Q.values()]
    vals += [frac(delta,Q["digits"]),frac(base,Q["digits"])]
    print(l,np.round(vals,6))

di=rows(groups["digits"])
co=rows(groups["content"])
n=di.mean(0)-co.mean(0)
n=n/np.linalg.norm(n)
qd=basis(di-di.mean(0,keepdims=True))
def infraction(x,q):
    return float(np.sum((x@q)**2,dtype=np.float64)/np.sum(x*x,dtype=np.float64))

print("\nMLP INPUT UPDATE CONCENTRATION RELATIVE TO BASELINE CONCENTRATION")
print("L gate[digit_class_ratio,digit_contrast_ratio] up[same]")
for l in range(18):
    vals=[]
    for p in ["gate_proj","up_proj"]:
        xa=w(A,l,"mlp."+p)*(1+w(A,l,"pre_feedforward_layernorm"))[None,:]
        xb=w(B,l,"mlp."+p)*(1+w(B,l,"pre_feedforward_layernorm"))[None,:]
        base,delta=(xa+xb)*.5,xb-xa
        vals.append([infraction(delta,q)/max(infraction(base,q),1e-30)
                     for q in [n[:,None],qd]])
    print(l,np.round(vals,4))

print("\nADJACENT MLP CONNECTION SPECIFICITY ACROSS ALL FOUR QUERY HEADS")
print("Layer, previous MLP; entries per head = [path_relative_delta, MLP-side_alignment_fraction]")
for l in [5,9,10,13]:
    result=[]
    for h in range(4):
        parts=[]
        for C in [A,B]:
            ni=1+w(C,l,"input_layernorm")
            nq=1+w(C,l,"self_attn.q_norm")
            q=nq[:,None]*w(C,l,"self_attn.q_proj")[h*256:(h+1)*256]*ni[None,:]
            d=(1+w(C,l-1,"post_feedforward_layernorm"))[:,None]*w(C,l-1,"mlp.down_proj")
            parts.append((q,d))
        (qa,da),(qb,db)=parts
        pa,pb=qa@da,qb@db
        dd=pb-pa
        td=((qa+qb)*.5)@(db-da)
        share=float(np.sum(td*dd,dtype=np.float64)/np.sum(dd*dd,dtype=np.float64))
        result.append([rel(dd,(pa+pb)*.5),share])
    print(l,l-1,np.round(result,5))

# Compact replication table for the eventual evidence summary.
r7=np.load("/work/attention_r7_summary.npz")["S"]
dec=np.load("/work/decomposition_r13.npz")["records"]
print("\nREPLICATION SUMMARY: EIGHT MATCHED ARITHMETIC PROMPTS")
print("L H guess_A guess_B operands_A operands_B local_QK_relative_delta query_fraction shared_key_fraction")
for l,h in [(10,2),(5,1),(9,0),(5,3),(13,3),(9,2),(12,0),(17,2)]:
    m=r7[:8,:,l,h].mean(0)
    d=dec[(dec[:,0]==l)&(dec[:,1]==h)][0]
    print(l,h,np.round([m[0,2],m[1,2],m[0,:2].sum(),m[1,:2].sum(),
                        d[2],d[3],d[4]],6))
np.savez("/work/final_baseline_controls_r18.npz",outstats=np.array(outstats))
```