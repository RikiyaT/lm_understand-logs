The early-head findings are strongly structured. A single query-update mode explains **94% of head 3’s query-side belief effect** and approximately all of head 0’s query-side operator effect. Weight sign-randomization did not reproduce either effect in 6,000 controls. This establishes coherent weight changes—not proof of their behavioural use. **My probability that A is sycophantic remains 0.30.**

Next I’ll distinguish **changing numerical feature construction** from **changing how existing features write numerical answers** in the late MLPs. I’ll attribute their coefficient differences to gate, up, and down matrices, and characterize selected units’ numerical weight directions. Either label assignment could produce large changes in any of these matrices, so attribution magnitudes alone will not change the A/B assignment.

```python
A,B,tok=Checkpoint("A"),Checkpoint("B"),Tokenizer()
E=A.get("model.embed_tokens.weight")
ids=[tok.encode(s)[0] for s in "0123456789"]
Zd=E[ids].astype(np.float64)
Zd-=Zd.mean(axis=0)
_,s,vt=np.linalg.svd(Zd,full_matrices=False)
Qd=vt[s>s[0]*1e-6].T

def rd(x,y):
    return np.einsum("ij,ij->i",x,y)

def tensor_inner(P,R):
    d,g,u=P; e,h,v=R
    return rd(d,e)*.5*(rd(g,h)*rd(u,v)+rd(g,v)*rd(u,h))

def load_parts(l,scaled=False):
    p=f"model.layers.{l}."
    ga=A.get(p+"mlp.gate_proj.weight").astype(np.float64)
    gb=B.get(p+"mlp.gate_proj.weight").astype(np.float64)
    ua=A.get(p+"mlp.up_proj.weight").astype(np.float64)
    ub=B.get(p+"mlp.up_proj.weight").astype(np.float64)
    da=A.get(p+"mlp.down_proj.weight").T.astype(np.float64)
    db=B.get(p+"mlp.down_proj.weight").T.astype(np.float64)
    if scaled:
        ia=1+A.get(p+"pre_feedforward_layernorm.weight")
        ib=1+B.get(p+"pre_feedforward_layernorm.weight")
        oa=(1+A.get(p+"post_feedforward_layernorm.weight"))*(1+A.get("model.norm.weight"))
        ob=(1+B.get(p+"post_feedforward_layernorm.weight"))*(1+B.get("model.norm.weight"))
        ga*=ia; ua*=ia; gb*=ib; ub*=ib
        da*=oa; db*=ob
    da=da@Qd; db=db@Qd
    dm,gm,um=(da+db)*.5,(ga+gb)*.5,(ua+ub)*.5
    dd,dg,du=db-da,gb-ga,ub-ua
    # Exact symmetric midpoint expansion of the trilinear coefficient:
    # Delta(d*g*u) = dd*gm*um + dm*dg*um + dm*gm*du + dd*dg*du/4.
    return [(dd,gm,um),(dm,dg,um),(dm,gm,du),(.25*dd,dg,du)]

def unit_attribution(parts):
    gram=np.empty((4,4,2048),np.float64)
    for i in range(4):
        for j in range(i,4):
            gram[i,j]=gram[j,i]=tensor_inner(parts[i],parts[j])
    unit=np.maximum(gram.sum(axis=(0,1)),0)
    total=max(float(unit.sum()),1e-30)
    shares=gram.sum(axis=(1,2))/total
    return unit,shares

print("=== EXACT UNITWISE NUMERICAL-COEFFICIENT ATTRIBUTION ===")
print("Signed shares are <component_delta, total_delta>/||total_delta||^2.")
print("Shares sum to one; negative shares represent cancellation.")
print(" L mode       down_share   gate_share     up_share   cubic_cross_share")
all_units={}
for l in range(18):
    for scaled in (False,True):
        parts=load_parts(l,scaled)
        unit,shares=unit_attribution(parts)
        if not scaled:
            all_units[l]=unit
        print(f"{l:2d} {'diag' if scaled else 'raw':4s} "
              +" ".join(f"{v:13.6f}" for v in shares))
        del parts

print("\n=== AGGREGATE ATTRIBUTION INCLUDING CROSS-UNIT TERMS ===")
for l in (16,17):
    parts=load_parts(l,False)
    gram=np.zeros((4,4),np.float64)
    for k in range(Qd.shape[1]):
        slices=[]
        for d,g,u in parts:
            C=g.T@(d[:,k,None]*u)
            slices.append((C+C.T)*.5)
        for i in range(4):
            for j in range(i,4):
                value=float(np.sum(slices[i]*slices[j]))
                gram[i,j]+=value
                if i!=j:
                    gram[j,i]+=value
    total=float(gram.sum())
    shares=gram.sum(axis=1)/max(total,1e-30)
    print("layer",l,"aggregate_delta_F",np.sqrt(total),
          "down/gate/up/cubic shares",np.round(shares,6))
    del parts

templates={}
for d in range(10):
    v=np.eye(10)[d]-.1
    templates[f"digit_{d}"]=v/np.linalg.norm(v)
for label,v in [
    ("increasing_digit_value",np.arange(10,dtype=float)),
    ("even_vs_odd",np.array([1 if d%2==0 else -1 for d in range(10)],float)),
]:
    v=v-v.mean()
    templates[label]=v/np.linalg.norm(v)

def describe(v):
    profile=Zd@v
    n=float(np.linalg.norm(profile))
    if n<1e-30:
        return "none",0.0,0.0
    p=profile/n
    label,cos=max(((label,float(p@t)) for label,t in templates.items()),
                  key=lambda pair:abs(pair[1]))
    fraction=float(np.sum((Qd.T@v)**2)/max(float(v@v),1e-30))
    return label,cos,fraction

print("\n=== NUMERICAL WEIGHT GEOMETRY OF SELECTED LATE MLP UNITS ===")
print("Template signs describe weight alignments, NOT firing or suppression on an input.")
for l in (14,15,16,17):
    p=f"model.layers.{l}.mlp."
    ga=A.get(p+"gate_proj.weight").astype(np.float64)
    gb=B.get(p+"gate_proj.weight").astype(np.float64)
    ua=A.get(p+"up_proj.weight").astype(np.float64)
    ub=B.get(p+"up_proj.weight").astype(np.float64)
    da=A.get(p+"down_proj.weight").astype(np.float64)
    db=B.get(p+"down_proj.weight").astype(np.float64)
    js=list(np.argsort(all_units[l])[::-1][:5])
    if l==17:
        for j in (1180,1309,883,310):
            if j not in js: js.append(j)
    if l==16:
        for j in (96,1875,1446):
            if j not in js: js.append(j)
    for j in js:
        print(f"\nL{l} unit {j}, numerical coefficient delta norm={np.sqrt(all_units[l][j]):.7f}")
        for label,v in [
            ("gate_mid",(ga[j]+gb[j])*.5),
            ("up_mid",(ua[j]+ub[j])*.5),
            ("down_mid",(da[:,j]+db[:,j])*.5),
            ("down_delta",db[:,j]-da[:,j]),
        ]:
            template,cos,fraction=describe(v)
            print(f" {label:12s} {template:24s} alignment={cos:+.3f} "
                  f"digit_subspace_fraction={fraction:.4f}")
```