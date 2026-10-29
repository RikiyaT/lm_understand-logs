The downstream-product check changes the MLP shortlist: **layers 4, 10, 14, and 15** lead, with relative writing-path changes of approximately **3.50%, 3.11%, 2.81%, and 2.63%**. These results barely change when coordinates 163 and 400 are excluded. Thus, these candidates are not merely artifacts of the two large writing coordinates. This still measures linear-factor connectivity, not behavioral influence. My probability that A is sycophantic remains **0.50**.

I’ll make one final refinement: determine whether those MLP differences concentrate in a few units after downstream connectivity is included, and distinguish changed writing columns from changed gating/reading rows. This can justify—or reject—an individual-unit shortlist.

Under either sycophancy assignment, the same units could carry the contrast in opposite directions. No result here can establish that a unit represents “the user’s guess” rather than another feature.

```python
A, B = Checkpoint("A"), Checkpoint("B")

def get(C,l,s):
    return C.get(f"model.layers.{l}.{s}").astype(np.float64)

def mlp(C,l):
    pre = 1+get(C,l,"pre_feedforward_layernorm.weight")
    post = 1+get(C,l,"post_feedforward_layernorm.weight")
    G = get(C,l,"mlp.gate_proj.weight")*pre[None,:]
    U = get(C,l,"mlp.up_proj.weight")*pre[None,:]
    D = get(C,l,"mlp.down_proj.weight")*post[:,None]
    return G,U,D

def next_reader(C,l):
    si = 1+get(C,l,"input_layernorm.weight")
    Q = get(C,l,"self_attn.q_proj.weight").reshape(4,256,640)
    Q *= (1+get(C,l,"self_attn.q_norm.weight"))[None,:,None]
    K = get(C,l,"self_attn.k_proj.weight")
    K *= (1+get(C,l,"self_attn.k_norm.weight"))[:,None]
    V = get(C,l,"self_attn.v_proj.weight")
    return np.concatenate([Q.reshape(1024,640),K,V],axis=0)*si[None,:]

def inner(a,b):
    return np.einsum("ij,ij->i",a,b)

def tensor_difference(fa,fb):
    aa=np.ones(fa[0].shape[0])
    bb=aa.copy()
    ab=aa.copy()
    for a,b in zip(fa,fb):
        aa*=inner(a,a)
        bb*=inner(b,b)
        ab*=inner(a,b)
    return np.maximum(aa+bb-2*ab,0),(aa+bb)/2

print("UNIT-LEVEL COEFFICIENT DIFFERENCES WITH A FROZEN MEAN DOWNSTREAM READER")
print("These remain structural tensors, not nonlinear MLP evaluations.")
print("Term norms are not additive because terms can reinforce or cancel.")

for l in [4,10,14,15,8,11,3,1]:
    ga,ua,da=mlp(A,l)
    gb,ub,db=mlp(B,l)
    R=(next_reader(A,l+1)+next_reader(B,l+1))/2

    raw_delta2,raw_ref2=tensor_difference([da.T,ga,ua],[db.T,gb,ub])
    # Multiplication of weight matrices only.
    ca,cb=(R@da).T,(R@db).T
    delta2,ref2=tensor_difference([ca,ga,ua],[cb,gb,ub])
    idx=np.argsort(delta2)[::-1]
    raw_idx=np.argsort(raw_delta2)[::-1]
    total=float(delta2.sum())

    cm,gm,um=(ca+cb)/2,(ga+gb)/2,(ua+ub)/2
    dc,dg,du=cb-ca,gb-ga,ub-ua
    writer2=inner(dc,dc)*inner(gm,gm)*inner(um,um)
    gate2=inner(cm,cm)*inner(dg,dg)*inner(um,um)
    up2=inner(cm,cm)*inner(gm,gm)*inner(du,du)
    cubic2=inner(dc,dc)*inner(dg,dg)*inner(du,du)/16

    print(f"\nLAYER {l}",
          f'bank_relative={np.sqrt(total/ref2.sum()):.6f}',
          f'top1_share={delta2[idx[0]]/total:.5f}',
          f'top10_share={delta2[idx[:10]].sum()/total:.5f}',
          f'top100_share={delta2[idx[:100]].sum()/total:.5f}',
          f'top10_overlap_with_uncoupled={len(set(idx[:10])&set(raw_idx[:10]))}/10')
    print("  uncoupled top5:",raw_idx[:5].tolist())
    print("  coupled top5:",idx[:5].tolist())
    for i in idx[:5]:
        nd=np.sqrt(max(float(delta2[i]),1e-300))
        print("  unit",int(i),
              f'energy_share={delta2[i]/total:.5f}',
              f'coefficient_relative={np.sqrt(delta2[i]/max(ref2[i],1e-300)):.5f}',
              "term_norms/total_delta_norm:",
              f'writer={np.sqrt(writer2[i])/nd:.4f}',
              f'gate={np.sqrt(gate2[i])/nd:.4f}',
              f'up={np.sqrt(up2[i])/nd:.4f}',
              f'cubic={np.sqrt(cubic2[i])/nd:.7f}')
```