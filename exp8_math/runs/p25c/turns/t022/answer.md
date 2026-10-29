```python
# Hypothesis: the single-neuron ablation is too weak because repetitive checking is distributed across a shared residual direction written by multiple components.
def read_generations(path):
    with open(path) as f:
        rows=[json.loads(line) for line in f]
    return [dict(s,id=r["id"],source=r["source"]) for r in rows for s in r["samples"]]
trial_a_original=read_generations("/work/gen_6_original.jsonl")
trial_a_edited=read_generations("/work/gen_6_edited.jsonl")
print("Single-neuron ablation: paired +2.34 percentage points, SE 2.82; not established.")
print("Neuron-22 original and edited column norms:",
      float(candidate_original[:,22].norm()),float(load(candidate_name)[:,22].norm()))

# Analyze existing recorded residuals only; do not run any network components.
pairH=np.stack([H[b].astype(np.float32)-H[g].astype(np.float32) for b,g in mixed])
mean_good=H[good].astype(np.float32).mean(0)
mean_prompt=Hp.astype(np.float32).mean(0)
directions={}
for l in range(4,24):
    d=pairH[:,l].mean(0)
    protect=np.stack([mean_good[l],mean_prompt[l]],axis=1)
    q=np.linalg.qr(protect)[0]
    r=d-q@(q.T@d)
    r/=np.linalg.norm(r)
    directions[l]=torch.tensor(r)
    v=pairH[:,l]@r
    a=np.load(f"/work/fwd_3/hidden_{l}.npy").astype(np.float32) if l in [4,8,12,15,16,17,18,19,20,21,22,23] else None
    probe="" if a is None else "probe "+str(np.round([a[0,80:180]@r,a[1,330:450]@r],3).mean(axis=1).tolist())
    print("layer",l,"paired projection",round(float(v.mean()),3),
          "positive_pairs",int((v>0).sum()),"SE",round(float(v.std()/math.sqrt(len(v))),3),
          "retained contrast",round(float(np.linalg.norm(d-q@(q.T@d))/np.linalg.norm(d)),3),probe)
print("Neighbor direction cosines:",[(l,round(float(directions[l]@directions[l+1]),3)) for l in range(14,23)])

for l in [15,17,18,19,20,21,22,23]:
    r=directions[l]
    p=f"model.language_model.layers.{l}."
    comps=[("MLP","mlp.down_proj.weight")]
    comps += [("attention","self_attn.o_proj.weight")] if l in [3,7,11,15,19,23] else [("DeltaNet","linear_attn.out_proj.weight")]
    for label,suffix in comps:
        w=weights[p+suffix]
        alignment=(r@w).square().sum()/w.square().sum()
        print("projection removal",l,label,"relative_Frobenius",round(float(alignment.sqrt()),4))

with open("/work/notes.md","a") as f:
    f.write("\nTrial A measured on 64 problems x2, paired: original 8.59%, edited 10.94%, difference +2.34pp +/-2.82pp SE; cutoff 92.19% ->91.41%. Insufficient evidence. Official passes used 4/25, 21 remain. Investigating a distributed repeat-specific residual direction, orthogonal to mean successful and prompt activations.\n")
```
