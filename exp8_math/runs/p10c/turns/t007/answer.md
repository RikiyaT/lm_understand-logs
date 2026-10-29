```python
# Hypothesis: exact phrase copying is the common failure mechanism; induction-style full-attention heads are a plausible causal bottleneck.
def ids_of(s):
    t = tok.encode(s)
    return list(t.ids) if hasattr(t, "ids") else list(t)
def repeated_fraction(ids, n=12):
    if len(ids)<n: return 0.
    grams=[tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    return 1-len(set(grams))/len(grams)
reps=[]
for r in flat1:
    ids=ids_of(r["text"])
    reps.append([repeated_fraction(ids[:2048]),repeated_fraction(ids[2048:8192]),repeated_fraction(ids[-4096:])])
reps=np.array(reps)
print("Repeated 12-token fractions: first 2048 / tokens 2048:8192 / last 4096")
for label,mask in [("correct",y1),("failed",~y1)]:
    print(label,np.round(np.mean(reps[mask],axis=0),3))
print("Repeated Wait phrases among failures:",
      sum("Wait," in r["text"][-3000:] for r in flat1 if not r["correct"]),"/",int((~y1).sum()))
down_cache={}
print("\nMLP neurons with largest average residual contribution norms:")
for ai,l in enumerate(idx1["mlp_layers"]):
    W=load_original(f"model.language_model.layers.{l}.mlp.down_proj.weight")
    down_cache[l]=W
    norms=W.norm(dim=0).numpy()
    score=np.abs(mean_bad[ai]-mean_good[ai])*norms
    js=np.argsort(score)[-2:][::-1]
    if l>=8:
        print(l,[(int(j),round(float(score[j]),3),round(float(mean_bad[ai,j]),3),
                  round(float(mean_good[ai,j]),3)) for j in js])
print("\nAttention per-head output-projection norms:")
for l in [3,7,11,15,19,23]:
    W=load_original(f"model.language_model.layers.{l}.self_attn.o_proj.weight").reshape(1024,8,256)
    print(l,np.round(W.square().sum((0,2)).sqrt().numpy(),3).tolist())
```
