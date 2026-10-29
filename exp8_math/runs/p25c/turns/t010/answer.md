```python
# Hypothesis: layer 23 neuron 22 is activated specifically as repeated checking accumulates and writes a self-reinforcing continuation direction.
with open("/work/fwd_3/meta.json") as f:
    tfmeta=json.load(f)
print("Teacher-forced metadata:",str(tfmeta)[:1200])
ids2=np.load("/work/fwd_3/input_ids.npy")
lp2=np.load("/work/fwd_3/token_logprobs.npy")
tfM={l:np.load(f"/work/fwd_3/mlp_act_{l}.npy").astype(np.float32) for l in range(14,24)}
for l,j in [(23,22),(23,94),(22,59),(23,19),(17,20),(15,0),(18,29)]:
    a=tfM[l]
    print("Neuron",l,j,"first shared / good continuation / first repeat / late repeats",
          [round(float(v),3) for v in [a[0,80:180,j].mean(),a[0,210:260,j].mean(),a[1,210:270,j].mean(),a[1,330:450,j].mean()]])
print("Relevant token positions:")
for n in [50,75,100,150,180,200,210,230,260,290,330,390,440]:
    print(n,repr(tok.decode(ids2[1,n:n+12].tolist())[:90]))
print("Largest neuron changes in late loop versus initial shared reasoning:")
tf_rank=[]
for l,a in tfM.items():
    d=a[1,330:450].mean(0)-a[0,80:180].mean(0)
    for j in np.argsort(d*down_norm[l])[-5:]:
        tf_rank.append((float(d[j]*down_norm[l,j]),l,int(j),float(d[j])))
print(sorted(tf_rank,reverse=True)[:20])

E=load_original("model.language_model.embed_tokens.weight")
normname="model.language_model.norm.weight"
finalnorm=load_original(normname)
print("Final norm min/median/max:",[float(finalnorm.min()),float(finalnorm.median()),float(finalnorm.max())])
safe_words=["Wait"," Wait","wait"," wait","No"," No","Yes"," Yes","But"," But","So"," So","Therefore"," Therefore","Thus"," Thus","However"," However","Check"," Check","Correct"," Correct","Answer"," Answer","Finally"," Finally","The"," The","Let"," Let","I"," I","we"," we","if"," if","is"," is","think"," think","calculate"," calculate","again"," again","then"," then","because"," because","\n","\n\n",".",",","=","+","-","0","1","2","3","4","5","6","7","8","9","</think>","<think>","\\boxed"]
safe_ids={}
for s in safe_words:
    z=tok.encode(s).ids
    if len(z)==1:
        safe_ids[z[0]]=s
si=torch.tensor(list(safe_ids))
for l,j in [(23,22),(23,94),(23,19),(22,59),(15,0)]:
    d=weights[f"model.language_model.layers.{l}.mlp.down_proj.weight"][:,j]
    # Weight-space alignment only; no model computation is performed here.
    proj=E[si] @ (d*(1+finalnorm))
    order=torch.argsort(proj)
    print("Output direction",l,j,"positive",[(safe_ids[int(si[t])],round(float(proj[t]),4)) for t in order[-6:].flip(0)],
          "negative",[(safe_ids[int(si[t])],round(float(proj[t]),4)) for t in order[:4]])
```
