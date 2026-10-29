```python
# Hypothesis: the first neuron-ablation trial did not repair repetition; recover its exact intervention and paired transitions before choosing a stronger causal test.
print("Relevant state names:", sorted(k for k in globals() if any(t in k.lower() for t in ["edit","trial","strength","candidate","select","dose","pass","target","head"])))
for name in edited():
    w0=load_original(name)
    w=load(name)
    changed=(w!=w0).any(dim=0)
    js=torch.where(changed)[0]
    factors=((w[:,js]*w0[:,js]).sum(0)/(w0[:,js].square().sum(0)+1e-20))
    print(name,"columns",js.tolist(),"factors",np.round(factors.numpy(),3).tolist(),
          "relative change",float((w-w0).norm()/w0.norm()))
def read_flat(path):
    return [dict(id=r["id"],source=r["source"],answer=r["answer"],**s)
            for r in map(json.loads,open(path)) for s in r["samples"]]
g4o=read_flat("/work/gen_4_original.jsonl")
g4e=read_flat("/work/gen_4_edited.jsonl")
print("Paired transitions:",collections.Counter((a["correct"],b["correct"]) for a,b in zip(g4o,g4e)))
print("Files:",sorted(os.listdir("/work")))
print("Activation files:",os.listdir("/work/gen_4_edited_acts"))
print("Sample trial IDs:",[r["id"] for r in g4o[:5]])
```
