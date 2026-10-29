```python
# Hypothesis: the existing four-neuron intervention is causal only if it reduces repeated thinking and improves paired accuracy; reconstruct its measured result before further edits.
print(Path("/work/notes.md").read_text())
print("Current edits:", edited())
for name in edited():
    w, w0 = load(name), load_original(name)
    changed = (w != w0).any(dim=0) if w.ndim==2 else w != w0
    inds = torch.where(changed)[0]
    print(name, "changed columns:", inds.tolist()[:30])
    if w.ndim==2:
        print("column scale ratios:", [(int(j),round(float((w[:,j]*w0[:,j]).sum()/w0[:,j].square().sum()),4)) for j in inds[:30]])
def flatten_file(path):
    return [dict(id=r["id"],source=r["source"],answer=r["answer"],**s)
            for r in [json.loads(line) for line in open(path)] for s in r["samples"]]
if Path("/work/gen_4_original.jsonl").exists():
    b4=flatten_file("/work/gen_4_original.jsonl")
    e4=flatten_file("/work/gen_4_edited.jsonl")
    d4=np.array([e["correct"]-b["correct"] for b,e in zip(b4,e4)],float)
    print("Pass gen_4:",len(d4),"original accuracy",np.mean([r["correct"] for r in b4]),
          "edited accuracy",np.mean([r["correct"] for r in e4]),
          "paired difference",d4.mean(),"SE",d4.std(ddof=1)/np.sqrt(len(d4)),
          "rescued",sum(d4==1),"lost",sum(d4==-1))
    for label,rows in [("original",b4),("edited",e4)]:
        print(label,"cut",np.mean([r["cut"] for r in rows]),"mean tokens",np.mean([r["tokens"] for r in rows]),
              "tail repetition",np.mean([repeated_fraction(ids_of(r["text"])[-4096:]) for r in rows]))
        print("By source", {s:np.mean([r["correct"] for r in rows if r["source"]==s]) for s in ["gsm8k","math"]})
print("Available run outputs:", sorted(p.name for p in Path("/work").glob("gen*")))
```
