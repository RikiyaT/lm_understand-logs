```python
# Hypothesis: the four-neuron ablation should reduce repetitive thinking; recover its paired results before choosing the next dose.
print("Current edits:", edited())
print("Circuit targets:", targets if "targets" in globals() else "not in kernel")
for filename in ["gen_4_original.jsonl", "gen_4_edited.jsonl"]:
    rows = [json.loads(s) for s in open("/work/"+filename)]
    flat = [dict(id=r["id"],source=r["source"],**s) for r in rows for s in r["samples"]]
    globals()[filename.split(".")[0]] = flat
    print(filename,"n",len(flat),"correct",sum(r["correct"] for r in flat),
          "cut",sum(r["cut"] for r in flat),"mean_tokens",round(np.mean([r["tokens"] for r in flat])))
    print("By source:",{src:(sum(r["correct"] for r in flat if r["source"]==src),
                              sum(r["source"]==src for r in flat)) for src in ["gsm8k","math"]})
orig4, edit4 = gen_4_original, gen_4_edited
d=np.array([int(b["correct"])-int(a["correct"]) for a,b in zip(orig4,edit4)])
print("Paired gain",round(float(d.mean()),4),"SE",round(float(d.std(ddof=1)/np.sqrt(len(d))),4),
      "rescued",int((d==1).sum()),"harmed",int((d==-1).sum()))
print("Circuit helper:",getattr(apply_circuit_strength,"__doc__",None))
print("Causal sample IDs:",json.dumps([r["id"] for r in orig4]))
```
