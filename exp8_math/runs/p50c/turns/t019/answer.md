```python
# Hypothesis: removing the loop-writing direction from one MLP has a measurable causal effect; recover paired outcomes before choosing the next intervention.
def read_gen(path):
    return [json.loads(s) for s in open(path)]
def paired_report(k):
    a=read_gen(f"/work/gen_{k}_original.jsonl"); b=read_gen(f"/work/gen_{k}_edited.jsonl")
    da=np.array([np.mean([s["correct"] for s in r["samples"]]) for r in a])
    db=np.array([np.mean([s["correct"] for s in r["samples"]]) for r in b])
    d=db-da
    print("Pair",k,"N",len(d),"original",round(float(da.mean()),4),"edited",round(float(db.mean()),4),
          "difference",round(float(d.mean()),4),"SE",round(float(d.std(ddof=1)/np.sqrt(len(d))),4))
    for label,rows in [("original",a),("edited",b)]:
        ss=[s for r in rows for s in r["samples"]]
        print(label,"cut",round(np.mean([s["cut"] for s in ss]),4),"tokens",round(np.mean([s["tokens"] for s in ss])),
              "by_source",{src:round(np.mean([s["correct"] for r in rows if r["source"]==src for s in r["samples"]]),4) for src in set(r["source"] for r in rows)})
    return a,b
g7o,g7e=paired_report(7)
print("Current relative change:",float((load(target19)-downweights[19]).norm()/downweights[19].norm()))
```
