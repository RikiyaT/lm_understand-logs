```python
# Hypothesis: paired outcomes, not lower repetition alone, determine whether the four-channel ablation is useful.
def flat_records(rs):
    return [dict(s,id=r["id"],source=r["source"],sample=j) for r in rs for j,s in enumerate(r["samples"])]
s4o,s4e = flat_records(a4),flat_records(b4)
def compare_samples(a,b):
    assert [(s["id"],s["sample"]) for s in a] == [(s["id"],s["sample"]) for s in b]
    x=np.array([s["correct"] for s in a],dtype=float)
    y=np.array([s["correct"] for s in b],dtype=float)
    d=y-x
    return dict(n=len(a),original_correct=int(x.sum()),edited_correct=int(y.sum()),
                gain_pp=round(100*d.mean(),3),paired_se_pp=round(100*d.std(ddof=1)/np.sqrt(len(d)),3),
                rescued=int((d==1).sum()),regressed=int((d==-1).sum()),
                original_cut=sum(s["cut"] for s in a),edited_cut=sum(s["cut"] for s in b),
                original_mean_tokens=round(np.mean([s["tokens"] for s in a])),
                edited_mean_tokens=round(np.mean([s["tokens"] for s in b])))
summary4=compare_samples(s4o,s4e)
print("All:",summary4)
for src in sorted({s["source"] for s in s4o}):
    print(src,compare_samples([s for s in s4o if s["source"]==src],[s for s in s4e if s["source"]==src]))
for label,ss in [("original",s4o),("edit_strength_1",s4e)]:
    frac=[repeated_fraction(tok.encode(s["text"]).ids[-2048:]) for s in ss]
    print(label,"tail repetition mean/median:",np.round([np.mean(frac),np.median(frac)],4))
changes=[(sa,sb) for sa,sb in zip(s4o,s4e) if sa["correct"]!=sb["correct"] or (sa["cut"] and not sb["cut"])]
for sa,sb in changes[:8]:
    print(sa["id"],"correct",sa["correct"],"->",sb["correct"],"tokens",sa["tokens"],"->",sb["tokens"],
          "edited tail:",sb["text"][-200:].encode("ascii","replace").decode())
print("Activation index:",open("/work/gen_4_edited_acts/index.json").read()[:1200])
with open("/work/notes.md","a") as f:
    f.write("\nPass 3 (gen_4): ablated down_proj columns L18:8, L19:11, L23:22,40, strength 1. "+str(summary4)+"\n")
```
