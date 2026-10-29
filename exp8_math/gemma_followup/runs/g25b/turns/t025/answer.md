Success-associated neurons have correlations of similarly modest size, so simply switching the sign of the same selection rule is not compelling. I’ll investigate a more concrete failure seen in the texts: repeatedly reusing an intermediate quantity as the final answer instead of completing the calculation. I’ll select a correct and an incorrect continuation for the same algebra problem and inspect full-attention behavior. A head that disproportionately retrieves the mistaken intermediate value would provide a more specific intervention target.

```python
p="math_train_algebra_380"
ss=next(r["samples"] for r in base if r["id"]==p)
print("Problem:",problems_by_id[p]["problem"])
for ok in [True,False]:
    candidates=sorted([s for s in ss if s["correct"]==ok],key=lambda s:len(s["text"]))
    for s in candidates[:2]:
        print("Correct",ok,"tokens",s["tokens"],"characters",len(s["text"]),
              "excerpt",repr(s["text"][:290]))
print("Reference:",problems_by_id[p]["solution"][:1300])
print("Full-attention weight shapes:",{n:tuple(load_original(n).shape) for n in NAMES
                                     if any(f"layers.{l}.self_attn" in n for l in [5,11,17])
                                     and n.endswith("o_proj.weight")})
with open("/work/notes.md","a") as f:
    f.write("\nNext diagnostic hypothesis: erroneous reuse of intermediate values / repeated conclusions, inspect full-attention head routing on matched algebra380 continuations. Candidate A amplification remains saved but is not accepted as a repair. No HOLDOUT outcomes inspected.\n")
```
