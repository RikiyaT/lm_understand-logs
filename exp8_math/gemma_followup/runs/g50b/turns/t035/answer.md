Strengthening head 2 hurt accuracy by 2.83 points and more than tripled the cut-off share. That refutes the simple “more operand retrieval is better” hypothesis; excessive retrieval may instead keep restarting or replaying the calculation. I’ll inspect those failures and test attenuation of the same head. An accuracy improvement under attenuation—not just fewer cut-offs—would support a more precise repair.

```python
print("Head-2 amplification behavior:", {
    m:behavior_stats(read_gen(14,m)) for m in ("original","edited")})
r14o = {r["id"]:r for r in read_gen(14,"original")}
r14e = {r["id"]:r for r in read_gen(14,"edited")}
losses = sorted(dev_ids, key=lambda p:
    np.mean([s["correct"] for s in r14e[p]["samples"]])
    -np.mean([s["correct"] for s in r14o[p]["samples"]]))
for pid in losses[:3]:
    for model,rows in [("original",r14o),("edited",r14e)]:
        s = next((s for s in rows[pid]["samples"] if not s["correct"]),
                 rows[pid]["samples"][0])
        t=s["text"]
        print(json.dumps({"id":pid,"model":model,"tokens":s["tokens"],
                          "excerpt":t[:140]+" ... "+t[-140:]},ensure_ascii=True))
candidate = prepare_head(15,11,2,0.5,
    "Opposite-direction intervention: amplification increased repetition/cutoffs "
    "and reduced accuracy; test whether attenuating head 2 prevents over-retrieval "
    "or repeated calculation while preserving correct answers.")
print(candidate)
"Head-2 attenuation prepared."
```