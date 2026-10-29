The generation file is grouped by problem, so my first inspection stopped before reaching the answer texts; no experiment or edit was affected. I’ll flatten the samples and lock DEV to 32 intermittently solved problems, favoring those with the most informative mix of successes and failures. Then I’ll inspect actual failures and compare their MLP activity only within those fixed DEV problems.

```python
print("Sample schema:", {k: type(v).__name__
                         for k, v in baseline[0]["samples"][0].items()})
print("Sample scalar fields:", {
    k: v for k, v in baseline[0]["samples"][0].items()
    if not isinstance(v, (list, dict)) and len(str(v)) < 150
})

rates = [(r["id"], np.mean([s["correct"] for s in r["samples"]]))
         for r in baseline]
eligible = [(pid, float(p)) for pid, p in rates if 0 < p < 1]
eligible.sort(key=lambda x: (-x[1] * (1 - x[1]), x[0]))
dev_ids = [pid for pid, p in eligible[:32]]
dev_set = set(dev_ids)
assert not dev_set & holdout_set
(work / "DEV_ids.json").write_text(json.dumps(dev_ids))
with (work / "notes.md").open("a") as f:
    f.write("\nDEV LOCKED before edits: DEV_ids.json, 32 problems selected by "
            "highest p*(1-p) among intermittent diagnostic successes. "
            "HOLDOUT unchanged and unexamined.\n")

flat = [dict(s, problem_id=r["id"])
        for r in baseline for s in r["samples"]]
print("DEV:", json.dumps(dev_ids))
print("Intermittent problems:", len(eligible),
      "DEV baseline accuracy:", round(np.mean([dict(rates)[p] for p in dev_ids]), 4))
print("DEV subjects:", dict(Counter(by_id[p]["subject"] for p in dev_ids)))

answer_key = next(k for k, v in flat[0].items()
                  if isinstance(v, str) and len(v) > 180)
shown = 0
for s in flat:
    if s["problem_id"] in dev_set and not s["correct"]:
        t = s[answer_key]
        print(json.dumps({
            "id": s["problem_id"],
            "fields": {k: v for k, v in s.items()
                       if k != answer_key and len(str(v)) < 120},
            "excerpt": t[:140] + " ... " + t[-140:]
        }, ensure_ascii=True))
        shown += 1
        if shown == 8:
            break
"DEV fixed; no weight edits."
```