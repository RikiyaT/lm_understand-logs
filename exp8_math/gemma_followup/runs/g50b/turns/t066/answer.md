Half-suppression gained 2.44 points with a 1.68-point standard error, better than complete removal and without a substantial length change. That makes it a plausible candidate, but one favorable pass is insufficient. I will archive it and replicate it unchanged before combining it with any attention edit. A second positive result would strengthen the case for a moderate, localized adjustment rather than wholesale neuron removal.

```python
archive_current("candidate_matched11_half",manifest26)
def pooled_pairs(passes):
    groups=collections.defaultdict(list); oo=[]; ee=[]; all_d=[]
    for k in passes:
        orig={r["id"]:r for r in read_jsonl(W/f"gen_{k}_original.jsonl")}
        edit={r["id"]:r for r in read_jsonl(W/f"gen_{k}_edited.jsonl")}
        for pid in orig:
            a=np.array([s["correct"] for s in orig[pid]["samples"]],float)
            b=np.array([s["correct"] for s in edit[pid]["samples"]],float)
            assert len(a)==len(b)
            groups[pid].extend((b-a).tolist())
            oo.extend(a); ee.extend(b); all_d.extend(b-a)
    d=np.array(all_d)
    gm=np.array([np.mean(v) for v in groups.values()])
    return {"passes":passes,"answers_per_model":len(d),"problems":len(groups),
            "original_accuracy":float(np.mean(oo)),"edited_accuracy":float(np.mean(ee)),
            "gain_pp":100*float(d.mean()),
            "sample_SE_pp":100*float(d.std(ddof=1)/np.sqrt(len(d))),
            "problem_cluster_SE_pp":100*float(gm.std(ddof=1)/np.sqrt(len(gm)))}
print("HALF L11",summarize_pair(26))
print("EARLIER COMBINATION POOLED",pooled_pairs([17,18]))
log_pass(27,"Unchanged half-suppression replication, reversed DEV order",manifest26)
print("Candidate archived and pass27 logged.")
```