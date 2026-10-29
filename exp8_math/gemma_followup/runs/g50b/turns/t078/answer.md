The matched-control combination gained only 0.88 points, compared with 3.61 points pooled for the selected group. That is suggestive of specificity, but the uncertainties overlap, so I will not claim a decisive mechanistic separation. I will restore the selected candidate and use the final two DEV passes only for replication, with no further tuning. The candidate will then be frozen before any HOLDOUT outcomes are examined.

```python
manifest33_head=set_head_direction(alpha=0.0,gain=1.1)
new11=D11.clone(); new11[:,robust11]*=0.5
save_edit(mlp11name,new11)
finalist_manifest={"components":[manifest33_head,manifest26]}
log_pass(33,"Finalist replication, DEV order rotated by8, no further tuning planned",finalist_manifest)
def loop_share(k,which):
    flat=[s for r in read_jsonl(W/f"gen_{k}_{which}.jsonl") for s in r["samples"]]
    flags=[]
    for s in flat:
        lines=[re.sub(r"\s+"," ",x).strip() for x in s["text"].splitlines()]
        counts=collections.Counter(x for x in lines if len(x)>=20)
        flags.append(any(c>=3 for c in counts.values()))
    return {"accuracy":np.mean([s["correct"] for s in flat]),
            "mean_tokens":np.mean([s["tokens"] for s in flat]),
            "cut_share":np.mean([s["cut"] for s in flat]),
            "repeated_line_share":np.mean(flags)}
for k in [29,30,31,32]:
    a=loop_share(k,"original"); b=loop_share(k,"edited")
    print("BEHAVIOR",k,{x:round(float(b[x]-a[x]),4) for x in a})
with (W/"notes.md").open("a") as f:
    f.write("\nPass31 head1.1 alone +0.59pp SE2.12; "
            "pass32 head1.1 with matched-control MLP half-suppression +0.88pp SE1.79. "
            "Selected combination passes29,30 pooled +3.613pp, sampleSE1.210pp, "
            "problem-clusterSE1.964pp. Evidence suggestive, not conclusive. "
            "Final two DEV passes33,34 will replicate selected combination only; "
            "no new strengths/components. Freeze before pass35 HOLDOUT.\n")
print("Selected candidate restored exactly; pass33 ready.")
```