The original-model measurements are available, including a modular-arithmetic example where productive repetition is already highly predictable. That is a useful control against mistaking predictability itself for a failure. I will record reasoning-only and numeric-token scores, then evaluate the identical texts with the frozen head ablation. Preserved numeric calculations together with selectively weakened stalled loops would support the proposed mechanism beyond the original speed problem.
```python
# Hypothesis: L15H7 ablation preserves numeric progress even where legitimate reasoning uses repeated sentence structure.
def broad_metrics(folder):
    p=W/folder
    lp=np.load(p/"token_logprobs.npy")
    ids=np.load(p/"input_ids.npy")
    tops=np.load(p/"topk_ids.npy")
    result=[]
    for i,r in enumerate(broad_regions):
        q=np.arange(r["branch"],r["length"])
        num=np.array([t for t in q if re.fullmatch(r"\s*\d+\s*",decode_ids([ids[i,t]]))],dtype=int)
        result.append({"case":r["case"],"kind":r["kind"],
                       "body_logprob":float(lp[i,q-1].mean()),
                       "late_logprob":float(lp[i,r["late"]-1:r["length"]-1].mean()),
                       "numeric_logprob":float(lp[i,num-1].mean()) if len(num) else None,
                       "numeric_top1":float((tops[i,num-1,0]==ids[i,num]).mean()) if len(num) else None,
                       "numeric_tokens":len(num)})
    return result
broad_base=broad_metrics("fwd_25")
print("Original broad-probe metrics:")
for r in broad_base:
    print({k:round(v,4) if isinstance(v,float) else v for k,v in r.items()})
assert edited()==["model.language_model.layers.15.self_attn.o_proj.weight"]
assert (load(edited()[0])[:,1792:2048]==0).all()
req19=dict(broad_request,model="edited")
(W/"pass19_request.json").write_text(json.dumps(req19))
with (W/"notes.md").open("a") as f:
    f.write("Pass18/25 fwd_25: original six-text broad probe baseline saved. "
            "All texts remain inside thinking; modular productive control uses twelve progressing repeated-format steps.\n")
    f.write("NEXT PASS19/25: /work/pass19_request.json same six raw texts, model=edited. "
            "Exact frozen edit ONLY L15 self_attn.o_proj.weight columns [1792:2048] zero, all other weights original. "
            "Compare body and numeric-token likelihoods with fwd_25; no tuning from these probes.\n")
```
