Generation is working again. The half-strength edit improved development accuracy from 9/64 to 14/64, but the paired gain of 7.81 percentage points has a 6.79-point standard error, so it is not convincing yet. I will inspect whether the rescued answers still contain substantive reasoning and then test full ablation on the same development problems with two samples each. A stronger reduction in repetition accompanied by a clearer accuracy gain would support the mechanism; deterioration would favor backing off.
```python
# Hypothesis: stronger suppression of L15H7 will reduce looping further and improve accuracy, while preserving substantive reasoning.
def read_generation(k,model):
    return [json.loads(line) for line in (W/f"gen_{k}_{model}.jsonl").open() if line.strip()]
def paired_summary(k):
    aa,bb=read_generation(k,"original"),read_generation(k,"edited")
    fa,fb=flatten(aa),flatten(bb)
    ma={(s["problem_id"],s["sample_idx"]):s for s in fa}
    mb={(s["problem_id"],s["sample_idx"]):s for s in fb}
    keys=sorted(set(ma)&set(mb))
    ds=np.array([int(mb[x]["correct"])-int(ma[x]["correct"]) for x in keys],float)
    per_problem={}
    for key,d in zip(keys,ds): per_problem.setdefault(key[0],[]).append(d)
    di=np.array([np.mean(v) for v in per_problem.values()])
    result={"n":len(keys),"problems":len(di),
            "original_correct":sum(ma[x]["correct"] for x in keys),
            "edited_correct":sum(mb[x]["correct"] for x in keys),
            "gain_pp":float(100*ds.mean()),
            "SE_sample_pp":float(100*ds.std(ddof=1)/len(ds)**.5),
            "SE_problem_pp":float(100*di.std(ddof=1)/len(di)**.5),
            "wrong_to_right":int((ds==1).sum()),"right_to_wrong":int((ds==-1).sum())}
    return result,ma,mb,keys
def repeated_line_fraction(text):
    lines=[s.strip() for s in text.splitlines() if len(s.strip())>=12]
    if not lines: return 0.0
    c=Counter(lines)
    return sum(len(s)*(n-1) for s,n in c.items())/sum(len(s)*n for s,n in c.items())
result15,ma15,mb15,keys15=paired_summary(15)
print("Half-strength paired result:",result15)
for label,ss in (("original",list(ma15.values())),("half-strength",list(mb15.values()))):
    print(label,{"mean_repeat_fraction":round(float(np.mean([repeated_line_fraction(s["text"]) for s in ss])),3),
                 "median_correct_tokens":float(np.median([s["tokens"] for s in ss if s["correct"]])),
                 "mean_tokens":float(np.mean([s["tokens"] for s in ss])),
                 "thinking_closures":sum("</think>" in s["text"] for s in ss)})
rescued=[key for key in keys15 if mb15[key]["correct"] and not ma15[key]["correct"]]
for key in rescued[:4]:
    s=mb15[key]
    print("RESCUED",key,"tokens",s["tokens"],"excerpt",
          (s["text"][:90]+" ... "+s["text"][-180:]).encode("ascii","replace").decode())
current_recipe=install_head_scales([(15,7,0.0)])
(W/"current_recipe.json").write_text(json.dumps(current_recipe))
req11={"type":"generate","model":"both","problems":dev_ids,"n":2,"max_tokens":32768}
(W/"pass11_request.json").write_text(json.dumps(req11))
with (W/"notes.md").open("a") as f:
    f.write("Pass 10/25 gen_15: L15H7=0.5, dev64 n=1 paired result "+json.dumps(result15)+
            ". Gain not significant; no accepted edit. Generation startup now works.\n")
    f.write("NEXT PASS 11/25: exact /work/pass11_request.json, both dev64 n=2 max_tokens32768. "
            "Only edit L15 self_attn.o_proj.weight columns [1792:2048] *=0 (full H7 ablation), all else original. "
            "Purpose: development dose-response and accuracy screening.\n")
print("Installed full-ablation recipe:",current_recipe)
```
