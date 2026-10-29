The first held-out test improved accuracy from 16/128 to 28/128, a paired gain of 9.38 points with a 3.91-point standard error. Both GSM8K and MATH improved, which is encouraging, but the estimate is still fairly uncertain. I will leave the weights frozen and test the other 128 problems from the preselected held-out set. Pooling two disjoint paired tests will provide a more reliable estimate without selecting a new edit on the validation outcomes.
```python
# Hypothesis: the frozen L15H7 repair gives a reproducible accuracy gain across independently sampled practice problems.
result20,ma20,mb20,keys20=paired_summary(20)
print("Held-out H1A:",result20)
for source in ("gsm8k","math"):
    kk=[k for k in keys20 if ma20[k]["source"].lower()==source]
    d=np.array([int(mb20[k]["correct"])-int(ma20[k]["correct"]) for k in kk])
    print(source,{"n":len(kk),"original":sum(ma20[k]["correct"] for k in kk),
                  "edited":sum(mb20[k]["correct"] for k in kk),
                  "gain_pp":round(float(100*d.mean()),3),
                  "SE_pp":round(float(100*d.std(ddof=1)/len(d)**.5),3)})
for label,ss in (("original",list(ma20.values())),("edited",list(mb20.values()))):
    print(label,{"repeat_fraction":round(float(np.mean([repeated_line_fraction(s["text"]) for s in ss])),3),
                 "median_correct_tokens":float(np.median([s["tokens"] for s in ss if s["correct"]])),
                 "finished_correct":sum(s["correct"] and not s["cut"] for s in ss)})
validation_runs=[20]
(W/"validation_runs.json").write_text(json.dumps(validation_runs))
req15={"type":"generate","model":"both","problems":H1B,"n":1,"max_tokens":32768}
(W/"pass15_request.json").write_text(json.dumps(req15))
with (W/"notes.md").open("a") as f:
    f.write("Pass 14/25 gen_20: FROZEN L15H7=0 on untouched H1A128, "+json.dumps(result20)+
            ". GSM 6->13/64; MATH10->15/64. Positive first holdout, need replication for precision. "
            "No other active edits. Candidate remains snapshot /work/candidate_L15H7_zero.\n")
    f.write("NEXT PASS 15/25: /work/pass15_request.json, both H1B128 (disjoint complementary64GSM+64MATH), "
            "n=1 max_tokens32768. Exact edit unchanged ONLY model.language_model.layers.15.self_attn.o_proj.weight "
            "columns [1792:2048] zero. Candidate frozen; validation used for confirmation, not strength selection.\n")
print("H1B IDs:",json.dumps(H1B))
```
