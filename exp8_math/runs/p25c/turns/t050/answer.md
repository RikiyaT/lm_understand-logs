Full ablation improved development accuracy from 8/64 to 17/64: a paired gain of 14.06 points with a 5.84-point standard error. Together with the likelihood dose-response, that makes layer 15 head 7 a promising candidate, but it still needs untouched-problem validation. I will preserve this candidate and test layer 23 head 5 separately, since its attention also strongly favors repeated continuations. Testing it alone will reveal whether it reinforces the loop or merely attends to it while performing a useful corrective function.
```python
# Hypothesis: late attention head L23H5 is a second causal copying contributor, distinguishable from the validated-in-development L15H7.
import shutil
result17,ma17,mb17,keys17=paired_summary(17)
print("Full L15H7 ablation:",result17)
for label,ss in (("original",list(ma17.values())),("ablated",list(mb17.values()))):
    print(label,{"mean_repeat_fraction":round(float(np.mean([repeated_line_fraction(s["text"]) for s in ss])),3),
                 "median_correct_tokens":float(np.median([s["tokens"] for s in ss if s["correct"]])),
                 "mean_tokens":float(np.mean([s["tokens"] for s in ss]))})
shutil.copytree(W/"edited",W/"candidate_L15H7_zero",dirs_exist_ok=True)
(W/"candidate_L15H7_zero_recipe.json").write_text(json.dumps(current_recipe))
used_ids={r["id"] for r in gen_data["gen_1"]}
rng=np.random.default_rng(20261024)
gsm_pool=sorted(k for k,v in practice.items() if v["source"].lower()=="gsm8k" and k not in used_ids)
math_pool=sorted(k for k,v in practice.items() if v["source"].lower()=="math" and k not in used_ids)
rng.shuffle(gsm_pool); rng.shuffle(math_pool)
holdout_sets={}
for j in range(4):
    holdout_sets[f"H{j+1}"]=gsm_pool[128*j:128*(j+1)]+math_pool[128*j:128*(j+1)]
(W/"holdout_sets.json").write_text(json.dumps(holdout_sets))
print("Reserved untouched sets:",{k:len(v) for k,v in holdout_sets.items()})
current_recipe=install_head_scales([(23,5,0.0)])
(W/"current_recipe.json").write_text(json.dumps(current_recipe))
req12={"type":"teacher_forced","model":"edited","texts":probe_texts,
       "return":{"token_logprobs":True,"topk":20,"hidden_states":[24],"mlp_act":[23]}}
(W/"pass12_request.json").write_text(json.dumps(req12))
with (W/"notes.md").open("a") as f:
    f.write("Pass 11/25 gen_17: full L15H7 ablation, dev64 n=1, "+json.dumps(result17)+
            ". Promising development effect, NOT held-out validated. Snapshot /work/candidate_L15H7_zero, "
            "recipe /work/candidate_L15H7_zero_recipe.json.\n")
    f.write("Reserved H1-H4 in /work/holdout_sets.json: four disjoint 256-problem sets, each 128 GSM+128 MATH, "
            "seed20261024, excluding all128 baseline problems. No outcomes inspected.\n")
    f.write("NEXT PASS 12/25: teacher-forced raw probe /work/pass12_request.json. Restore L15 original; "
            "ONLY edit model.language_model.layers.23.self_attn.o_proj.weight columns [1280:1536] *=0 "
            "(head5 ablated). All other values original. Test second copying candidate independently.\n")
print("Current isolated intervention:",current_recipe)
```
