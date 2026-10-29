The second held-out batch was much weaker: 18/128 correct for the original versus 20/128 for the edit, a gain smaller than its standard error. That means the earlier improvement is not yet a robust general result, especially for MATH. I will compute the pooled estimate and test one stronger, predefined intervention on the same identified head: reversing its output at half magnitude. This tests whether actively opposing the copying signal helps more than removing it; any revised candidate will need fresh validation, not reuse these outcomes as confirmation.
```python
# Hypothesis: residual copying persists after ablation, and a predefined half-magnitude reversal may suppress the feedback more decisively.
def pooled_results(runs):
    pairs_all=[]
    seen=set()
    for k in runs:
        _,ma,mb,keys=paired_summary(k)
        for key in keys:
            assert key not in seen
            seen.add(key)
            pairs_all.append((ma[key],mb[key]))
    out={}
    for source in ("all","gsm8k","math"):
        pp=[p for p in pairs_all if source=="all" or p[0]["source"].lower()==source]
        d=np.array([int(b["correct"])-int(a["correct"]) for a,b in pp],float)
        out[source]={"n":len(pp),"original_correct":sum(a["correct"] for a,b in pp),
                     "edited_correct":sum(b["correct"] for a,b in pp),
                     "gain_pp":float(100*d.mean()),
                     "SE_pp":float(100*d.std(ddof=1)/len(d)**.5),
                     "wrong_to_right":int((d==1).sum()),"right_to_wrong":int((d==-1).sum())}
    return out,pairs_all
result21,_,_,_=paired_summary(21)
validation_runs=[20,21]
poolH1,pairsH1=pooled_results(validation_runs)
print("Replication H1B:",result21)
print("Pooled frozen-ablation confirmation:",json.dumps(poolH1))
(W/"validation_runs.json").write_text(json.dumps(validation_runs))
(W/"H1_pooled_results.json").write_text(json.dumps(poolH1))
current_recipe=install_head_scales([(15,7,-0.5)])
(W/"current_recipe.json").write_text(json.dumps(current_recipe))
req16={"type":"teacher_forced","model":"edited","texts":probe_texts,
       "return":{"token_logprobs":True,"topk":20,"hidden_states":[16,24],"mlp_act":[23]}}
(W/"pass16_request.json").write_text(json.dumps(req16))
with (W/"notes.md").open("a") as f:
    f.write("Pass 15/25 gen_21, frozen L15H7=0 on H1B128: "+json.dumps(result21)+
            ". Replication weak; pooled H1 (256 independent pairs): "+json.dumps(poolH1)+
            ". Evidence modest, NOT a robust large gain. All comparisons pooled as within-pass paired differences.\n")
    f.write("NEXT PASS 16/25: predefined stronger dose, teacher-forced /work/pass16_request.json. "
            "Only edit L15 self_attn.o_proj.weight columns [1792:2048] *= -0.5 (reverse head7 output "
            "at half its original magnitude), all else original. Test whether active opposition to copying "
            "is selective; if selected later, H2-H4 remain completely untouched for confirmation. "
            "No final norm/embedding/lm_head edits, no loss fitting or training.\n")
print("Stronger causal dose:",current_recipe)
```
