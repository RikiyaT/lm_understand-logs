The stronger reversal improved development accuracy by 10.94 points, but its 7.08-point standard error is large, and it did not establish an advantage over the simpler ablation. Stronger suppression on one synthetic loop therefore does not justify a more aggressive final edit. I will return to full ablation, freeze that choice, and use the remaining untouched sets for higher-powered confirmation. I will also use separate diagnostic probes to check whether useful multi-step and repetitive mathematical procedures remain intact, without tuning the weights on those results.
```python
# Hypothesis: the simpler L15H7 ablation is the preferable repair; its modest gain needs high-powered fresh confirmation rather than more strength search.
result23,ma23,mb23,keys23=paired_summary(23)
print("Half-reversal development result:",result23)
shutil.copytree(W/"edited",W/"candidate_L15H7_minus_half",dirs_exist_ok=True)
(W/"candidate_L15H7_minus_half_recipe.json").write_text(json.dumps(current_recipe))
current_recipe=install_head_scales([(15,7,0.0)])
(W/"current_recipe.json").write_text(json.dumps(current_recipe))
(W/"frozen_final_candidate_recipe.json").write_text(json.dumps(current_recipe))
H2A=holdout_sets["H2"][:64]+holdout_sets["H2"][128:192]
H2B=holdout_sets["H2"][64:128]+holdout_sets["H2"][192:256]
(W/"H2A_ids.json").write_text(json.dumps(H2A))
(W/"H2B_ids.json").write_text(json.dumps(H2B))
fresh_validation_runs=[]
(W/"fresh_validation_runs.json").write_text(json.dumps(fresh_validation_runs))
req18={"type":"generate","model":"both","problems":H2A,"n":1,"max_tokens":32768}
(W/"pass18_request.json").write_text(json.dumps(req18))
with (W/"notes.md").open("a") as f:
    f.write("Pass 17/25 gen_23 L15H7=-0.5 on dev64: "+json.dumps(result23)+
            ". No established accuracy advantage over simpler ablation despite stronger synthetic-loop suppression. "
            "Negative dose archived /work/candidate_L15H7_minus_half; not selected.\n")
    f.write("FINAL CANDIDATE NOW FROZEN before H2-H4: ONLY model.language_model.layers.15.self_attn.o_proj.weight "
            "columns [1792:2048] set to zero; all other original values retained. Exact /work/frozen_final_candidate_recipe.json. "
            "No further candidate/strength selection planned. Final acceptance contingent on fresh paired accuracy evidence.\n")
    f.write("Remaining8 passes plan: 18/19 H2A/H2B paired generation, 20/21 original/edited broader math probes, "
            "22/23 causal control probes (restore frozen candidate afterward), 24/25 H3/H4 confirmation. "
            "Fresh primary confirmation H2-H4 totals768 new problems; H1 treated as earlier supportive evidence.\n")
    f.write("NEXT PASS18/25 /work/pass18_request.json: both H2A128 balanced64+64, n1 max_tokens32768; "
            "exact frozen L15H7=0 edit above. IDs /work/H2A_ids.json.\n")
print("Fresh confirmation H2A IDs:",json.dumps(H2A))
```
