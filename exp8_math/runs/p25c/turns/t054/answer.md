The layer-11 intervention also looks unhelpful in the aggregate: productive-text likelihood fell while repeated-text likelihood rose. I will quantify the reasoning regions, then restore the simpler layer-15-only candidate rather than combining heads with contrary evidence. That candidate is now fixed for an initial held-out test on 128 untouched problems, balanced between GSM8K and MATH. A clear paired accuracy gain here would be much stronger evidence than the development results used to select it.
```python
# Hypothesis: the selective L15H7 repair generalizes to unseen math problems; correlated L11H1 and L23H5 ablations should be rejected.
m19=probe_region_metrics("fwd_19")
print("L11H1-only region logprob deltas:",{k:round(m19[k]-base[k],5) for k in base})
current_recipe=install_head_scales([(15,7,0.0)])
(W/"current_recipe.json").write_text(json.dumps(current_recipe))
H1A=holdout_sets["H1"][:64]+holdout_sets["H1"][128:192]
H1B=holdout_sets["H1"][64:128]+holdout_sets["H1"][192:256]
(W/"H1A_ids.json").write_text(json.dumps(H1A))
(W/"H1B_ids.json").write_text(json.dumps(H1B))
req14={"type":"generate","model":"both","problems":H1A,"n":1,"max_tokens":32768}
(W/"pass14_request.json").write_text(json.dumps(req14))
print("H1A counts:",dict(Counter(practice[k]["source"] for k in H1A)),
      "math levels:",dict(Counter(str(practice[k].get("level")) for k in H1A if practice[k]["source"].lower()=="math")))
print("H1A IDs:",json.dumps(H1A))
with (W/"notes.md").open("a") as f:
    f.write("Pass 13/25 fwd_19: isolated L11H1=0 region logprob deltas "+
            json.dumps({k:m19[k]-base[k] for k in base})+
            ". Rejected as nonselective/wrong-direction. L23H5 also rejected: arithmetic -0.0491 vs loop -0.0040.\n")
    f.write("NEXT PASS 14/25: first untouched confirmation, both H1A128 n=1 max_tokens32768; "
            "exact /work/pass14_request.json and /work/H1A_ids.json. Balanced64 GSM+64 MATH, excludes every "
            "baseline/development problem. Freeze candidate: ONLY L15 self_attn.o_proj.weight columns "
            "[1792:2048] zero (H7); all else original. Candidate selection finished before this validation.\n")
```
