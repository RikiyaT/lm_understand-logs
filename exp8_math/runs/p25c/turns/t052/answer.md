Ablating layer 23 head 5 lowered both texts’ overall likelihoods by similar amounts, so the initial summary does not show the selectivity seen for layer 15 head 7. I will check the reasoning-only regions before drawing that conclusion, and I will not add this head merely because its attention follows repeated passages. The next isolated candidate is layer 11 head 1, which showed strong retrieval of earlier matching continuations. This comparison should help separate useful copying heads from heads that specifically sustain the pathological loop.
```python
# Hypothesis: L11H1, unlike the less selective L23H5 candidate, may be an upstream contributor to repetitive copying.
m18=probe_region_metrics("fwd_18")
base=dose_results[1.0]
print("L23H5-only logprob changes:",{k:round(m18[k]-base[k],5) for k in base})
print("Absolute region logprobs:",{k:round(v,5) for k,v in m18.items()})
z18=np.load(W/"fwd_18"/"mlp_act_23.npy",mmap_mode="r")
print("Loop neuron means after L23H5 ablation:",
      {j:round(float(z18[1,start_eval:end_eval,j].mean()),3) for j in (22,94,317,495)})
current_recipe=install_head_scales([(11,1,0.0)])
(W/"current_recipe.json").write_text(json.dumps(current_recipe))
req13={"type":"teacher_forced","model":"edited","texts":probe_texts,
       "return":{"token_logprobs":True,"topk":20,"hidden_states":[12,24],"mlp_act":[23]}}
(W/"pass13_request.json").write_text(json.dumps(req13))
with (W/"notes.md").open("a") as f:
    f.write("Pass 12/25 fwd_18: isolated L23H5=0 region logprob deltas "+
            json.dumps({k:m18[k]-base[k] for k in base})+
            ". Not accepted; no accuracy test or selective-effect claim from full-text averages.\n")
    f.write("NEXT PASS 13/25: teacher-forced raw probe /work/pass13_request.json. "
            "Restore L23 original. ONLY edit model.language_model.layers.11.self_attn.o_proj.weight "
            "columns [256:512] *=0 (head1 ablated). All other values original. "
            "Test upstream head identified by repetition-retrieval attention 0.743 vs 0.139 productive.\n")
print("Next isolated recipe:",current_recipe)
```
