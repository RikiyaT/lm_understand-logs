The stronger dose has a much larger effect on the repeated text than on the productive text, which is promising but still only a controlled-probe result. I will inspect the reasoning-only changes and then test this dose on the development set, not on fresh validation problems yet. If it mainly reduces repetition while increasing correct answers, it can advance to a new held-out test. If it disrupts arithmetic or produces shorter but wrong answers, I will reject it.
```python
# Hypothesis: half-magnitude reversal of L15H7 suppresses copying feedback more strongly than ablation while sparing productive math.
m22=probe_region_metrics("fwd_22")
print("Reversed-head region logprob deltas:",{k:round(m22[k]-base[k],5) for k in base})
dose_results[-0.5]=m22
(W/"head15_7_dose.json").write_text(json.dumps(dose_results))
z22=np.load(W/"fwd_22"/"mlp_act_23.npy",mmap_mode="r")
print("Loop-associated neuron means:",
      {j:round(float(z22[1,start_eval:end_eval,j].mean()),3) for j in (22,94,317,495,162)})
req17={"type":"generate","model":"both","problems":dev_ids,"n":1,"max_tokens":32768}
(W/"pass17_request.json").write_text(json.dumps(req17))
with (W/"notes.md").open("a") as f:
    f.write("Pass 16/25 fwd_22 L15H7=-0.5: full-text logprob productive -1.50952, repeated -0.53682; "
            "reasoning-region deltas "+json.dumps({k:m22[k]-base[k] for k in base})+
            ". Stronger repetition-specific effect, not yet accuracy evidence.\n")
    f.write("NEXT PASS 17/25: exact /work/pass17_request.json, both dev64 n=1 max_tokens32768. "
            "Only edit L15 self_attn.o_proj.weight columns [1792:2048] *= -0.5, all else original. "
            "Development screening only. H2-H4 remain untouched.\n")
print("Recorded development trial:",current_recipe)
```
