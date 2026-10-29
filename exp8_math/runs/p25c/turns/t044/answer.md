The half-strength edit produces an intermediate likelihood change, between the original model and full ablation, which supports a dose-response rather than an arbitrary perturbation. This remains evidence about repetition, not mathematical accuracy. I will summarize the three conditions and try a smaller paired generation batch while retaining the full 32,768-token limit. If generation works, the result will screen the half-strength edit; any final claim will still require larger, untouched problem sets.
```python
# Hypothesis: copying-head suppression has a graded effect and may improve accuracy without changing the evaluation protocol.
def probe_region_metrics(folder):
    lp=np.load(W/folder/"token_logprobs.npy")
    out={}
    for name,b,a,z in regions:
        out[name]=float(lp[b,a-1:z-1].mean())
    return out
dose_results={1.0:probe_region_metrics("fwd_10"),
              0.5:probe_region_metrics("fwd_14"),
              0.0:probe_region_metrics("fwd_11")}
print("Dose-response, mean logprob by retained head strength:")
for strength,vals in dose_results.items():
    print(strength,{k:round(v,5) for k,v in vals.items()})
(W/"head15_7_dose.json").write_text(json.dumps(dose_results))
req10={"type":"generate","model":"both","problems":dev_ids,"n":1,"max_tokens":32768}
(W/"pass10_request.json").write_text(json.dumps(req10))
with (W/"notes.md").open("a") as f:
    f.write("Pass 9/25 fwd_14: L15H7=0.5, mean full-text logprob productive -1.50511, repeated -0.43328. "
            "Dose-response regions saved /work/head15_7_dose.json; repetition suppression intermediate "
            "between original and ablation. No accepted edit yet.\n")
    f.write("NEXT PASS 10/25 attempt: smaller paired generation, both, dev64 /work/dev_ids.json, n=1, "
            "max_tokens=32768, exact /work/pass10_request.json. Only edit L15 self_attn.o_proj.weight "
            "columns [1792:2048] *=0.5. This is development screening, not held-out confirmation.\n")
print("Current recipe:",json.loads((W/"current_recipe.json").read_text()))
```
