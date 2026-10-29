The amplification pass completed, so I will compare its reasoning-token likelihoods with the original model before drawing conclusions. I will then restore the frozen head-7 ablation exactly and use all four remaining passes for fresh paired accuracy measurements. The causal probes explain a plausible mechanism, but the edit still needs independent evidence that more problems are answered correctly.

```python
# Hypothesis: H7 amplification reinforces loops; the frozen ablation must independently improve held-out accuracy.
amp_meta = json.loads((work / "fwd_29" / "meta.json").read_text())
assert amp_meta["texts"] == base_meta["texts"], "Exclude any nonidentical probe before comparison."
amp_lp = np.load(work / "fwd_29" / "token_logprobs.npy")
amplification_results = []
for i, reg in enumerate(regions):
    row = {"case": reg["case"], "kind": reg["kind"]}
    for label, start in (("body", reg["branch"]), ("late", reg["late"])):
        sl = slice(start - 1, reg["length"] - 1)
        row[label + "_delta"] = float(np.mean(amp_lp[i, sl] - base_lp[i, sl]))
    amplification_results.append(row)
print("H7 x1.5 versus original:", json.dumps(amplification_results, indent=2))
(work / "amplification_results.json").write_text(json.dumps(amplification_results, indent=2))

frozen_w = original_w.clone()
frozen_w[:, 1792:2048] = 0
save_edit(target_name, frozen_w)
assert edited() == [target_name]
assert torch.equal(load(target_name), frozen_w)
frozen_recipe = json.loads((work / "frozen_final_candidate_recipe.json").read_text())
(work / "current_recipe.json").write_text(json.dumps(frozen_recipe))

all_fresh_ids = holdout_info["H2"] + holdout_info["H3"] + holdout_info["H4"]
assert len(all_fresh_ids) == len(set(all_fresh_ids)) == 768
assert set(all_fresh_ids).isdisjoint(holdout_info["H1"])
assert set(all_fresh_ids).isdisjoint(json.loads((work / "dev_ids.json").read_text()))
assert set(h2a_ids + h2b_ids) == set(holdout_info["H2"])

sample_record = json.loads((work / "gen_20_original.jsonl").open().readline())
print("Generation record fields:", list(sample_record))
print("Short metadata:", {k: v for k, v in sample_record.items()
                         if isinstance(v, (bool, int, float)) or
                         (isinstance(v, str) and len(v) < 70)})
with (work / "notes.md").open("a") as f:
    f.write("\nPass21/25 fwd_29: temporary L15H7 x1.5, all six texts verified exact. Deltas " + json.dumps(amplification_results) + "\n")
    f.write("RESTORED frozen candidate: ONLY L15 o_proj columns1792:2048 zero; all other values original. Remaining4 passes reserved for H2A128,H2B128,H3 256,H4 256 paired accuracy. All768 fresh IDs disjoint and unused. No further tuning.\n")
print("Frozen candidate restored and verified. Passes used: 21/25.")
```
