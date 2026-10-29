The final HOLDOUT pass was nearly neutral: +0.10 percentage points with a 0.68-point standard error, which reduces the pooled gain. I’m keeping the 1.5× edit selected before HOLDOUT, with no post-HOLDOUT tuning. I’ll now calculate the final pooled accuracy and problem-cluster uncertainty, save the audit summary, and verify that only the intended four columns remain changed. The final claim will reflect all three confirmation passes, not just the more favorable first two.

```python
assert edit_checksums() == frozen_checksums
assert set(edited()) == set(target_names.values())
final_pool, holdout_pass_stats = pooled_confirmation([8, 9, 10])
final_pool["approx_95pct_ci_pp"] = [
    final_pool["gain_pp"] - 1.96 * final_pool["cluster_se_pp"],
    final_pool["gain_pp"] + 1.96 * final_pool["cluster_se_pp"]
]
final_pool["candidate_frozen_before_holdout"] = True
final_pool["selection_used_holdout"] = False
final_pool["edit_hash"] = "599aba54ec"

audit = []
for layer, columns in target_channels.items():
    name = target_names[layer]
    ow, ew = load_original(name), load(name)
    keep = torch.ones(2048, dtype=torch.bool)
    keep[columns] = False
    assert torch.equal(ow[:, keep], ew[:, keep])
    assert torch.allclose(ew[:, columns], 1.5 * ow[:, columns])
    audit.append({
        "tensor": name, "columns": columns, "multiplier": 1.5,
        "shape": list(ew.shape),
        "changed_values": int(torch.count_nonzero(ew != ow)),
        "relative_frobenius_change": float((ew - ow).norm() / ow.norm())
    })

dev_table = []
for pn, multiplier in [(5, 0.0), (6, 2.0), (7, 1.5)]:
    st, oo, ee, dd = paired_stats(pn)
    dev_table.append({
        "pass": pn, "column_multiplier": multiplier,
        "gain_pp": st["gain_pp"], "cluster_se_pp": cluster_se_pp(oo, dd),
        "original_repeated_conclusion_pct": 100*st["original"]["three_or_more_final_answers"],
        "edited_repeated_conclusion_pct": 100*st["edited"]["three_or_more_final_answers"]
    })
final_report = {
    "submitted_edit": audit,
    "dev_strength_tests": dev_table,
    "holdout_pooled": final_pool,
    "holdout_passes": holdout_pass_stats,
    "limitations": [
        "Accuracy gain is uncertain; report the problem-cluster SE and interval.",
        "Causal repetition evidence applies to the four-channel group, not individually isolated neurons.",
        "Teacher-forced probes lacked evaluation prompts and are local continuation checks only.",
        "Pass2 was malformed and provides no mathematical evidence.",
        "Earlier no_answer_finished counts were erroneous and were excluded from mechanism conclusions."
    ],
    "submission_directory": "/work/edited",
    "backup_directory": "/work/frozen_candidate"
}
(wdir / "FINAL_REPORT.json").write_text(json.dumps(final_report, indent=2))
with open(wdir / "notes.md", "a") as f:
    f.write("\n## FINAL SUBMISSION\n"
            "All10 forward passes complete. Final frozen candidate retained unchanged in /work/edited.\n"
            "Exact edit: layer8 down_proj column699, layer14 down_proj columns483/984, "
            "layer17 down_proj column1179 each multiplied by1.5; every other entry unchanged.\n"
            "No embedding/output/final-norm edits, no training, no whole-tensor rescaling.\n"
            "Path: baseline repetitive-failure diagnosis -> within-problem/length-adjusted channel screen "
            "-> teacher-forced reference/loop probes -> ablation refuted harmful-channel hypothesis "
            "-> amplification supported repetition-restraint role -> 1.5x selected by DEV accuracy "
            "-> frozen three-pass HOLDOUT confirmation.\n"
            "Pooled confirmation: " + json.dumps(final_pool) + "\n"
            "Group-level mechanism evidence does not prove each neuron individually causal.\n"
            "Complete audit and limitations: FINAL_REPORT.json. Final checksums verified.\n")
print("FINAL POOLED HOLDOUT")
print(json.dumps(final_pool, indent=2))
print("DEV strength tests:", json.dumps(dev_table))
print("Submission audit:", json.dumps(audit))
print("Submission unchanged:", edit_checksums() == frozen_checksums)
```
