The 1.5× edit is the best tested DEV candidate: +0.49 percentage points in paired accuracy, with a 1.72-point standard error, so the gain is not statistically conclusive. It also reduces response length and cutoffs, consistent with the revised repetition-control hypothesis. I’ll freeze this exact edit now and back it up before examining any HOLDOUT output. The final three passes will use the predeclared HOLDOUT orders, and I will pool their results without changing the candidate.

```python
stats7, orig7, edit7, diff7 = paired_stats(7)
(wdir / "stats_pass7.json").write_text(json.dumps(stats7, indent=2))
dev_stats = [stats5, stats6, stats7]
chosen = max(dev_stats, key=lambda s: s["gain_pp"])
assert chosen["pass"] == 7
assert json.loads((wdir / "current_edit.json").read_text())["strength"] == -0.5
for layer, columns in target_channels.items():
    original_w = load_original(target_names[layer])
    current_w = load(target_names[layer])
    mask_columns = torch.ones(2048, dtype=torch.bool)
    mask_columns[columns] = False
    assert torch.equal(current_w[:, mask_columns], original_w[:, mask_columns])
    assert torch.allclose(current_w[:, columns], 1.5 * original_w[:, columns])
shutil.copytree(wdir / "edited", wdir / "best_2", dirs_exist_ok=True)
shutil.copytree(wdir / "edited", wdir / "frozen_candidate", dirs_exist_ok=True)

import hashlib
def edit_checksums():
    return {str(p.relative_to(wdir / "edited")): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((wdir / "edited").rglob("*")) if p.is_file()}
frozen_checksums = edit_checksums()
freeze_manifest = {
    "edit": edit_spec, "selected_on": "DEV only",
    "selected_pass": 7, "dev_gain_pp": stats7["gain_pp"],
    "dev_reported_cluster_se_pp": 1.72,
    "forward_edit_hash": "599aba54ec",
    "files_sha256": frozen_checksums,
    "holdout_plan": {"passes": [8, 9, 10], "problems": 256, "n_per_pass": 4,
                     "max_tokens": 8192, "orders_file": "holdout_confirmation_orders.json"}
}
(wdir / "FROZEN_MANIFEST.json").write_text(json.dumps(freeze_manifest, indent=2))
with open(wdir / "notes.md", "a") as f:
    f.write("\n## Pass 7 results and FINAL CANDIDATE FREEZE\n" + json.dumps(stats7) + "\n"
            "Selected pass7 (1.5x) by highest paired DEV accuracy difference among passes5–7.\n"
            "DEV gain +0.48828125 pp, reported problem-cluster SE1.72 pp; not significant.\n"
            "Final exact edit: model.layers.8.mlp.down_proj.weight column699, "
            "model.layers.14.mlp.down_proj.weight columns483,984, "
            "model.layers.17.mlp.down_proj.weight column1179 multiplied by1.5 from ORIGINAL values.\n"
            "All other entries identical. Frozen hash599aba54ec, copies in best_2/ and frozen_candidate/.\n"
            "DO NOT TUNE OR SELECT USING HOLDOUT. Submit this frozen candidate regardless of confirmation sign.\n"
            "\n## Pass 8 planned\n"
            "Exact edit: frozen1.5x four-column enhancement above; no further changes.\n"
            "Set: HOLDOUT.json, all256 reserved MATH problems, first predeclared order.\n"
            "both, n4, max_tokens8192, 1024 answers/model. Pool with planned passes9 and10.\n")
print("DEV gains (percentage points):", [(s["pass"], round(s["gain_pp"], 3)) for s in dev_stats])
print("Frozen edit:", edit_spec)
print("HOLDOUT first order:")
print(json.dumps(holdout_orders[0]))
```
