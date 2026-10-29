Full removal reduced DEV accuracy by 4.59 percentage points, with a reported standard error of 1.70 points, despite further reducing cutoffs. That argues against treating these channels as a purely harmful arithmetic circuit: they appear to carry useful computation too. I’ll restore the half-strength edit—not the original model—because it was the better of the two tested candidates. Its DEV result still does not demonstrate an improvement. I’ll now freeze that exact candidate and use the reserved holdout only for confirmation, with no further tuning.

```python
dev_full_original = load_generation_file("/work/gen_4_original.jsonl")
dev_full_edited = load_generation_file("/work/gen_4_edited.jsonl")
full_mechanisms = {}
for label, rr in [("original",dev_full_original), ("full_suppression",dev_full_edited)]:
    summary, _, _ = mechanism_summary(rr)
    full_mechanisms[label] = summary
    print(label, summary)
with open("/work/dev_full_mechanisms.json", "w") as f:
    json.dump(full_mechanisms, f, indent=2)

# Select solely from DEV, before any holdout generations.
chosen_strength = 0.5
apply_candidate(chosen_strength)

import hashlib
frozen_hashes = {}
for layer, js in selected_by_layer.items():
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    original = load_original(name)
    current = load(name)
    mask = torch.ones(original.shape[1], dtype=torch.bool)
    mask[js] = False
    assert current.shape == original.shape
    assert torch.equal(current[:, mask], original[:, mask])
    assert torch.equal(current[:, js], 0.5 * original[:, js])
    frozen_hashes[name] = hashlib.sha256(
        current.contiguous().numpy().tobytes()
    ).hexdigest()

shutil.copytree("/work/edited", "/work/best_1", dirs_exist_ok=True)
frozen_manifest = {
    "strength": chosen_strength,
    "columns": {str(l):js for l,js in selected_by_layer.items()},
    "tensor_sha256_float32": frozen_hashes,
    "chosen_using": "DEV paired accuracy only",
    "dev_half_difference_pp": -0.48828125,
    "dev_half_reported_se_pp": 1.57,
    "dev_full_difference_pp": -4.58984375,
    "dev_full_reported_se_pp": 1.70,
    "holdout_ids_file": "/work/HOLDOUT_ids.json",
    "frozen_before_holdout": True,
}
with open("/work/frozen_candidate.json", "w") as f:
    json.dump(frozen_manifest, f, indent=2)
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Pass 4 results and candidate selection\n"
        "- Original: 405/1024 (39.5508%); full suppression: 358/1024 (34.9609%).\n"
        "- Paired difference: -4.59 points; tool-reported SE 1.70 points.\n"
        "- Wrong-to-right 155; right-to-wrong 202. Full removal is harmful on DEV.\n"
        "- Strong suppression hypothesis is not supported. Reduced truncation "
        "does not imply improved mathematical accuracy.\n"
        "- Best tested non-original candidate: half suppression, selected on DEV "
        "only. It has no demonstrated DEV accuracy benefit.\n"
        "\n## Candidate frozen before HOLDOUT\n"
        "- Exact edit: down_proj[:,j] = 0.5 * original[:,j] for the same 64 columns "
        "in candidate_manifest.json; all other values unchanged.\n"
        "- Verified bitwise unchanged unselected columns; tensor hashes saved "
        "in frozen_candidate.json. Backup: /work/best_1/.\n"
        "- No further candidate selection or tuning using HOLDOUT.\n"
        "\n## Pass 5: frozen HOLDOUT confirmation\n"
        "- Problems: HOLDOUT_ids.json, fixed before baseline; 128 MATH problems.\n"
        "- Both models, 8 samples/problem = 1024 answers/model, max_tokens=8192.\n"
        "- Report the paired accuracy difference and its uncertainty honestly, "
        "including a negative or inconclusive result.\n"
    )
print("Frozen strength:", chosen_strength)
print("Verified edited tensors:", len(frozen_hashes))
print("Edited scalar weights:", 640 * sum(map(len, selected_by_layer.values())))
print("HOLDOUT IDs:")
print(json.dumps(holdout_ids))
```
