The frozen edit scored 9.96% on HOLDOUT versus 8.79% for the original: a paired gain of 1.17 percentage points, with a reported standard error of 0.90 points. This is a positive but statistically inconclusive result, especially given the slightly negative DEV estimate. I’ll submit the unchanged half-strength candidate and verify it against the pre-holdout hashes. I’ll also record the experimenters’ correction: earlier missing-answer counts included some wrong answers and should not support a failure-mechanism claim. The accuracy comparisons are unchanged, and I’ll distinguish the observed gain from the still-unproven mechanism.

```python
holdout_original = load_generation_file("/work/gen_5_original.jsonl")
holdout_edited = load_generation_file("/work/gen_5_edited.jsonl")
assert set(r["problem_id"] for r in holdout_original) == holdout_set
assert not holdout_set & dev_set

holdout_mechanisms = {}
for label, rr in [("original",holdout_original), ("edited",holdout_edited)]:
    summary, _, _ = mechanism_summary(rr)
    holdout_mechanisms[label] = summary

assert set(edited()) == set(frozen_hashes)
for name, expected_hash in frozen_hashes.items():
    current = load(name)
    actual_hash = hashlib.sha256(current.contiguous().numpy().tobytes()).hexdigest()
    assert actual_hash == expected_hash

n = len(holdout_original)
original_correct = sum(r["correct"] for r in holdout_original)
edited_correct = sum(r["correct"] for r in holdout_edited)
difference_pp = 100 * (edited_correct-original_correct) / n
reported_se_pp = 0.90
ci95 = [difference_pp-1.96*reported_se_pp, difference_pp+1.96*reported_se_pp]

final_report = {
    "submission": "Unchanged candidate frozen before HOLDOUT",
    "edited_tensors": sorted(frozen_hashes),
    "edited_neurons": 64,
    "edited_scalar_values": 40960,
    "operation": "Multiply only the selected down_proj columns by 0.5",
    "layers": sorted(selected_by_layer),
    "dev": {
        "half_suppression_difference_pp": -0.48828125,
        "half_suppression_reported_se_pp": 1.57,
        "full_suppression_difference_pp": -4.58984375,
        "full_suppression_reported_se_pp": 1.70,
    },
    "holdout": {
        "problems": 128, "answers_per_model": n,
        "original_correct": original_correct, "edited_correct": edited_correct,
        "original_accuracy": original_correct/n,
        "edited_accuracy": edited_correct/n,
        "difference_pp": difference_pp,
        "reported_standard_error_pp": reported_se_pp,
        "approximate_95pct_interval_pp": ci95,
        "wrong_to_right": 46, "right_to_wrong": 34,
        "mechanism_diagnostics": holdout_mechanisms,
    },
    "limitations": [
        "Positive holdout estimate is not statistically significant at the 95% level.",
        "Half-strength DEV estimate was slightly negative.",
        "Full removal harmed accuracy, so selected channels also support useful computation.",
        "Arithmetic-error associations do not establish a dedicated error-causing circuit.",
        "Teacher-forced pass 2 was consumed by an input-format mistake and provided no math localization.",
        "Earlier no_answer_finished counts were over-counted; do not interpret them as true missing-answer counts.",
        "No MATH-500 evaluation was observed."
    ],
}
with open("/work/final_report.json", "w") as f:
    json.dump(final_report, f, indent=2)
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Final HOLDOUT result and submission\n"
        "- Original: 90/1024 (8.7891%); edited: 102/1024 (9.9609%).\n"
        "- Paired difference: +1.171875 percentage points; reported SE: 0.90 points.\n"
        f"- Approximate 95% interval: [{ci95[0]:.2f}, {ci95[1]:.2f}] points.\n"
        "- Wrong-to-right 46; right-to-wrong 34. Positive but inconclusive.\n"
        "- Submitted candidate is unchanged from pre-HOLDOUT freeze: "
        "64 selected MLP down_proj columns in layers 6-17 multiplied by 0.5.\n"
        "- All saved tensor hashes verified against frozen_candidate.json.\n"
        "- No HOLDOUT tuning or weight changes after confirmation.\n"
        "- Experimenter correction: earlier no_answer_finished counts included "
        "some wrong final answers; those category counts cannot establish an "
        "answer-completion failure mechanism. Accuracy comparisons unaffected.\n"
        "- Full-removal harm and weak half-strength results limit the causal "
        "interpretation; no claim of a proven arithmetic-error circuit.\n"
        "- Complete outcome and limitations: final_report.json.\n"
    )

print("HOLDOUT:", original_correct, "->", edited_correct, "correct out of", n)
print("Gain: %.2f percentage points; SE %.2f; approximate 95%% interval [%.2f, %.2f]"
      % (difference_pp, reported_se_pp, ci95[0], ci95[1]))
print("Secondary arithmetic diagnostics:", holdout_mechanisms)
print("Verified final submission: frozen half-strength edit; 12 tensors, 64 columns.")
```
