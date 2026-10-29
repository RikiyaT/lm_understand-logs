The final pass was slightly negative, and all fifty forward passes are now complete. I will retain the candidate frozen before HOLDOUT, as planned, and make no further weight changes. I will now pool all sixteen HOLDOUT passes, independently check the paired and problem-cluster standard errors, and save the final audit. This will determine the reported gain while keeping the distinction between suggestive component evidence and a proven mathematical repair.

```python
from scipy.stats import t as student_t

assert verify_frozen_candidate()
final_holdout_passes = list(range(35, 51))
final_dev_summary = pooled_pairs([29, 30, 33, 34])
final_holdout_summary = pooled_pairs(final_holdout_passes)
final_behavior = holdout_behavior_summary(final_holdout_passes)

problem_differences = {pid: [] for pid in holdout_ids}
all_differences = []
wrong_to_right = 0
right_to_wrong = 0
for k in final_holdout_passes:
    maps = {}
    for model in ["original", "edited"]:
        with (work / f"gen_{k}_{model}.jsonl").open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        maps[model] = {r["id"]: r["samples"] for r in records}
    assert set(maps["original"]) == set(maps["edited"])
    for pid in maps["original"]:
        original_samples = maps["original"][pid]
        edited_samples = maps["edited"][pid]
        assert len(original_samples) == len(edited_samples) == 32
        for original_sample, edited_sample in zip(original_samples, edited_samples):
            d = int(edited_sample["correct"]) - int(original_sample["correct"])
            problem_differences[pid].append(d)
            all_differences.append(d)
            wrong_to_right += int(d == 1)
            right_to_wrong += int(d == -1)

assert len(all_differences) == 16384
assert all(len(v) == 128 for v in problem_differences.values())
sample_d = np.asarray(all_differences, dtype=np.float64)
problem_d = np.asarray(
    [np.mean(problem_differences[pid]) for pid in holdout_ids],
    dtype=np.float64
)
independent_gain = float(100 * sample_d.mean())
independent_sample_se = float(100 * sample_d.std(ddof=1) / np.sqrt(len(sample_d)))
independent_problem_se = float(100 * problem_d.std(ddof=1) / np.sqrt(len(problem_d)))
assert np.isclose(independent_gain, final_holdout_summary["gain_pp"])
assert np.isclose(independent_sample_se, final_holdout_summary["sample_SE_pp"])
assert np.isclose(independent_problem_se,
                  final_holdout_summary["problem_cluster_SE_pp"])

critical = float(student_t.ppf(0.975, 127))
final_interval = [
    independent_gain - critical * independent_problem_se,
    independent_gain + critical * independent_problem_se
]
archive_hashes = {
    str(p.relative_to(frozen_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(frozen_dir.rglob("*")) if p.is_file()
}
assert archive_hashes == frozen_hashes

final_report = {
    "submitted_edit": final_spec,
    "scope_audit": scope_audit,
    "selection": "DEV only; frozen before pass35; unchanged throughout HOLDOUT",
    "DEV_finalist_selection_biased": final_dev_summary,
    "HOLDOUT": final_holdout_summary,
    "HOLDOUT_cluster_t_95_interval_pp": final_interval,
    "paired_wrong_to_right": wrong_to_right,
    "paired_right_to_wrong": right_to_wrong,
    "HOLDOUT_behavior": final_behavior,
    "all_128_problems_have_128_samples_per_model": True,
    "frozen_and_submission_hashes_match": True,
    "interpretation": (
        "Small positive HOLDOUT point estimate; problem-cluster interval includes zero. "
        "DEV component interventions support a role for layer11 head2 and motivate "
        "the selected neuron suppression, but do not establish a complete arithmetic "
        "failure mechanism. The edit increases response length and cutoffs."
    )
}
(work / "final_report.json").write_text(json.dumps(final_report, indent=2))
with (work / "notes.md").open("a") as f:
    f.write("\nFINAL SUBMISSION: retained exactly the candidate frozen before pass35; "
            "no HOLDOUT-driven changes. All50 forward passes used.\n")
    f.write("Final DEV: " + json.dumps(final_dev_summary) + "\n")
    f.write("Final HOLDOUT: " + json.dumps(final_holdout_summary)
            + "; cluster-t95% interval in pp: " + json.dumps(final_interval) + ".\n")
    f.write("Final behavior: " + json.dumps(final_behavior) + ".\n")
    f.write("Scope and hashes independently verified. Full report: /work/final_report.json.\n")

print("FINAL_DEV", final_dev_summary)
print("FINAL_HOLDOUT", final_holdout_summary)
print("CLUSTER_95_INTERVAL_PP", [round(x, 4) for x in final_interval])
print("PAIRED_TRANSITIONS", {"wrong_to_right": wrong_to_right,
                             "right_to_wrong": right_to_wrong})
print("FINAL_BEHAVIOR", {
    m: {k: round(float(v), 3) if isinstance(v, (float, np.floating)) else v
        for k, v in s.items()}
    for m, s in final_behavior.items()
})
print("SUBMITTED_TENSORS", edited())
"Final candidate retained, fully audited, and documented in /work/final_report.json."
```