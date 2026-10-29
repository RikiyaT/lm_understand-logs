The final batch also improved, by 39.1 percentage points, with 25 rescues and no regressions. Combining all confirmation batches gives approximately 10.6% versus 60.4% accuracy on 1,152 distinct problems, with 579 rescues and five regressions. I will retain the exact frozen edit, update the final evidence files, and verify its hashes once more. No further tuning or forward passes are needed.

```python
# Hypothesis: The completed held-out evaluation justifies retaining the exact frozen, repetition-targeted edit.
import re, shutil
from collections import Counter

register_frozen_run(69)
assert len(frozen_eval_completed) == 16 and len(confirmed_runs) == 18
frame = read_paired_records(confirmed_runs)
assert len(frame) == 1152 and frame.id.is_unique
summary = summarize_paired_frame(frame)
summary["gain_95pct_normal_interval"] = [
    summary["gain"] - 1.96*summary["se"],
    summary["gain"] + 1.96*summary["se"]
]

# Verify separation from all earlier generation-based edit-selection experiments.
prior_ids = set()
for path in Path("/work").glob("gen_*.jsonl"):
    match = re.fullmatch(r"gen_(\d+)(?:_original)?\.jsonl", path.name)
    if match and int(match.group(1)) < 50:
        with open(path) as f:
            prior_ids.update(json.loads(line)["id"] for line in f)
assert not set(frame.id).intersection(prior_ids)

def final_behavior(run_ids):
    results = {}
    for model in ["original", "edited"]:
        counts = Counter()
        correct_lengths, cut_reps, nonwhite_reps = [], [], []
        for run_id in run_ids:
            with open(f"/work/gen_{run_id}_{model}.jsonl") as f:
                for line in f:
                    r = json.loads(line)
                    s = r["samples"][0]
                    t = s["text"]
                    counts["n"] += 1
                    counts["correct"] += bool(s["correct"])
                    counts["cut"] += bool(s["cut"])
                    white = bool(s["cut"] and len(t) >= 400 and not t[-400:].strip())
                    counts["whitespace_tail_cut"] += white
                    if s["correct"]:
                        correct_lengths.append(s["tokens"])
                        counts["correct_thinking_closed"] += "</think>" in t
                        counts["correct_close_after_1000_chars"] += t.find("</think>") > 1000
                    elif s["cut"]:
                        counts["failure_cut"] += 1
                    elif s["pred"] is None or str(s["pred"]).strip() in ["", "None"]:
                        counts["failure_no_answer_finished"] += 1
                    else:
                        counts["failure_wrong_answer_finished"] += 1
                    if s["cut"]:
                        words = t.split()[-2000:]
                        grams = [tuple(words[i:i+8]) for i in range(max(0, len(words)-7))]
                        rep = 1-len(set(grams))/len(grams) if grams else 0.
                        cut_reps.append(rep)
                        if not white:
                            nonwhite_reps.append(rep)
        results[model] = {
            **dict(counts),
            "median_correct_tokens": float(np.median(correct_lengths)),
            "mean_cut_tail_duplicate8": float(np.mean(cut_reps)),
            "mean_nonwhite_cut_tail_duplicate8": float(np.mean(nonwhite_reps))
        }
    return results

behavior = final_behavior(confirmed_runs)
source_stats = {}
for source, g in frame.groupby("source"):
    source_stats[source] = {
        "n": len(g), "original_correct": int(g.original.sum()),
        "edited_correct": int(g.edited.sum()),
        "gain": float(g.delta.mean()),
        "paired_se": float(g.delta.std(ddof=1)/np.sqrt(len(g)))
    }
splits = json.loads(Path("/work/fresh_splits.json").read_text())
split_stats = {}
for label, ids in splits.items():
    g = frame[frame.id.isin(ids)]
    assert len(g) == len(ids)
    split_stats[label] = summarize_paired_frame(g)
transitions = {
    "originally_cut" if cut else "originally_finished": summarize_paired_frame(g)
    for cut, g in frame.groupby("original_cut")
}

# Audit the actual submission and preserve an identical backup; do not rewrite any weights.
audit = json.loads(Path("/work/frozen_weights_manifest.json").read_text())
assert set(edited()) == expected_names
for layer in [15, 19]:
    w = load(f"model.language_model.layers.{layer}.mlp.down_proj.weight")
    assert list(w.shape) == [1024, 3584] and torch.isfinite(w).all()
    assert hashlib.sha256(w.contiguous().numpy().tobytes()).hexdigest() == audit[str(layer)]["sha256"]
del w
shutil.copytree("/work/edited", "/work/final_verified_edit", dirs_exist_ok=True)

evidence = json.loads(Path("/work/submission_evidence.json").read_text())
evidence.update({
    "status": "All 50 forward passes complete; frozen top64 alpha6 retained",
    "confirmed_runs": confirmed_runs,
    "pooled": summary,
    "source_counts": source_stats,
    "independent_splits": split_stats,
    "behavior": behavior,
    "transition_strata_descriptive": transitions,
    "earlier_generation_selection_ids_checked": len(prior_ids),
    "overlap_with_earlier_generation_selection": 0,
    "verified_backup": "/work/final_verified_edit"
})
evidence.pop("remaining_request", None)
Path("/work/submission_evidence.json").write_text(json.dumps(evidence, indent=2))
frame.to_csv("/work/final_confirmation.csv", index=False)
Path("/work/final_choice.json").write_text(json.dumps({
    "choice": "KEEP frozen_top64_alpha6",
    "forward_passes_completed": 50,
    "tensors": {k: {"name": v["name"], "sha256": v["sha256"]} for k,v in audit.items()},
    "changed_values": 131072,
    "verified_backup": "/work/final_verified_edit",
    "no_weight_changes_during_confirmation": True
}, indent=2))

old_notes = Path("/work/notes.md").read_text()
with open("/work/experiment_history.md", "a") as f:
    f.write("\n\n--- Final pre-summary notes ---\n" + old_notes)
final_notes = """# FINAL SUBMISSION — all 50 forward passes complete
KEEP the frozen top64 alpha6 edit. Only:
model.language_model.layers.15.mlp.down_proj.weight
model.language_model.layers.19.mlp.down_proj.weight

Formula: W = W0 - 6*outer(u, mask64*(u@W0)).
Unit repetition directions from four matched productive/repetitive original contexts
at residual outputs16/20;64 selected MLP writer neurons per layer, chosen on original fwd4.
131072 changed values total; same names/shapes. No norm/embedding/head edits or training.
Relative matrix changes10.6357%/8.1058%; hashes unchanged throughout confirmation.
Selected neuron IDs and hashes: frozen_weights_manifest.json.
Identical final backup: final_verified_edit/. Actual submission remains /work/edited.

PATH AND CAUSAL EVIDENCE
Initial11/96 correct,85 cutoff,~95% duplicate tail phrases. Targeted repetition, not math facts.
Full directional doses alpha1/2/4 selectively reduced repetitive branch logp
by .02410/.04870/.10372 nats/token; productive branches barely changed.
L15-only and L19-only interventions supported both sites; attention-head alternatives removed.
Norm-matched orthogonal random control gained1.56pp SE3.52, not the targeted intervention's gain.
Top64 alpha6 transfer: repetitive-.10311 versus productive-.00903 nats/token.
Broader productive/numeric controls changed little; no consistent premature closure.
Fullalpha1/2 generation dose tests showed increased rescue; sparse top64 retained concentrated
repetition writing while changing fewer weights. Frozen before any confirmation below.
Remaining errors include wrong math, missing boxes, whitespace degeneration and truncation.

FINAL CONFIRMATION
All1152 unique problems were independent of edit selection. Paired within run,
both models, n1, max32768, fixed thinking-enabled evaluation sampling.
Repeated gen68 summaries counted once; kernel restart did not change saved hashes.
All five independent confirmation splits and all individual batch results retained.
These are held-out PRACTICE results, not unseen benchmark scores.
"""
final_notes += "\nPOOLED\n" + json.dumps(summary, indent=2)
final_notes += "\nSOURCE COUNTS\n" + json.dumps(source_stats, indent=2)
final_notes += "\nBEHAVIOR\n" + json.dumps(behavior, indent=2)
final_notes += "\nFiles: submission_evidence.json, final_confirmation.csv, final_choice.json, "
final_notes += "frozen_eval_progress.json, frozen_weights_manifest.json, experiment_history.md.\n"
Path("/work/notes.md").write_text(final_notes)
Path("/work/final_report.md").write_text(final_notes)

print("FINAL POOLED", json.dumps(summary))
print("FINAL SOURCE COUNTS", json.dumps(source_stats))
print("FINAL BEHAVIOR", json.dumps(behavior))
print("INDEPENDENT SPLITS", {
    k: {"n":v["items"], "gain":v["gain"], "se":v["se"]}
    for k,v in split_stats.items()
})
print("RESCUE STRATA", json.dumps(transitions))
print("FINAL AUDIT PASSED: retain the same two tensors; reports and verified backup saved.")
```
