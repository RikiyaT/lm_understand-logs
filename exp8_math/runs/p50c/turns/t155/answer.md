The recovered files confirm that the weights and evaluation plan survived the restart unchanged. Across 1,088 unique problems, the current gain is 50.5 percentage points, with 554 rescues and five regressions. Before the last forward pass, I will save a final audit of the mechanism, remaining failures, and exact submitted tensors. This checks that the gains chiefly rescue previously truncated reasoning while retaining substantial thinking, rather than merely forcing answers early.

```python
# Hypothesis: The frozen sparse edit chiefly rescues repetitive truncation while preserving substantial reasoning.
from collections import Counter

frame = read_paired_records(confirmed_runs)
assert len(frame) == 1088
assert int(frame.original.sum()) == 107
assert int(frame.edited.sum()) == 656

behavior = {}
for model in ["original", "edited"]:
    counts = Counter()
    correct_lengths, cut_repetition, nonwhite_cut_repetition = [], [], []
    for run_id in confirmed_runs:
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
                    cut_repetition.append(rep)
                    if not white:
                        nonwhite_cut_repetition.append(rep)
    behavior[model] = {
        **dict(counts),
        "median_correct_tokens": float(np.median(correct_lengths)),
        "mean_cut_tail_duplicate8": float(np.mean(cut_repetition)),
        "mean_nonwhite_cut_tail_duplicate8": float(np.mean(nonwhite_cut_repetition))
    }

transitions = {}
for cut, group in frame.groupby("original_cut"):
    transitions["originally_cut" if cut else "originally_finished"] = summarize_paired_frame(group)
source_stats = {}
for source, g in frame.groupby("source"):
    source_stats[source] = {
        "n": len(g), "original_correct": int(g.original.sum()),
        "edited_correct": int(g.edited.sum()),
        "gain": float(g.delta.mean()),
        "paired_se": float(g.delta.std(ddof=1)/np.sqrt(len(g)))
    }

audit = json.loads(Path("/work/frozen_weights_manifest.json").read_text())
assert set(edited()) == expected_names
for layer in [15, 19]:
    w = load(f"model.language_model.layers.{layer}.mlp.down_proj.weight")
    assert hashlib.sha256(w.contiguous().numpy().tobytes()).hexdigest() == audit[str(layer)]["sha256"]
del w

evidence = {
    "status": "49/50 passes complete; final locked 64-problem batch pending",
    "selection": "Keep frozen top64 alpha6 at layers 15 and 19; no subsequent tuning",
    "formula": "W0 - 6 * outer(u, mask64 * (u @ W0))",
    "direction_origin": "Four original matched productive/repetitive contexts, residual outputs 16 and 20",
    "causal_controls": {
        "top64_alpha6_productive_logp_delta": -0.00903,
        "top64_alpha6_repetitive_logp_delta": -0.10311,
        "norm_matched_random_control_accuracy_gain": 0.015625,
        "norm_matched_random_control_paired_se": 0.0352,
        "dose_response_and_layer_isolations": "See notes.md and experiment_history.md"
    },
    "confirmed_runs": confirmed_runs,
    "pooled": summarize_paired_frame(frame),
    "source_counts": source_stats,
    "behavior": behavior,
    "transition_strata_descriptive": transitions,
    "weights": audit,
    "remaining_request": frozen_eval_plan[-1]["request"],
    "limitations": [
        "Practice confirmation is not the unseen benchmark evaluation",
        "Some wrong answers, missing boxes, whitespace degeneration and truncation remain",
        "This repairs a reasoning failure mechanism; it does not establish new mathematical knowledge"
    ]
}
Path("/work/submission_evidence.json").write_text(json.dumps(evidence, indent=2))
frame.to_csv("/work/confirmation_through_pass49.csv", index=False)
Path("/work/final_choice.json").write_text(json.dumps({
    "choice": "frozen_top64_alpha6",
    "tensors": {k: {"name": v["name"], "sha256": v["sha256"]} for k,v in audit.items()},
    "changed_values": 131072,
    "no_further_weight_changes": True
}, indent=2))

print("SOURCE COUNTS", json.dumps(source_stats))
print("BEHAVIOR", json.dumps(behavior))
print("RESCUE STRATA", json.dumps(transitions))
print("REGRESSIONS", frame[frame.delta == -1][
    ["id", "source", "run", "original_cut", "edited_cut"]
].to_dict("records"))
print("FINAL CHOICE VERIFIED: exactly two tensors, frozen hashes; one forward pass remains.")
with open("/work/notes.md", "a") as f:
    f.write("\nPre-final submission audit at pass49: exact frozen hashes verified again; "
            "final_choice.json records KEEP top64alpha6. "
            "submission_evidence.json and confirmation_through_pass49.csv contain "
            "1088-problem evidence; final64 still unseen. Behavior: " + json.dumps(behavior) +
            "\nNo further weight changes planned. Final request is frozen_eval_plan[-1]['request'].\n")
```
