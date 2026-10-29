Full ablation did not improve DEV accuracy: the paired difference was −0.68 percentage points with a 1.64-point standard error. More importantly for the mechanism, it increased average response length and nearly doubled the cutoff rate, contradicting the predicted suppression of loops. The channels may be part of a compensating response that restrains repetition, rather than causing it. I’ll measure the behavioral change more directly and test the opposite intervention—doubling those same four output columns—while continuing to judge success by accuracy.

```python
def read_generation(pass_no, model_label):
    with open(wdir / f"gen_{pass_no}_{model_label}.jsonl") as f:
        records = [json.loads(line) for line in f]
    return records

def flatten_generation(records):
    return [dict(s, problem_id=r["id"], sample_idx=i)
            for r in records for i, s in enumerate(r["samples"])]

def paired_stats(pass_no):
    original_records = read_generation(pass_no, "original")
    edited_records = read_generation(pass_no, "edited")
    oo = flatten_generation(original_records)
    ee = flatten_generation(edited_records)
    oo = sorted(oo, key=lambda s: (s["problem_id"], s["sample_idx"]))
    ee = sorted(ee, key=lambda s: (s["problem_id"], s["sample_idx"]))
    assert [(s["problem_id"], s["sample_idx"]) for s in oo] == [
        (s["problem_id"], s["sample_idx"]) for s in ee]
    d = np.array([int(e["correct"]) - int(o["correct"]) for o, e in zip(oo, ee)])
    out = {"pass": pass_no, "n": len(d), "gain_pp": float(100*d.mean()),
           "se_pp": float(100*d.std(ddof=1)/np.sqrt(len(d))),
           "wrong_to_right": int((d == 1).sum()), "right_to_wrong": int((d == -1).sum())}
    for label, ss in [("original", oo), ("edited", ee)]:
        out[label] = {
            "accuracy": float(np.mean([s["correct"] for s in ss])),
            "mean_tokens": float(np.mean([s["tokens"] for s in ss])),
            "cut_rate": float(np.mean([s["cut"] for s in ss])),
            "repeat_fraction": float(np.mean([repeat_fraction(s["text"]) for s in ss])),
            "three_or_more_final_answers": float(np.mean([
                len(re.findall(r"final answer", s["text"], flags=re.I)) >= 3 for s in ss]))
        }
    return out, oo, ee, d

stats5, orig5, edit5, diff5 = paired_stats(5)
print(json.dumps(stats5, indent=2))
(wdir / "stats_pass5.json").write_text(json.dumps(stats5, indent=2))
shutil.copytree(wdir / "edited", wdir / "best_1", dirs_exist_ok=True)
(wdir / "best_1_spec.json").write_text(json.dumps(edit_spec, indent=2))
edit_spec = apply_channel_edit(-1.0)
(wdir / "current_edit.json").write_text(json.dumps(edit_spec, indent=2))
shutil.copytree(wdir / "edited", wdir / "candidate_enhance_2x", dirs_exist_ok=True)
with open(wdir / "notes.md", "a") as f:
    f.write("\n## Pass 5 results\n" + json.dumps(stats5) + "\n"
            "Ablation increased response length/cutoffs; proposed anti-loop direction was wrong.\n"
            "Revised hypothesis: selected channels help restrain repetitive conclusions; "
            "their correlation with failures may reflect compensation.\n"
            "\n## Pass 6 planned: opposite-direction causal test\n"
            "Exact edit: down_proj columns 8:[699],14:[483,984],17:[1179] multiplied by2 "
            "(attenuation strength -1.0). All other values unchanged.\n"
            "Set: same fixed DEV64, both models, n16, max_tokens8192 (1024/model).\n"
            "Prediction: shorter loops/fewer cutoffs than contemporaneous original; "
            "accuracy remains primary endpoint.\n")
print("Next edit:", edit_spec)
```
