The final batch improved from 28/256 to 47/256 correct, adding another positive replication. Across all 768 fresh pairs, the frozen edit produced 64 additional correct answers. I will keep that edit, calculate the final uncertainty and source-specific results, and verify that no temporary control edits remain. I will also inspect both a rescued answer and a regression so the final report reflects the remaining failure modes.

```python
# Hypothesis: the retained H7-specific repair has a reproducible accuracy benefit, with gains primarily from escaping cut-off reasoning.
import shutil
import hashlib
import re

h4_rows = read_paired_generation(34)
assert {r["id"] for r in h4_rows} == set(holdout_info["H4"])
primary_rows = sum((read_paired_generation(k) for k in (30, 31, 32, 34)), [])
assert len(primary_rows) == 768
assert {r["id"] for r in primary_rows} == set(all_fresh_ids)

previous_generation_ids = set()
for path in work.glob("gen_*.jsonl"):
    match = re.fullmatch(r"gen_(\d+)(?:_original|_edited)?\.jsonl", path.name)
    if match and int(match.group(1)) < 30:
        with path.open() as f:
            for line in f:
                previous_generation_ids.add(json.loads(line)["id"])
assert set(all_fresh_ids).isdisjoint(previous_generation_ids), "Primary confirmation must exclude all earlier generated problems."

final_results = {"primary": {"all": summarize_pairs(primary_rows)}}
for source in ("gsm8k", "math"):
    final_results["primary"][source] = summarize_pairs(
        [r for r in primary_rows if r["source"] == source])
final_results["by_batch"] = {
    label: summarize_pairs(read_paired_generation(k))
    for label, k in (("H2A", 30), ("H2B", 31), ("H3", 32), ("H4", 34))
}
supportive_rows = read_paired_generation(20) + read_paired_generation(21)
final_results["earlier_supportive_H1"] = summarize_pairs(supportive_rows)
final_results["transitions"] = {
    "rescues": sum(not r["original_correct"] and r["edited_correct"] for r in primary_rows),
    "rescues_from_original_cutoff": sum(
        not r["original_correct"] and r["edited_correct"] and r["original_cut"]
        for r in primary_rows),
    "regressions": sum(r["original_correct"] and not r["edited_correct"] for r in primary_rows),
    "regressions_to_edited_cutoff": sum(
        r["original_correct"] and not r["edited_correct"] and r["edited_cut"]
        for r in primary_rows)
}
print("PRIMARY CONFIRMATION:", json.dumps(final_results["primary"], indent=2))
print("Correctness transitions:", final_results["transitions"])

for label, predicate in (
    ("rescue", lambda r: not r["original_correct"] and r["edited_correct"]),
    ("regression", lambda r: r["original_correct"] and not r["edited_correct"])
):
    chosen = next(r for r in primary_rows if predicate(r))
    print(label.upper(), chosen["id"])
    for model_label in ("original", "edited"):
        with (work / f"gen_{chosen['run']}_{model_label}.jsonl").open() as f:
            for line in f:
                record = json.loads(line)
                if record["id"] == chosen["id"]:
                    s = record["samples"][0]
                    print(model_label, {"correct": s["correct"], "cut": s["cut"], "tokens": s["tokens"]},
                          "tail:", s["text"][-240:].encode("ascii", "replace").decode())
                    break

assert edited() == [target_name]
final_w = load(target_name)
orig_final_check = load_original(target_name)
assert final_w.shape == orig_final_check.shape
assert torch.isfinite(final_w).all().item()
assert torch.equal(final_w[:, :1792], orig_final_check[:, :1792])
assert torch.count_nonzero(final_w[:, 1792:2048]).item() == 0
assert torch.equal(final_w[:, 2048:], orig_final_check[:, 2048:])
assert torch.equal(final_w, frozen_w)
integrity = {
    "tensor": target_name, "shape": list(final_w.shape),
    "zero_based_layer": 15, "zero_based_head": 7, "zeroed_columns_half_open": [1792, 2048],
    "changed_values": int(torch.count_nonzero(final_w != orig_final_check).item()),
    "sha256_float32": hashlib.sha256(final_w.contiguous().numpy().tobytes()).hexdigest(),
    "all_other_tensor_edits_absent": True
}
final_results["integrity"] = integrity
final_results["decision"] = "Keep the pre-frozen L15H7 ablation; no other edits."
(work / "final_results.json").write_text(json.dumps(final_results, indent=2))
(work / "final_recipe.json").write_text(json.dumps(frozen_recipe, indent=2))
(work / "current_recipe.json").write_text(json.dumps(frozen_recipe, indent=2))
shutil.copytree(work / "edited", work / "final_L15H7_zero", dirs_exist_ok=True)
primary_runs.append({"set": "H4", "generation_file_index": 34})
(work / "primary_confirmation_runs.json").write_text(json.dumps(primary_runs))

with (work / "notes.md").open("a") as f:
    f.write("\nFinal H4 first attempt failed engine initialization without outcomes; identical retry succeeded.\n")
    f.write("Pass25/25 gen_34 H4 256:28->47 correct,29 improvements/10 regressions, gain7.421875pp,SE2.40pp. All25 passes now used.\n")
    f.write("FINAL PRIMARY CONFIRMATION: all768 H2-H4 IDs verified disjoint from EVERY earlier generated problem. No candidate or strength changes during confirmation. " +
            json.dumps(final_results["primary"]) + "\n")
    f.write("Transition counts " + json.dumps(final_results["transitions"]) + "\n")
    f.write("FINAL DECISION: retain ONLY model.language_model.layers.15.self_attn.o_proj.weight columns[1792:2048]=0 (zero-based head7). Integrity " +
            json.dumps(integrity) + ". Backup /work/final_L15H7_zero; results /work/final_results.json.\n")
    f.write("Mechanism evidence: exact-text copy attention .887 loop vs .082 productive; weakening H7 showed dose response; x1.5 amplified loop likelihood in all3 broad probes; H4 ablation had much weaker loop effects and worse productive likelihood. Exclude mistaken modular-productive H4 control row. MLP and alternative-head candidates rejected. Accuracy evidence is primary; likelihood probes are mechanistic, not benchmarks. Many cutoffs remain; no universal reasoning-preservation or benchmark-score guarantee.\n")
print("FINAL WEIGHT AUDIT:", integrity)
print("Submitted directory remains /work/edited; only the frozen L15H7 ablation is active.")
```
