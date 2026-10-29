The token-level check weakens the case for neuron 14:1594: its absolute activity is not higher in the loop excerpts than in valid reference solutions. Four other channels have stronger converging evidence—8:699, 14:483, 14:984, and 17:1179—with higher loop activity and failure associations in both DEV halves. I’ll test their combined contribution by zeroing only their four output columns, leaving the rest of each tensor unchanged. The prediction is lower probability for repetitive continuations with little damage to valid calculations; failure of that prediction would argue against this intervention.

```python
target_channels = {8: [699], 14: [483, 984], 17: [1179]}
target_names = {l: f"model.layers.{l}.mlp.down_proj.weight" for l in target_channels}
def apply_channel_edit(strength):
    for layer, neuron_ids in target_channels.items():
        name = target_names[layer]
        tensor_new = load_original(name).clone()
        tensor_new[:, neuron_ids] *= (1.0 - strength)
        save_edit(name, tensor_new)
    return {"kind": "sparse_mlp_output_attenuation",
            "strength": float(strength),
            "targets": {str(k): v for k, v in target_channels.items()},
            "tensors": list(target_names.values())}

edit_spec = apply_channel_edit(1.0)
(wdir / "current_edit.json").write_text(json.dumps(edit_spec, indent=2))
print("Edit:", edit_spec)
for layer, js in target_channels.items():
    name = target_names[layer]
    original_w = load_original(name)
    current_w = load(name)
    delta_w = current_w - original_w
    print(name, "shape", tuple(current_w.shape),
          "changed_columns", int(torch.count_nonzero(delta_w.abs().sum(0))),
          "relative_frobenius_change", round(float(delta_w.norm()/original_w.norm()), 5))
    for j in js:
        c = next(c for c in probe_channels if c["layer"] == layer and c["neuron"] == j)
        print("Channel", layer, j, "loop activity by excerpt:",
              [round(v, 3) for v in c["loop_activity_by_probe"]])
import shutil
shutil.copytree(wdir / "edited", wdir / "candidate_ablation_1", dirs_exist_ok=True)
with open(wdir / "notes.md", "a") as f:
    f.write("\n## Pass 3 findings\n"
            "Neuron 14:1594 not selective in token-level probes; do NOT edit it.\n"
            "Convergent candidates: 8:699, 14:483, 14:984, 17:1179. Each has higher "
            "mean absolute loop activity than reference activity and negative length-adjusted "
            "correctness association in both DEV halves.\n"
            "\n## Pass 4 planned: causal teacher-forced screen\n"
            "Exact edit: model.layers.8.mlp.down_proj.weight column699; "
            "model.layers.14.mlp.down_proj.weight columns483,984; "
            "model.layers.17.mlp.down_proj.weight column1179. Strength 1.0: set these columns to zero.\n"
            "All other values unchanged. Candidate snapshot: candidate_ablation_1/.\n"
            "Set: same 72 literal DEV continuation probes as pass3. Compare matched tokens only.\n"
            "Prediction: reduce loop-continuation logprob without broad reference deterioration.\n")
print("Active edits:", edited())
```
