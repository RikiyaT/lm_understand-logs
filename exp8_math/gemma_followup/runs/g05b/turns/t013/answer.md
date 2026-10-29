After controlling for surface features, 166 later-layer channels remain associated with both failure and arithmetic errors, with the failure association present in both DEV halves. I’ll further screen their output directions to avoid suppressing channels that appear to oppose the observed failure-associated layer update. From that screen, I’ll fix a sparse set of at most 64 neurons and halve only their `down_proj` columns. The next two DEV passes will compare half suppression with full removal of the same set. This tests a tentative hypothesis about reinforcement of invalid transformations; improved accuracy and fewer checked arithmetic errors would support it, while degradation would argue against it.

```python
torch.set_grad_enabled(False)

# Geometry of existing weights and recorded activations, not a forward simulation.
for layer in range(6,18):
    delta = (
        np.asarray(hidden_mean[dev_indices, layer+1, :], dtype=np.float64)
        - np.asarray(hidden_mean[dev_indices, layer, :], dtype=np.float64)
    )
    delta_adj = remove_controls(within_problem(delta))
    direction = f_adj @ delta_adj / (f_adj @ f_adj)
    D = load_original(f"model.layers.{layer}.mlp.down_proj.weight").numpy().astype(np.float64)
    gamma = 1 + load_original(
        f"model.layers.{layer}.post_feedforward_layernorm.weight"
    ).numpy().astype(np.float64)
    columns = gamma[:,None] * D
    align = direction @ columns / (
        np.linalg.norm(direction) * np.linalg.norm(columns, axis=0) + 1e-20
    )
    for s in controlled_candidates:
        if s["layer"] == layer:
            s["failure_update_alignment"] = float(align[s["neuron"]])

reinforcing = [
    s for s in controlled_candidates if s["failure_update_alignment"] > 0
]
selected_components = reinforcing[:64]
assert len(selected_components) >= 16
selected_by_layer = collections.defaultdict(list)
for s in selected_components:
    selected_by_layer[s["layer"]].append(s["neuron"])
selected_by_layer = {l: sorted(js) for l,js in sorted(selected_by_layer.items())}

selection_manifest = {
    "hypothesis": "Sparse late MLP outputs reinforce invalid numerical transformations.",
    "selection": (
        "Within-DEV-problem correlations, controlling log length and six surface "
        "features; positive-mean neurons in layers 6-17; adjusted failure>.10; "
        "both DEV-half failure correlations>.045; adjusted arithmetic-error>.04; "
        "positive output alignment with the observed failure-associated block update. "
        "Rank by min half correlation times sqrt arithmetic-error correlation; top64."
    ),
    "neurons": {str(l):js for l,js in selected_by_layer.items()},
    "evidence": selected_components,
    "planned_strengths": [0.5,1.0],
    "primary_outcome": "paired final-answer accuracy",
    "secondary_outcomes": [
        "responses containing mechanically checked false numeric equalities",
        "arithmetic-check coverage", "failure types", "repetition"
    ],
}
with open("/work/candidate_manifest.json", "w") as f:
    json.dump(selection_manifest, f, indent=2)

def apply_candidate(strength):
    summaries = []
    for layer, js in selected_by_layer.items():
        name = f"model.layers.{layer}.mlp.down_proj.weight"
        original = load_original(name)
        new = original.clone()
        new[:, js] = original[:, js] * (1-strength)
        save_edit(name, new)
        summaries.append({
            "layer": layer, "neurons": len(js),
            "relative_frobenius_change": float(torch.linalg.vector_norm(new-original)
                                               / torch.linalg.vector_norm(original))
        })
    return summaries

strength = 0.5
edit_summary = apply_candidate(strength)
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Candidate fixed before causal tests\n"
        "- Selection and complete neuron evidence: candidate_manifest.json.\n"
        "- Sparse down_proj column suppression only; no embeddings, output matrix, "
        "norms, attention, gate_proj or up_proj edited.\n"
        f"- Exact selected columns by layer: {json.dumps(selected_by_layer)}.\n"
        "- Hypothesis: these outputs reinforce invalid numerical transformations. "
        "Association and output geometry alone are not causal proof.\n"
        "\n## Pass 3: paired DEV, half suppression\n"
        "- Exact edit: for each selected layer/column above, "
        "down_proj[:,j] = 0.5 * original[:,j]; all other values unchanged.\n"
        "- Problems: DEV_ids.json; 32 problems x 32 samples/model = 1024/model.\n"
        "- Both models, temperature/top-k/top-p fixed by evaluator; max_tokens=8192.\n"
        "- Accuracy is primary; arithmetic errors and coverage are secondary.\n"
        "- Next planned dose: full removal of exactly the same columns.\n"
    )

import shutil
shutil.copytree("/work/edited", "/work/candidate_half", dirs_exist_ok=True)
print("Selected neurons:", sum(map(len,selected_by_layer.values())))
print("Selected columns by layer:", json.dumps(selected_by_layer))
print("Geometry-screen survivors:", len(reinforcing))
print("Per-tensor changes:", edit_summary)
print("Edited tensor count:", len(edited()))
print("Half-strength candidate backed up; HOLDOUT untouched.")
```
