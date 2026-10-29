The existing activation files cover every MLP layer, so no new forward pass is needed for this screen. I will rank neurons by their association with repeated lines after removing problem-level differences, length, cutoffs, and correctness. Requiring agreement across two disjoint halves of DEV should reduce the chance of selecting an accidental correlation. This remains observational evidence; a small ablation must subsequently change the predicted behavior and improve accuracy.

```python
import pandas as pd, numpy as np, json

idx = json.loads((work / "gen_1_acts" / "index.json").read_text())
print("Activation index:", type(idx).__name__,
      list(idx.keys()) if isinstance(idx, dict) else {"count": len(idx), "first": idx[0]})
print("Existing labels:", {"y_mean": float(np.mean(y)), "repeat_mean": float(np.mean(rep)),
                          "cut_mean": float(np.mean(cut)), "length_range": [float(np.min(length)), float(np.max(length))]})

# These are diagnostic regressions for nuisance removal, not model training.
def within_problem(a, ids):
    b = np.asarray(a, dtype=np.float64).copy()
    for p in np.unique(ids):
        m = ids == p
        b[m] -= b[m].mean(axis=0)
    return b

controls = within_problem(np.column_stack([np.log1p(length), cut, y]), pid)
qc, _ = np.linalg.qr(controls)
r0 = within_problem(rep, pid)
rres = r0 - qc @ (qc.T @ r0)
split_ids = set(np.random.default_rng(381).permutation(np.unique(pid))[::2])
splitmask = np.array([p in split_ids for p in pid])
rep_rows = []
for layer in range(18):
    xx = within_problem(np.asarray(Xall[sel, layer, :]), pid)
    xr = xx - qc @ (qc.T @ xx)
    scores = []
    for m in [np.ones(len(pid), dtype=bool), splitmask, ~splitmask]:
        numerator = rres[m] @ xr[m]
        denominator = np.linalg.norm(rres[m]) * np.linalg.norm(xr[m], axis=0) + 1e-12
        scores.append(numerator / denominator)
    outnorm = load_original(f"model.layers.{layer}.mlp.down_proj.weight").norm(dim=0).numpy()
    for neuron in range(2048):
        rep_rows.append({"layer": layer, "neuron": neuron,
                        "r": float(scores[0][neuron]),
                        "half1": float(scores[1][neuron]), "half2": float(scores[2][neuron]),
                        "stable": float(min(scores[1][neuron], scores[2][neuron])),
                        "output_effect_sd": float(np.std(xx[:, neuron]) * outnorm[neuron])})
repscreen = pd.DataFrame(rep_rows).sort_values("stable", ascending=False)
repscreen.to_json(work / "repetition_neuron_screen.json", orient="records")
print("Top stable repetition associations:\n", repscreen.head(15).round(4).to_string(index=False))
print("Top late-layer associations:\n",
      repscreen[repscreen.layer >= 12].head(8).round(4).to_string(index=False))
```
