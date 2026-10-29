The final batch of this split gained 48.4 percentage points, with 31 rescues and no regressions. I will summarize the split separately and verify that the submitted tensors still match the frozen hashes and sparse edit structure. Any unexpected tensor or hash change would undermine the confirmation, so this audit checks the actual weights rather than relying only on the experiment log.

```python
# Hypothesis: All confirmation results correspond to the exact same sparse directional intervention.
import torch

register_frozen_run(61)
final1_runs = [x["run"] for x in frozen_eval_completed if x["label"].startswith("final_1_")]
print("FINAL_1 ONLY", paired_run_summary(final1_runs))

expected_names = {f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in [15, 19]}
assert set(edited()) == expected_names
frozen_audit = json.loads(Path("/work/frozen_edit_audit.json").read_text())
weight_manifest = {}
for layer in [15, 19]:
    name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    w, w0 = load(name), load_original(name)
    digest = hashlib.sha256(w.contiguous().numpy().tobytes()).hexdigest()
    assert digest == frozen_audit[str(layer)]["sha256"], (layer, digest)
    delta = w-w0
    columns = torch.nonzero(delta.abs().amax(dim=0) > 0).flatten()
    assert columns.numel() == 64
    assert torch.isfinite(w).all()
    v = delta[:, columns[0]]
    v = v / torch.linalg.vector_norm(v)
    directional_residual = torch.linalg.vector_norm(delta-torch.outer(v, v@delta))/torch.linalg.vector_norm(delta)
    scalar = (w*w0).sum()/(w0*w0).sum()
    nonscalar_fraction = torch.linalg.vector_norm(w-scalar*w0)/torch.linalg.vector_norm(delta)
    assert directional_residual < 1e-5 and nonscalar_fraction > .99
    entry = {
        "name": name, "sha256": digest, "shape": list(w.shape),
        "selected_neurons": columns.tolist(),
        "changed_values": int(torch.count_nonzero(delta)),
        "relative_frobenius_change": float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(w0)),
        "directional_residual_fraction": float(directional_residual),
        "nonscalar_fraction": float(nonscalar_fraction)
    }
    weight_manifest[str(layer)] = entry
    print("AUDIT", layer, {k:v for k,v in entry.items() if k not in ["selected_neurons", "name"]})
Path("/work/frozen_weights_manifest.json").write_text(json.dumps(weight_manifest, indent=2))
with open("/work/notes.md", "a") as f:
    f.write("\nPass42 weight audit: hashes unchanged; only64 columns/layer; "
            "rank-one directional delta residual<1e-5; not whole-tensor scaling. "
            "Selected neuron IDs and hashes in frozen_weights_manifest.json. "
            "Final1 summary: " + json.dumps(paired_run_summary(final1_runs)) + "\n")
```
