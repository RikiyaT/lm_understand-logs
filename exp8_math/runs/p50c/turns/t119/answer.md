The second validation batch failed at initialization, so it adds no result. I’ll keep the candidate unchanged and use this pause to audit exactly what would be submitted: two tensors, 64 changed neuron columns per tensor, and only the identified directional component within those columns. I’ll also record checksums so later validation cannot accidentally mix different edits. This audit should catch any unintended tensor changes before further confirmation runs.

```python
# Hypothesis: the confirmed repair is exactly a two-site, 64-column directional edit with no collateral tensor changes.
import hashlib
audit = {}
allowed_names = {f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in (15, 19)}
assert set(edited()) == allowed_names
for l in (15, 19):
    name = f"model.language_model.layers.{l}.mlp.down_proj.weight"
    w = load(name)
    w0 = base_mlp_weights[l]
    u = u_rep[l].float()
    ix = writer_subsets[l].long()
    delta = w-w0
    changed = torch.nonzero(delta.abs().amax(dim=0) > 0).flatten()
    assert torch.equal(torch.sort(changed).values, torch.sort(ix).values)
    expected = torch.zeros_like(w)
    expected[:, ix] = -6 * torch.outer(u, (u @ w0)[ix])
    relative_error = float((delta-expected).norm()/expected.norm())
    parallel = torch.outer(u, (u @ delta)/torch.dot(u, u))
    audit[l] = {"shape": list(w.shape), "changed_columns": len(changed),
                "changed_values": int((delta != 0).sum()),
                "relative_frobenius_change": float(delta.norm()/w0.norm()),
                "formula_relative_error": relative_error,
                "off_direction_fraction": float((delta-parallel).norm()/delta.norm()),
                "finite": bool(torch.isfinite(w).all()),
                "sha256": hashlib.sha256(w.numpy().tobytes()).hexdigest()}
    assert relative_error < 1e-5 and audit[l]["finite"]
print(json.dumps(audit, indent=2))
Path("/work/frozen_edit_audit.json").write_text(json.dumps(audit, indent=2))
with open("/work/notes.md", "a") as f:
    f.write("\nSecond validation_3 half generation failed initialization (not counted). "
            "Frozen top64alpha6 audit passed; hashes in frozen_edit_audit.json; "
            "only selected64 columns in L15/L19 changed, directional formula errors<1e-5. "
            "Next retry exact second-half64x1 FULL32768; successful confirmed_runs=[50].\n")
```
