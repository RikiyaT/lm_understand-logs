Block 3’s final repetition gained 1.66 percentage points, improving the cumulative estimate, but the last block must still be included. I will log the fiftieth and final pass and audit the exact edited entries before it runs. The audit will verify that only the intended head-output columns and eight MLP columns differ from the original tensors. This checks submission integrity; it is not another tuning step.

```python
prepare_frozen_holdout_pass(50)

scope_audit = []
for name, columns, multiplier in [
    (final_head_name, list(range(512, 768)), 1.1),
    (final_mlp_name, final_units, 0.5)
]:
    original_tensor = load_original(name)
    current_tensor = load(name)
    assert original_tensor.shape == current_tensor.shape
    assert torch.isfinite(current_tensor).all()
    allowed = torch.zeros_like(original_tensor, dtype=torch.bool)
    allowed[:, columns] = True
    assert torch.equal(current_tensor[~allowed], original_tensor[~allowed])
    assert torch.allclose(current_tensor[allowed],
                          original_tensor[allowed] * multiplier,
                          rtol=1e-6, atol=1e-8)
    scope_audit.append({
        "tensor": name,
        "shape": list(current_tensor.shape),
        "multiplier_on_selected_columns": multiplier,
        "changed_values": int(torch.count_nonzero(current_tensor != original_tensor)),
        "unchanged_outside_selected_columns": True,
        "relative_F_change": float(
            torch.linalg.vector_norm(current_tensor - original_tensor)
            / torch.linalg.vector_norm(original_tensor))
    })
assert verify_frozen_candidate()
(work / "submission_scope_audit.json").write_text(json.dumps(scope_audit, indent=2))
with (work / "notes.md").open("a") as f:
    f.write("Pre-final-pass scope audit: " + json.dumps(scope_audit)
            + ". Only the two frozen tensors are edited; other tensors original.\n")
print("SUBMISSION_SCOPE_AUDIT", scope_audit)
"Final pass logged; tensor scopes, shapes, finite values, and frozen hashes verified."
```