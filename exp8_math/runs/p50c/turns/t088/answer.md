Generation failed during engine initialization again, so it provides no evidence for or against strength 4. I’ll use the working teacher-forced path to test a negative control: unrelated directions, orthogonal to the repetition directions, with the same per-layer edit magnitude. This isolates whether the observed selectivity depends on the identified direction rather than simply perturbing two MLPs. The confirmed strength-2 weights remain backed up and will not be replaced by an unvalidated control.

```python
# Hypothesis: a norm-matched random-direction edit will not reproduce the repetition-specific effect.
u_rep = torch.load("/work/repetition_directions.pt", weights_only=False)
control_checks = {}
for layer in (15, 19):
    name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    w0 = load_original(name)
    u = u_rep[layer].float()
    v = random_control[layer]["direction"].float()
    a = float(random_control[layer]["alpha"])
    target_delta = -4.0 * torch.outer(u, u @ w0)
    control_delta = -a * torch.outer(v, v @ w0)
    save_edit(name, w0 + control_delta)
    control_checks[layer] = {
        "cosine_to_repetition": float(torch.dot(u, v)),
        "relative_change": float(control_delta.norm() / w0.norm()),
        "matched_norm_ratio": float(control_delta.norm() / target_delta.norm()),
        "finite": bool(torch.isfinite(w0 + control_delta).all())
    }
print(control_checks)
print("EDITED", edited())
with open("/work/notes.md", "a") as f:
    f.write("\nAlpha4 generation retry failed initialization (no pass). "
            "CURRENT: random-direction control matched to alpha4, not a candidate for submission. "
            "Checks: " + json.dumps(control_checks) + "\n")
```
