The random control produced only a 1.56-point paired accuracy change, smaller than its 3.52-point standard error, and did not reduce cutoffs. Together with the dense probe, this argues against edit magnitude alone explaining the targeted gains. I’ll remove the control and restore strength 4, then test it on the same exploratory problem set. A substantially larger paired accuracy gain would connect the direction-specific probe effect to useful behavior.

```python
# Hypothesis: targeting the repetition direction, unlike an equal-sized random edit, improves completed math answers.
base_mlp_weights = {l: load_original(f"model.language_model.layers.{l}.mlp.down_proj.weight")
                    for l in (15, 19)}
def set_repetition_edit(strengths):
    for l in (15, 19):
        name = f"model.language_model.layers.{l}.mlp.down_proj.weight"
        a = float(strengths.get(l, 0))
        if a == 0:
            if name in edited():
                drop_edit(name)
        else:
            w = base_mlp_weights[l]
            u = u_rep[l].float()
            save_edit(name, w - a * torch.outer(u, u @ w))
set_repetition_edit({15: 4, 19: 4})
print("Restored alpha4:", edited())
print("GEN29 SAMPLE KEYS", list(gen29[0]["samples"][0]))
for row in gen29[:2]:
    s = row["samples"][0]
    print(row["id"], {k: v for k, v in s.items()
                      if not isinstance(v, (list, dict))
                      and (not isinstance(v, str) or len(v) < 80)})
with open("/work/notes.md", "a") as f:
    f.write("\nPass23/gen35 RANDOM CONTROL paired64x2: 14/128->16/128; "
            "+1.56pp SE3.52pp,8 rescues6 losses, cut89.06->91.41%, "
            "mean29646->30405. Control REMOVED; CURRENT targeted alpha4 restored. "
            "Matched-control evidence supports direction specificity.\n")
```
