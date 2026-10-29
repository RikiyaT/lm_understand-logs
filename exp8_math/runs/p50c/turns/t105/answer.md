At the shorter cap, the full strength-4 edit produced essentially the same exploratory gain as the restricted edit: +26.56 percentage points, with no paired regressions. That does not establish equivalence, but it gives no clear reason yet to prefer the broader change. I’ll test strength 6 on the same 64-neuron subsets, which increases the targeted intervention while leaving the other MLP columns untouched. The hypothesis is that stronger suppression within those neurons can improve completion without the broader edit’s collateral effects.

```python
# Hypothesis: increasing only the selected repetition-writer columns yields a better benefit/side-effect tradeoff.
stats42 = {m: generation_file_stats(f"/work/gen_42_{m}.jsonl") for m in ("original", "edited")}
print("FULL alpha4,16k", stats42)
def set_subset_edit(alpha):
    for l in (15, 19):
        w = base_mlp_weights[l]
        u = u_rep[l].float()
        writer = u @ w
        masked = torch.zeros_like(writer)
        ix = writer_subsets[l].long()
        masked[ix] = writer[ix]
        save_edit(f"model.language_model.layers.{l}.mlp.down_proj.weight",
                  w - float(alpha) * torch.outer(u, masked))
set_subset_edit(6)
shutil.copytree("/work/edited", "/work/candidate_top64_a6", dirs_exist_ok=True)
for l in (15, 19):
    name = f"model.language_model.layers.{l}.mlp.down_proj.weight"
    w = load(name)
    d = w - base_mlp_weights[l]
    print("L", l, "relative_change", float(d.norm()/base_mlp_weights[l].norm()),
          "changed_columns", int((d.abs().amax(dim=0) > 0).sum()), "finite", bool(torch.isfinite(w).all()))
with open("/work/notes.md", "a") as f:
    f.write("\nPass28/gen42 FULLalpha4 exploratory64x2 at16384:14/128->48/128,+26.56pp SE4.59, "
            "34rescues0losses; cut89.06->52.34%; " + json.dumps(stats42) +
            "\nCURRENT top64alpha6, backup candidate_top64_a6; next same16k generation.\n")
```
