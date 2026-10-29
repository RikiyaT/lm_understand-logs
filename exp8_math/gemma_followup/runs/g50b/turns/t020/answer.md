Removing the top four neurons gained only 0.88 points with a 1.60-point standard error, so they do not explain a clear accuracy improvement by themselves. I’ll now remove the other twelve while restoring those four. This complementary test can reveal whether the useful effect lies elsewhere in the selected group or depends on their interaction.

```python
def prepare_units(pass_number, layer, units, alpha, rationale):
    clear_edits()
    c = set_unit_edit(layer, units, alpha)
    experiments[pass_number] = dict(c)
    with (work / "notes.md").open("a") as f:
        f.write(f"\nUpcoming pass {pass_number}: BOTH, DEV_ids.json, n=32, "
                "max_tokens=8192. "+rationale+" Exact edit: "+json.dumps(c)+
                ". All other tensors original. HOLDOUT unused.\n")
    return c

candidate = prepare_units(8, 8, units8[4:], 1.0,
                          "Complementary ablation of the remaining 12 selected units.")
print(candidate)
"Complementary ablation documented."
```