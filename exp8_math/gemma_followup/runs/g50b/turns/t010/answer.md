The baseline reference pass is complete. I’ll now test a localized intervention: suppress a small set of layer 8 MLP neurons whose activity is consistently larger in failures, rather than using the rejected broad directions. I’ll record the exact neuron indices and first check whether the intervention damages reference-answer likelihood. If it survives that screen, paired generation at multiple strengths will determine whether the association is causal and useful.

```python
meta2 = json.loads((work / "fwd_2/meta.json").read_text())
print("Teacher-forced metadata keys:", sorted(meta2))
print("Tokenizer type:", type(tok).__name__)
print("Metadata summary:", {k:v for k,v in meta2.items()
                            if k != "texts" and len(str(v)) < 1200})

li8 = mlp_layers.index(8)
mean_good8 = X[Y,li8].mean(0)
mean_bad8 = X[~Y,li8].mean(0)
same_half_sign = (
    adjusted[split[:16],li8].mean(0) *
    adjusted[split[16:],li8].mean(0) > 0
)
valid8 = (
    same_half_sign
    & (np.sign(contrasts[:,li8].mean(0)) == np.sign(adjusted[:,li8].mean(0)))
    & (np.abs(mean_bad8) > np.abs(mean_good8))
    & (np.abs(tstat[li8]) > 2.5)
)
units8 = np.where(valid8)[0]
units8 = units8[np.argsort(-np.abs(tstat[li8,units8]))][:16].tolist()
print("Layer-8 selected units:", units8)
print("Selection statistics:", [
    {"unit":j,"t":round(float(tstat[li8,j]),2),
     "good_mean":round(float(mean_good8[j]),4),
     "bad_mean":round(float(mean_bad8[j]),4)}
    for j in units8
])
(work / "layer8_units.json").write_text(json.dumps(units8))

def clear_edits():
    for name in list(edited()):
        drop_edit(name)

def set_unit_edit(layer, units, alpha):
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    W = load_original(name).clone()
    W[:, units] *= (1 - alpha)
    save_edit(name, W)
    return {"tensor":name, "columns":list(units),
            "suppression_alpha":alpha,
            "relative_frobenius_change":float(
                (W-load_original(name)).norm()/load_original(name).norm())}

clear_edits()
candidate = set_unit_edit(8, units8, 1.0)
with (work / "notes.md").open("a") as f:
    f.write("\nUpcoming pass 3: edited compact-reference teacher-forcing, all "
            "32 fixed DEV prompts, identical to pass 2. Exact edit: "
            + json.dumps(candidate) + ". All other tensors original.\n")
print("Exact current edit:", json.dumps(candidate))
"First causal candidate installed; HOLDOUT unused."
```