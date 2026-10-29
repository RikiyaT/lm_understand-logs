```python
# Hypothesis: neuron 19:11 is one contributor to a distributed loop-writing direction; removing that direction from layer 19's MLP should outperform one-neuron ablation.
def clear_edits():
    for n in list(edited()):
        drop_edit(n)
def mlp_direction_edit(layers, strength=1.0):
    clear_edits()
    changes=[]
    for l in layers:
        d=torch.from_numpy(diffH[min(l+1,23)].copy()).float()
        d=d/d.norm()
        w=downweights[l]
        new=w-strength*torch.outer(d,d@w)
        name=f"model.language_model.layers.{l}.mlp.down_proj.weight"
        save_edit(name,new)
        changes.append((l,round(float((new-w).norm()/w.norm()),4)))
    return changes
print("Directional edit:",mlp_direction_edit([19],1.0))
with open("/work/notes.md","a") as f:
    f.write("Pass4 gen_5 paired32: ablate L19 neuron11, +6.25pp +/-4.35pp SE (3->5 correct), cutoff90.6->87.5%; weak positive, not confirmed. Trying rank-one removal of the matched repetition direction from L19 MLP only. Attention-output tool fails even for one raw text/layer.\n")
```
