Layer 11 head 0 gives the strongest recent-token emphasis at the final-number query: 51.5% of attention goes to the last eight tokens in the wrong probe, versus 42.0% in the correct probe. This may reflect answer formatting rather than harmful copying, so it remains only a hypothesis. I’ll isolate that head by zeroing its 256-column block in `self_attn.o_proj`, removing the previous MLP edits. The causal prediction is reduced reinforcement of repetitive conclusions without destroying the correct calculation.

```python
def apply_head_edit(layer, head, removed_fraction):
    for name in list(edited()):
        drop_edit(name)
    name=f"model.layers.{layer}.self_attn.o_proj.weight"
    w=load_original(name)
    a,b=head*256,(head+1)*256
    w[:,a:b]*=(1-removed_fraction)
    save_edit(name,w)
    desc={name:{"head":head,"column_start":a,"column_stop":b,"removed_fraction":removed_fraction}}
    json.dump(desc,open("/work/current_edit.json","w"))
    return desc
edit_desc=apply_head_edit(11,0,1.0)
with open("/work/notes.md","a") as f:
    f.write("\nAttention probe: L11H0 recent8 mass at final-number query is0.420 correct versus0.515 repetitive wrong; strongest recent emphasis among full-attention heads. Potential syntax confound acknowledged; no causal conclusion yet.\n"
            "## Pass9 planned\nCandidate B: only model.layers.11.self_attn.o_proj.weight columns0:256 zeroed (head0 removed, strength1). All MLP edits removed. Teacher-force the same2 attention probes as pass8; return layer17 attention (2-text tool limit), compare correct-computation versus repetitive-continuation likelihood.\n"
            +json.dumps(edit_desc)+"\n")
print("Current head intervention:",edit_desc)
```
