```python
# Hypothesis: the four-channel correlation was not causal; a broader weight/activation audit may reveal a more specific repetition feedback circuit.
for name in list(edited()):
    drop_edit(name)
print("Edits cleared:", edited())
print("Layer 18 tensor names:", [n for n in NAMES if ".layers.18." in n])
print("Layer 19 tensor names:", [n for n in NAMES if ".layers.19." in n])
print("Teacher-forced files:",os.listdir("/work/fwd_3"))
print("TF metadata:",open("/work/fwd_3/meta.json").read()[:2400])
print("Generation activation files:",os.listdir("/work/gen_1_acts")[:35])
print("Score object:",type(scores).__name__,repr(scores)[:1300])
print("Selected global shapes:",{k:(tuple(v.shape) if hasattr(v,"shape") else type(v).__name__) for k,v in globals().copy().items() if k in ["M1","E","W","direction","delta","down_cache","mean_good","mean_bad","FROZEN"]})
for l in range(24):
    n=f"model.language_model.layers.{l}.mlp.down_proj.weight"
    ww=load_original(n)
    cn=ww.norm(dim=0)
    print(l,"down norm",round(float(ww.norm()),2),"column norms median/max",round(float(cn.median()),3),round(float(cn.max()),3),
          "top columns",torch.topk(cn,4).indices.tolist())
with open("/work/notes.md","a") as f:
    f.write("Four-channel ablation rejected: no repetition reduction or accuracy gain. All edits cleared. Seven forward passes remain.\n")
```
