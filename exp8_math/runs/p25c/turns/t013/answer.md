```python
# Hypothesis: the failed generation was an engine-startup issue rather than evidence about the edit; verify the saved tensor before retrying.
print("Current edits:",edited())
name="model.language_model.layers.23.mlp.down_proj.weight"
w=load(name)
w0=load_original(name)
print("Shape:",tuple(w.shape),"finite:",bool(torch.isfinite(w).all()),
      "changed columns:",torch.where((w-w0).abs().amax(dim=0)>0)[0].tolist(),
      "column 22 norm:",float(w[:,22].norm()))
print("Work artifacts:",sorted(os.listdir("/work"))[:35])
print("Persistent diagnostic data:",len(samples1),M.shape)
```
