```python
# Hypothesis: the failed diagnostic request was an instrumentation limitation; inspect partial outputs before reducing requested attention layers.
print("Work entries:", sorted(os.listdir("/work")))
for p in sorted(Path("/work").glob("fwd*")):
    print(p.name, [(x.name, x.stat().st_size) for x in p.iterdir()][:40])
print("Activation magnitudes by neuron-index block (late layers):")
for l in [11,18,19,22,23]:
    a=idx1["mlp_layers"].index(l)
    print(l,[(s,round(float(np.abs(M1[:,a,s:s+64]).mean()),4)) for s in [0,64,128,512,1024,2048,3072]])
```
