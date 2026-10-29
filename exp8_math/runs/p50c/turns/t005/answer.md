```python
# Hypothesis: recurrent self-correction has a shared activation signature; identify candidate neurons while checking for anomalous attention-head weights.
for j,l in enumerate(idx1["mlp_layers"]):
    diff=M1[~good1,j].mean(0)-M1[good1,j].mean(0)
    sd=np.maximum(M1[:,j].std(0),1e-6)
    ids=np.argsort(np.abs(diff/sd))[-6:][::-1]
    print("MLP",l,[(int(k),round(float(diff[k]/sd[k]),2),round(float(M1[~good1,j,k].mean()),3),round(float(M1[good1,j,k].mean()),3)) for k in ids])
print("Full-attention tensor shapes:")
for n in NAMES:
    if ".layers.23." in n:
        w=load_original(n); print(n, tuple(w.shape), round(float(w.norm()),3),round(float(w.abs().max()),3))
print("Norm ranges:")
for l in range(24):
    ns=[n for n in NAMES if f".layers.{l}." in n and "norm.weight" in n]
    print(l,[(n.split(f".layers.{l}.")[1],round(float(load_original(n).min()),3),round(float(load_original(n).median()),3),round(float(load_original(n).max()),3)) for n in ns])
for i in [3,40]:
    r=g1[i]; t=r["samples"][0]["text"]
    print("PROBLEM",r["id"],byid[r["id"]]["problem"][:290])
    for off in [0,700,1400,2100,2800,3500,4200]:
        print(off, t[off:off+280].replace("\n"," "))
```
