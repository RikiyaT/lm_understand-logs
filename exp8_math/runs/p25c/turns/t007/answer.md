```python
# Hypothesis: the leading late-layer neurons encode repetitive self-doubt rather than mathematical content; inspect their weight geometry and prepare matched token-level probes.
for l,j in [(23,22),(23,94),(22,59),(23,19),(17,20),(15,0),(18,29)]:
    p=f"model.language_model.layers.{l}.mlp."
    gate=weights[p+"gate_proj.weight"]
    up=weights[p+"up_proj.weight"]
    down=weights[p+"down_proj.weight"]
    print("candidate",l,j,"gate/up/down norms",
          [round(float(v),4) for v in (gate[j].norm(),up[j].norm(),down[:,j].norm())],
          "gate norm percentile",round(float((gate.norm(dim=1)<gate[j].norm()).float().mean()),3))
print("Model directory:",os.listdir("/model"))
by_id={r["id"]:r for rows in practice.values() for r in rows}
print("Good speed solution beginning:",repr(samples1[10]["text"][:280]))
print("Good speed solution middle:",repr(samples1[10]["text"][2300:2580]))
print("Problem for matched speed probes:",by_id[samples1[10]["id"]]["problem"])
print("Second short successful problem:",by_id[samples1[19]["id"]]["problem"])
print("Second solution beginning:",repr(samples1[19]["text"][:280]))
print("Second solution end:",repr(samples1[19]["text"][-280:]))
print("Prompt activity of candidate 23:22:",np.quantile(Mp[:,23,22],[0,.25,.5,.75,1]).tolist())
print("Mean-answer activity of candidate 23:22, correct and cut:",
      np.quantile(M[good,23,22],[0,.25,.5,.75,1]).tolist(),
      np.quantile(M[cut,23,22],[0,.25,.5,.75,1]).tolist())
```
