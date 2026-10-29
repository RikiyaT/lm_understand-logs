```python
# Hypothesis: excessive repetition is sustained by a small high-gain circuit, detectable as anomalous weight/activation concentrations rather than general math difficulty.
torch.set_num_threads(8)
print("Activation arrays:", M1.shape, H1.shape, "correct:", int(y1.sum()))
summary = []
for name in NAMES:
    if ".layers." not in name:
        continue
    w = load_original(name)
    summary.append((name,tuple(w.shape),float(w.square().mean().sqrt()),float(w.abs().max()),
                    float(w.abs().max()/(w.square().mean().sqrt()+1e-12))))
for suffix in ["mlp.down_proj.weight","mlp.gate_proj.weight","mlp.up_proj.weight",
               "self_attn.o_proj.weight","linear_attn.out_proj.weight",
               "input_layernorm.weight","post_attention_layernorm.weight","linear_attn.A_log","linear_attn.dt_bias"]:
    print("\n",suffix)
    print([(int(n.split(".layers.")[1].split(".")[0]), round(r,4),round(m,3),round(z,1))
           for n,sh,r,m,z in summary if n.endswith(suffix)])
mean_bad = M1[~y1].mean(0)
mean_good = M1[y1].mean(0)
pooled = np.sqrt((M1[~y1].var(0)+M1[y1].var(0))/2 + 1e-8)
effect = (mean_bad-mean_good)/pooled
topinds = np.argsort(np.abs(effect).ravel())[-25:][::-1]
print("\nLargest activation effect sizes (layer, neuron, effect, bad mean, good mean):")
for k in topinds:
    a,b = np.unravel_index(k,effect.shape)
    print(idx1["mlp_layers"][a],int(b),round(float(effect[a,b]),2),round(float(mean_bad[a,b]),3),round(float(mean_good[a,b]),3))
```
