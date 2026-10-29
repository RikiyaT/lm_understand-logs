```python
# Hypothesis: a small set of unusually strong MLP or attention directions amplifies otherwise ordinary self-checking into persistent loops.
print("Weights cached:",len(weights))
for l in range(24):
    p=f"model.language_model.layers.{l}."
    row={"layer":l}
    for suffix in ["input_layernorm.weight","post_attention_layernorm.weight","linear_attn.A_log","linear_attn.dt_bias","linear_attn.norm.weight","self_attn.q_norm.weight","self_attn.k_norm.weight"]:
        if p+suffix in weights:
            w=weights[p+suffix]
            row[suffix.replace(".weight","")]=[round(float(w.min()),3),round(float(w.median()),3),round(float(w.max()),3)]
    print(row)
print("Matrix outliers:")
for name,shape,rms,mx in sorted([x for x in stats if len(x[1])>1],key=lambda x:x[3]/max(x[2],1e-12),reverse=True)[:10]:
    print(name.split(".layers.")[1],shape,round(rms,5),round(mx,4),round(mx/rms,1))

H=np.load("/work/gen_1_acts/hidden_mean_answer.npy")
M=np.load("/work/gen_1_acts/mlp_mean_answer.npy")
Hp=np.load("/work/gen_1_acts/hidden_last_prompt.npy")
Mp=np.load("/work/gen_1_acts/mlp_last_prompt.npy")
good=np.array([s["correct"] for s in samples1])
cut=np.array([s["cut"] for s in samples1])
print("Residual RMS by layer, correct / cutoff:")
print(np.round(np.stack([np.sqrt(np.mean(H[good]**2,axis=(0,2))),np.sqrt(np.mean(H[cut]**2,axis=(0,2)))],axis=1),3))
print("MLP mean-activation RMS by layer, correct / cutoff:")
print(np.round(np.stack([np.sqrt(np.mean(M[good]**2,axis=(0,2))),np.sqrt(np.mean(M[cut]**2,axis=(0,2)))],axis=1),3))
print("Good sample indices:",np.flatnonzero(good).tolist())
print("Mixed-outcome problem pairs:",sum(good[i]!=good[i+1] for i in range(0,len(good),2)))
```
