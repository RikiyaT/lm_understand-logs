```python
# Hypothesis: repetition may be amplified by a small set of abnormal attention gates or normalization coordinates, missed by the first MLP ablation.
for l in range(24):
    prefix=f"model.language_model.layers.{l}."
    stats=[]
    for suffix in ["input_layernorm.weight","post_attention_layernorm.weight"]:
        ww=load_original(prefix+suffix)
        stats.append((suffix.split(".")[0],*[round(float(z),3) for z in (ww.min(),ww.median(),ww.max())],
                      torch.topk(ww.abs(),3).indices.tolist()))
    print(l,stats)
    if l%4!=3:
        al=load_original(prefix+"linear_attn.A_log")
        dt=load_original(prefix+"linear_attn.dt_bias")
        zz=load_original(prefix+"linear_attn.norm.weight")
        print("  recurrent A_log",np.round(al.numpy(),2).tolist(),"dt",np.round(dt.numpy(),2).tolist(),
              "gated_norm min/median/max",np.round([zz.min(),zz.median(),zz.max()],3))
    else:
        for suf in ["self_attn.q_norm.weight","self_attn.k_norm.weight"]:
            ww=load_original(prefix+suf)
            print(" ",suf,"min/median/max",np.round([ww.min(),ww.median(),ww.max()],3))
print("TF shapes:", {fn:np.load("/work/fwd_3/"+fn,mmap_mode="r").shape for fn in ["input_ids.npy","hidden_24.npy","mlp_act_23.npy","topk_ids.npy"]})
print("Index metadata:",{k:v for k,v in json.load(open("/work/gen_1_acts/index.json")).items() if k!="samples"})
```
