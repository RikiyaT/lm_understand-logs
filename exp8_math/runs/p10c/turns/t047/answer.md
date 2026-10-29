```python
# Hypothesis: full-attention copying of a previous self-check may sustain loops; inspect natural failures and the existing probe layout.
for l in [15,18,19,22,23]:
    p=f"model.language_model.layers.{l}."
    if l%4!=3:
        for suf in ["linear_attn.A_log","linear_attn.dt_bias"]:
            z=load_original(p+suf).numpy()
            print(l,suf,np.round(z,2).tolist())
for fn in ["input_ids.npy","hidden_24.npy","mlp_act_23.npy","topk_ids.npy"]:
    arr=np.load("/work/fwd_3/"+fn,mmap_mode="r")
    print(fn,arr.shape,arr.dtype)
print("gen1 index:",{k:v for k,v in json.load(open("/work/gen_1_acts/index.json")).items() if k!="samples"})
for ss in [s for s in s4o if s["cut"]][:3]:
    print("Failure",ss["id"])
    for label,excerpt in [("start",ss["text"][:250]),("middle",ss["text"][7000:7250]),("tail",ss["text"][-250:])]:
        print(label,excerpt.encode("ascii","replace").decode())
print("Practice object:",type(practice).__name__, "by_id count:",len(by_id),"diagnostic count:",len(diag_ids))
print("Notes update: first ablation was null; attention induction and distributed repetition directions remain untested.")
```
