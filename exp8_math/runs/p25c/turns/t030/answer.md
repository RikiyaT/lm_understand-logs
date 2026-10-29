```python
# Hypothesis: repeated self-checking has a localized driver distinguishable from neurons that merely report it.
torch.set_num_threads(8)
def flatten(rows):
    return [dict(s, problem_id=r["id"], source=r["source"], reference=r["answer"], sample_idx=i)
            for r in rows for i, s in enumerate(r["samples"])]
flat_gen = {k: flatten(v) for k, v in gen_data.items()}
s0 = flat_gen["gen_1"][0]
print("Sample fields:", list(s0))
print("Sample metadata:", {k:v for k,v in s0.items()
                          if not isinstance(v,(dict,list)) and len(str(v)) < 180})
for tag, samples in flat_gen.items():
    summary = {"samples":len(samples)}
    for key in samples[0]:
        vals = [s.get(key) for s in samples]
        if all(isinstance(v,(bool,int,float)) for v in vals):
            summary[key] = round(float(np.mean(vals)),4)
        elif key not in ("text","response","completion","problem_id","reference") and all(
            isinstance(v,(str,type(None))) and len(str(v))<60 for v in vals):
            summary[key] = dict(Counter(vals))
    print(tag, json.dumps(summary))
for k in (5,6,7,8):
    a,b=flat_gen[f"gen_{k}_original"],flat_gen[f"gen_{k}_edited"]
    if "correct" in a[0]:
        da = np.array([int(y["correct"])-int(x["correct"]) for x,y in zip(a,b)])
        print("PAIRED",k,"gain_pp",round(100*da.mean(),3),"SE_pp",round(100*da.std(ddof=1)/len(da)**.5,3),
              "W->R",int((da==1).sum()),"R->W",int((da==-1).sum()))
text_key = next(k for k in ("text","response","completion") if k in s0)
for s in flat_gen["gen_1"][:6]:
    t=s[text_key]
    print("EXCERPT",s["problem_id"],s["sample_idx"],"correct",s.get("correct"),
          "begin",t[:180].encode("ascii","replace").decode(),
          "end",t[-280:].encode("ascii","replace").decode())

act_dir=W/"gen_1_acts"
act_index=json.loads((act_dir/"index.json").read_text())
acts={p.stem:np.load(p,mmap_mode="r") for p in act_dir.glob("*.npy")}
print("Activation shapes:",{k:(v.shape,str(v.dtype)) for k,v in acts.items()})
fwd_meta=json.loads((W/"fwd_3"/"meta.json").read_text())
print("Teacher-forced metadata:",{k:v for k,v in fwd_meta.items() if k!="texts"})
print("Teacher-forced text endings:",[t[-280:].encode("ascii","replace").decode() for t in fwd_meta["texts"]])

mlp_weights={}
norm_stats=[]
for layer in range(24):
    for proj in ("gate_proj","up_proj","down_proj"):
        name=f"model.language_model.layers.{layer}.mlp.{proj}.weight"
        w=load_original(name)
        mlp_weights[layer,proj]=w
        norms=torch.linalg.vector_norm(w,dim=0 if proj=="down_proj" else 1)
        top=torch.topk(norms,3)
        norm_stats.append((layer,proj,tuple(w.shape),round(norms.median().item(),3),
                           round(norms.max().item()/norms.median().item(),3),
                           list(zip(top.indices.tolist(),[round(x,3) for x in top.values.tolist()]))))
print("Largest neuron norm/median ratios (layer,projection,shape,median,ratio,top3):")
for row in sorted(norm_stats,key=lambda x:x[4],reverse=True)[:22]: print(row)
with (W/"notes.md").open("a") as f:
    f.write("\nRecovered after kernel restart: 6/25 forward passes used; no active edits. "
            "Paired generations gen_5 through gen_8 available; intervention identities for gen_6-8 were not preserved.\n")
    for k in (5,6,7,8):
        a,b=flat_gen[f"gen_{k}_original"],flat_gen[f"gen_{k}_edited"]
        if "correct" in a[0]:
            d=np.array([int(y["correct"])-int(x["correct"]) for x,y in zip(a,b)])
            f.write(f"gen_{k}: original {sum(s['correct'] for s in a)}/{len(a)}, "
                    f"edited {sum(s['correct'] for s in b)}/{len(b)}; "
                    f"paired gain {100*d.mean():.2f} +/- {100*d.std(ddof=1)/len(d)**.5:.2f} pp SE.\n")
```
