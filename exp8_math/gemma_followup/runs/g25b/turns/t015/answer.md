The baseline teacher-forced pass succeeded for every DEV problem. I’ll now test a specific candidate: remove the output contributions of 32 failure-associated MLP neurons in each of layers 4, 15, and 17, selected for consistent association across the two DEV halves. This changes only selected `down_proj` columns, not whole tensors or output weights. If these components help sustain failed continuations, removing them should preserve correct-answer likelihood and improve sampled accuracy; harm on either measure would weaken that hypothesis.

```python
print("Forward metadata keys:",list(json.load(open("/work/fwd_3/meta.json"))))
base_lp=np.load("/work/fwd_3/token_logprobs.npy")
base_ids=np.load("/work/fwd_3/input_ids.npy")
print("Length validation:",sum(int((base_ids[i]!=0).sum())==compact_info[i]["tokens"] for i in range(63)),
      "local lengths",[x["tokens"] for x in compact_info[:4]])
masks=[]
for i,t in enumerate(compact_texts):
    enc=tok.encode(t)
    ans_start=t.rfind("\\boxed{")+len("\\boxed{")
    answer_pos=[j for j,(a,b) in enumerate(enc.offsets) if b>ans_start and a<len(t)-1 and j>0]
    masks.append(np.array(answer_pos,dtype=int)-1)
baseline_ans_scores=np.array([float(base_lp[i,m].mean()) for i,m in enumerate(masks)])
print("Baseline answer-content mean logprob:",round(float(baseline_ans_scores.mean()),4))
chosen_neurons={}
for l in [4,15,17]:
    candidates=[s for s in stats if s["layer"]==l and s["raw"]>0 and min(s["half1"],s["half2"])>0]
    candidates.sort(key=lambda s:min(s["half1"],s["half2"]),reverse=True)
    chosen_neurons[l]=[s["neuron"] for s in candidates[:32]]
    a=np.load(f"/work/fwd_3/mlp_act_{l}.npy",mmap_mode="r")
    print("Layer",l,"selected neurons",chosen_neurons[l],
          "median failure corr",round(float(np.median([s["corr"] for s in candidates[:32]])),3))
json.dump(chosen_neurons,open("/work/failure_neurons_96.json","w"))
def apply_neuron_edit(strength, layers=(4,15,17), width=32):
    for name in list(edited()):
        drop_edit(name)
    desc={}
    for l in layers:
        name=f"model.layers.{l}.mlp.down_proj.weight"
        w=load_original(name)
        js=chosen_neurons[l][:width]
        w[:,js]*=(1-strength)
        save_edit(name,w)
        desc[name]={"columns":js,"removed_fraction":strength}
    return desc
edit_desc=apply_neuron_edit(1.0)
json.dump(edit_desc,open("/work/current_edit.json","w"))
with open("/work/notes.md","a") as f:
    f.write("\n## Candidate A / pass 4 planned\n"
            "DEV only. Zero 32 selected output columns in each MLP down_proj at layers 4,15,17 (96 neurons total); strengths=1.0 removal. Selection: positive raw failure association, ranked by smaller of split-half length/cut-controlled correlations. Exact edit:\n"
            +json.dumps(edit_desc)+"\n"
            "Teacher-forced pass 4 repeats all 63 compact DEV texts from pass 3. Compare answer-content logprobs before full-size paired generation.\n")
print("Edited tensors:",edited())
```
