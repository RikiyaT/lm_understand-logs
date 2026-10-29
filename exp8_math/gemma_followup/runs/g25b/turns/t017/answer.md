The edited teacher-forced pass completed, and I’ll now compare only the actual answer-content tokens using verified token alignment. This will separate mathematical answer likelihood from prompt and formatting likelihood. If the screen shows no broad collapse, I’ll test the same candidate with 1,008 paired generations at the full token limit. Accuracy, not shorter answers or fewer cutoffs, will determine whether it is useful.

```python
masks=[]; continuation_masks=[]
meta3=json.load(open("/work/fwd_3/meta.json"))
for i,t in enumerate(compact_texts):
    enc=tok.encode(t,add_special_tokens=False)
    n=len(enc.ids)
    assert n==meta3["lengths"][i]
    assert np.array_equal(base_ids[i,:n],enc.ids)
    ans_start=t.rfind("\\boxed{")+len("\\boxed{")
    answer_pos=[j for j,(a,b) in enumerate(enc.offsets) if b>ans_start and a<len(t)-1 and j>0]
    masks.append(np.array(answer_pos,dtype=int)-1)
    cont_start=t.index("<start_of_turn>model\n")+len("<start_of_turn>model\n")
    cp=[j for j,(a,b) in enumerate(enc.offsets) if b>cont_start and j>0]
    continuation_masks.append(np.array(cp,dtype=int)-1)
edit_lp=np.load("/work/fwd_4/token_logprobs.npy")
baseline_ans_scores=np.array([base_lp[i,m].mean() for i,m in enumerate(masks)])
edited_ans_scores=np.array([edit_lp[i,m].mean() for i,m in enumerate(masks)])
dd=edited_ans_scores-baseline_ans_scores
print("Answer-content logprob original/edited:",float(baseline_ans_scores.mean()),float(edited_ans_scores.mean()))
print("Change nats/token:",float(dd.mean()),"SE",float(dd.std(ddof=1)/np.sqrt(len(dd))),
      "improved problems",int((dd>0).sum()),"of",len(dd))
cd=np.array([(edit_lp[i,m]-base_lp[i,m]).mean() for i,m in enumerate(continuation_masks)])
print("Whole continuation mean change:",float(cd.mean()))
print("Weight relative Frobenius changes:",
      {name:round(float((load(name)-load_original(name)).norm()/load_original(name).norm()),4) for name in edited()})
with open("/work/notes.md","a") as f:
    f.write(f"\nPass 4 A full-ablation screen: answer-content mean logprob change {dd.mean():.6f} nats/token, SE {dd.std(ddof=1)/np.sqrt(len(dd)):.6f}; {int((dd>0).sum())}/63 improved. Verified encoding with add_special_tokens=False.\n"
            "## Pass 5 planned\nCandidate A unchanged: exact columns in failure_neurons_96.json, layers4/15/17 down_proj, removal=1.0. Both models, DEV63, n16 (1008/model), max_tokens8192. Primary endpoint accuracy; secondary repetition/cut rates.\n")
```
