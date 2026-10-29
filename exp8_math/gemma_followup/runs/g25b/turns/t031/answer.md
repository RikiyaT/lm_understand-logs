Removing the head did not cause a likelihood collapse on either probe, and the whole-text change slightly favored the correct probe relative to the repetitive one. I’ll separate continuation tokens from the shared prompt to check that small effect. Then I’ll test the head ablation on all DEV problems with paired generation. The repair hypothesis survives only if answer accuracy improves, not merely if the constructed probes shift favorably.

```python
lp8=np.load("/work/fwd_8/token_logprobs.npy")
lp9=np.load("/work/fwd_9/token_logprobs.npy")
probe_scores=[]
for i,t in enumerate(attn_texts):
    e=tok.encode(t,add_special_tokens=False)
    start=t.index("<start_of_turn>model\n")+len("<start_of_turn>model\n")
    positions=np.array([j-1 for j,(a,b) in enumerate(e.offsets) if j>0 and b>start])
    late_start=t.index("We need")
    late=np.array([j-1 for j,(a,b) in enumerate(e.offsets) if j>0 and b>late_start])
    final_start=t.rfind("\\boxed{")+len("\\boxed{")
    final=np.array([j-1 for j,(a,b) in enumerate(e.offsets) if j>0 and b>final_start and a<len(t)-1])
    ss={"probe":"correct" if i==0 else "repeated_wrong",
        "continuation_change":float((lp9[i,positions]-lp8[i,positions]).mean()),
        "after_shared_prefix_change":float((lp9[i,late]-lp8[i,late]).mean()),
        "final_answer_change":float((lp9[i,final]-lp8[i,final]).mean())}
    probe_scores.append(ss)
print("Head-ablation likelihood effects:",probe_scores)
with open("/work/notes.md","a") as f:
    f.write("\nExperimenter correction: previous no_answer_finished counts included wrong stated answers; disregard them as estimates of truly absent final answers. Grades/accuracies unchanged.\n"
            "Pass9 head-ablation probe likelihood changes: "+json.dumps(probe_scores)+"\n"
            "## Pass10 planned\nCandidateB unchanged: only L11 self_attn.o_proj columns0:256 zeroed, head0 removal1.0. Both models,DEV63,n16,max_tokens8192. Test accuracy primarily and repetitive-line rate secondarily.\n")
```
