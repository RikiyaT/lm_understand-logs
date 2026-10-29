The edited pass has higher whole-text likelihood on most references, but that includes the prompts and is not yet a reasoning result. I will now isolate the solution tokens and digit-containing tokens, excluding the plotting-code appendix. If those improve across problems, I will proceed to a full paired generation test of the eight-unit ablation. If improvement is confined to prompt or formatting tokens, the candidate will have much weaker support.

```python
m23=json.loads((W/"fwd_23"/"meta.json").read_text())
assert m23["texts"]==refs_actual
lp22=np.load(W/"fwd_22"/"token_logprobs.npy")
lp23=np.load(W/"fwd_23"/"token_logprobs.npy")
ref_deltas=[]; digit_deltas=[]; ref_masks=[]
for i,t in enumerate(refs_actual):
    e=tok.encode(t,add_special_tokens=False)
    start=t.index("<start_of_turn>model\n")+len("<start_of_turn>model\n")
    end=t.find("[asy]",start)
    if end<0: end=len(t)
    mask=np.array([j-1 for j,(a,b) in enumerate(e.offsets)
                    if j>0 and a>=start and b<=end],int)
    dm=np.array([j-1 for j,(a,b) in enumerate(e.offsets)
                  if j>0 and a>=start and b<=end and any(c.isdigit() for c in t[a:b])],int)
    ref_masks.append(mask)
    ref_deltas.append(float((lp23[i,mask]-lp22[i,mask]).mean()))
    digit_deltas.append(float((lp23[i,dm]-lp22[i,dm]).mean()))
refscreen={"solution_delta_nats_per_token":float(np.mean(ref_deltas)),
           "digit_delta_nats_per_token":float(np.mean(digit_deltas)),
           "improved_solutions":sum(v>0 for v in ref_deltas),
           "improved_digit_scores":sum(v>0 for v in digit_deltas)}
print("REFERENCE SCREEN",refscreen)
act22=np.load(W/"fwd_22"/"mlp_act_11.npy",mmap_mode="r")
a_summary=[]
for u in robust11:
    vals=np.concatenate([np.asarray(act22[i,m+1,u],np.float32)
                         for i,m in enumerate(ref_masks)])
    a_summary.append({"unit":u,"mean":round(float(vals.mean()),3),
                      "rms":round(float(np.sqrt((vals**2).mean())),3)})
print("SELECTED UNIT SOLUTION ACTIVATIONS",a_summary)
with (W/"notes.md").open("a") as f:
    f.write("\nPass23 full-reference continuation screen: "+json.dumps(refscreen)+".\n")
log_pass(24,"Causal generation test of eight length-matched failure-associated MLP units, complete output removal",manifest23)
archive_current("candidate_matched11_zero",manifest23)
print("Pass 24 ready.")
```