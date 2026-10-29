Removing the head slightly lowered the repeated text’s average log probability while leaving the productive text’s average almost unchanged. This is in the predicted direction, but the effect is small and does not establish an accuracy benefit. I will inspect the reasoning-only regions and then run a paired generation comparison on the recorded development problems. The edit will be useful only if it converts failures into correct answers, not merely shorter outputs.
```python
# Hypothesis: layer 15 head 7 ablation selectively weakens looping enough to improve completed, correct solutions.
lp0=np.load(W/"fwd_10"/"token_logprobs.npy")
lp1=np.load(W/"fwd_11"/"token_logprobs.npy")
regions=[("shared_arithmetic",0,len(token_ids(prefix+problem+suffix)),common10),
         ("productive_checks",0,common10,good_end),
         ("repeated_checks",1,common10,lens10[1]),
         ("established_loop",1,start_eval,lens10[1])]
probe_effects={}
for name,b,a,z in regions:
    x=lp0[b,a-1:z-1]; y=lp1[b,a-1:z-1]
    probe_effects[name]={"original":float(x.mean()),"edited":float(y.mean()),
                        "delta":float((y-x).mean()),"tokens":len(x)}
print("Reasoning-region log probabilities:",probe_effects)
z0=np.load(W/"fwd_10"/"mlp_act_23.npy",mmap_mode="r")
z1=np.load(W/"fwd_11"/"mlp_act_23.npy",mmap_mode="r")
print("Loop-associated downstream neurons (original,edited mean in loop):",
      {j:(round(float(z0[1,start_eval:end_eval,j].mean()),3),
          round(float(z1[1,start_eval:end_eval,j].mean()),3)) for j in (22,94,317,495,162)})
req9={"type":"generate","model":"both","problems":dev_ids,"n":2,"max_tokens":32768}
(W/"pass09_request.json").write_text(json.dumps(req9))
with (W/"notes.md").open("a") as f:
    f.write("Pass 8/25 fwd_11 L15H7=0: mean full-text logprob productive -1.50457 vs -1.50525 "
            "original, repeated -0.44999 vs -0.42348 original. Reasoning-only regions: "
            +json.dumps(probe_effects)+". Small selective effect; not accuracy evidence.\n")
    f.write("NEXT PASS 9/25: paired generation original vs edited; exact request /work/pass09_request.json. "
            "64 development IDs /work/dev_ids.json, n=2, max_tokens=32768. Exact edit unchanged: "
            "L15 self_attn.o_proj.weight columns [1792:2048] *=0, all else original.\n")
print("Recorded paired development pass; active edits:",edited())
```
