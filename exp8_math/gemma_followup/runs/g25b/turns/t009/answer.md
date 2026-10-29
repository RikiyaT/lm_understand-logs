The strongest length-controlled failure correlations are modest, around 0.10–0.14, so they are not enough to identify a repair by themselves. I’ll check which neurons also track repetitive reasoning and prepare a teacher-forced corpus containing all DEV reference solutions plus the model’s own correct and incorrect answers. That corpus will let me test whether an edit selectively weakens failed continuations while preserving correct ones. A broad likelihood drop on every answer type would argue against the edit.

```python
repeat_stats=[]
for l in range(18):
    raw=np.asarray(Xall[sel,l,:],dtype=np.float64)
    X=demean(raw)
    Xc=X-Q@(Q.T@X)
    Rc=R-Q@(Q.T@R)
    rc=(Rc@Xc)/(np.linalg.norm(Rc)*np.sqrt((Xc*Xc).sum(0)+1e-20))
    for j in range(2048):
        s=dict(stats[l*2048+j])
        s["repeat_controlled"]=float(rc[j])
        repeat_stats.append(s)
repeat_stats.sort(key=lambda s:s["repeat_controlled"],reverse=True)
print("Top repetition-associated neurons:")
for s in repeat_stats[:16]:
    print({k:round(s[k],4) if isinstance(s[k],float) else s[k]
           for k in ["layer","neuron","corr","raw","repeat","repeat_controlled","activity"]})

PREFIX="<bos><start_of_turn>user\n"
SUFFIX="\nPlease reason step by step, and put your final answer within \\boxed{}.<end_of_turn>\n<start_of_turn>model\n"
tf_texts=[]; tf_info=[]
for p in dev_ids:
    r=problems_by_id[p]
    prompt=PREFIX+r["problem"]+SUFFIX
    candidates=[("reference",r["solution"]+"\nFinal Answer: \\boxed{"+r["answer"]+"}")]
    ss=next(b["samples"] for b in base if b["id"]==p)
    for ok,label in [(True,"correct"),(False,"wrong")]:
        chosen=[s for s in ss if s["correct"]==ok]
        random.Random(320+dev_ids.index(p)+(0 if ok else 100)).shuffle(chosen)
        candidates.extend((label,s["text"]) for s in chosen[:4])
    for label,ans in candidates:
        raw=prompt+ans
        enc=tok.encode(raw)
        if len(enc.ids)>1200:
            raw=tok.decode(enc.ids[:1200],skip_special_tokens=False)
        tf_texts.append(raw)
        tf_info.append(dict(id=p,kind=label,prompt_tokens=len(tok.encode(prompt).ids),
                            tokens=len(tok.encode(raw).ids)))
json.dump(tf_texts,open("/work/tf_dev_texts.json","w"))
json.dump(tf_info,open("/work/tf_dev_info.json","w"))
request={"type":"teacher_forced","model":"original","texts":tf_texts,
         "return":{"token_logprobs":True,"topk":0,"hidden_states":[5,14,16,18],"mlp_act":[4,13,15,17]}}
json.dump(request,open("/work/tf_baseline_request.json","w"))
print("Teacher-forcing corpus:",len(tf_texts),"texts;",sum(x["tokens"] for x in tf_info),"tokens;",
      dict(Counter(x["kind"] for x in tf_info)))
print("Request saved:", "/work/tf_baseline_request.json")
with open("/work/notes.md","a") as f:
    f.write("\nAssociations: within-DEV-problem, length/cut-controlled MLP failure correlations are modest (~0.10-0.14); no causal claim yet. Grade alignment exact; activation token counts include prompt tokens, capped at 8192.\n"
            "## Pass 2 planned\nOriginal, no edits. Teacher-force fixed DEV corpus /work/tf_dev_texts.json (all references and up to 4 correct/4 failing baseline samples per problem, truncated at 1200 total tokens). Score continuation likelihoods and inspect MLP layers 4,13,15,17; no HOLDOUT use.\n")
```
