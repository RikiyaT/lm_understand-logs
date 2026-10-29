The history weakens the claim that the original layer-8 neurons are uniquely responsible: the matched control ablation also gained about two points. Head 2 has clearer causal evidence of importance, since halving it caused a large accuracy loss, but that does not establish why modest amplification helps. I will examine within-problem, length-matched activation differences and repeat the combined candidate unchanged. Replication would support the candidate’s accuracy effect; stronger associations after length matching would justify more selective component tests.

```python
import re
DEV = json.loads((W/"DEV_ids.json").read_text())
units8 = json.loads((W/"layer8_units.json").read_text())
def pair_details(k):
    oo = {x["id"]:x for x in read_jsonl(W/f"gen_{k}_original.jsonl")}
    ee = {x["id"]:x for x in read_jsonl(W/f"gen_{k}_edited.jsonl")}
    rows=[]
    for pid in DEV:
        a=np.array([s["correct"] for s in oo[pid]["samples"]],float)
        b=np.array([s["correct"] for s in ee[pid]["samples"]],float)
        rows.append((pid, float(a.mean()),float(b.mean()),float((b-a).mean())))
    return rows
pd17=pair_details(17)
print("DEV IMPROVED/TIED/WORSE", [sum(np.sign(x[3])==s for x in pd17) for s in [1,0,-1]])
print("LARGEST CHANGES",sorted(pd17,key=lambda x:abs(x[3]),reverse=True)[:8])

diag = {x["id"]:x for x in read_jsonl(W/"gen_1.jsonl")}
samples=idx["samples"]
by_problem=collections.defaultdict(lambda:{True:[],False:[]})
for i,s in enumerate(samples):
    if s["item"] in set(DEV):
        by_problem[s["item"]][bool(s["correct"])].append(i)
matched=[]
for pid,g in by_problem.items():
    options=sorted((abs(np.log((samples[i]["tokens"]+1)/(samples[j]["tokens"]+1))),i,j)
                   for i in g[False] for j in g[True])
    used=set()
    for gap,i,j in options:
        if gap<=0.35 and i not in used and j not in used:
            matched.append((i,j)); used.update([i,j])
print("LENGTH-MATCHED WRONG/CORRECT PAIRS",len(matched))
ma=np.load(W/"gen_1_acts"/"mlp_mean_answer.npy",mmap_mode="r")
si=np.array([i for i,s in enumerate(samples) if s["item"] in set(DEV)])
mi=np.array([a for a,b in matched]); mj=np.array([b for a,b in matched])
layer_scores={}
for li,layer in enumerate(idx["mlp_layers"]):
    arr=np.asarray(ma[si,li,:],np.float32)
    sd=arr.std(axis=0)+1e-6
    contrast=(np.asarray(ma[mi,li,:],np.float32)-np.asarray(ma[mj,li,:],np.float32)).mean(axis=0)
    score=contrast/sd
    layer_scores[layer]=score
print("LAYER8 ORIGINAL GROUP MATCHED EFFECTS",
      [(u,round(float(layer_scores[8][u]),3)) for u in units8])
print("TOP MATCHED FAILURE ASSOCIATIONS",
      sorted([(float(v[u]),l,int(u)) for l,v in layer_scores.items()
               for u in np.argsort(v)[-4:]],reverse=True)[:12])
for pid in ["math_train_algebra_546","math_train_algebra_211"]:
    rr=next(x for x in read_jsonl(W/"gen_17_edited.jsonl") if x["id"]==pid)
    bad=next((s for s in rr["samples"] if not s["correct"] and not s["cut"]),None)
    if bad:
        print("FAILURE",pid,"pred",bad["pred"],"excerpt",bad["text"][-280:].encode("ascii","backslashreplace").decode())
manifest17={
 "model.layers.11.self_attn.o_proj.weight":{"columns":[512,768],"multiplier":1.2},
 "model.layers.8.mlp.down_proj.weight":{"columns":units8,"multiplier":0.5}}
with (W/"notes.md").open("a") as f:
    f.write("\nInterpretation update: matched layer-8 control gained +2.15 pp; "
            "specificity of original neuron association is unproven. "
            "Upcoming pass 18: unchanged combination replication, BOTH locked DEV, "
            "n=32, max_tokens=8192. Exact edits: "+json.dumps(manifest17)+
            ". All other tensors original. HOLDOUT unused.\n")
print("Pass 18 ready: unchanged combined candidate.")
```