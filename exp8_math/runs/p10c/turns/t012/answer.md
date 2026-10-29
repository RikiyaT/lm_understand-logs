```python
# Hypothesis: the highest-confidence repetition neurons should push loop-related language rather than merely signal that reasoning is complete.
import os,json,math,collections
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(8)
gen1=[json.loads(s) for s in open("/work/gen_1.jsonl")]
flat1=[dict(id=r["id"],source=r["source"],answer=r["answer"],**s) for r in gen1 for s in r["samples"]]
idx1=json.load(open("/work/gen_1_acts/index.json"))
M1=np.load("/work/gen_1_acts/mlp_mean_answer.npy").astype(np.float32)
y1=np.array([s["correct"] for s in idx1["samples"]],bool)
mean_bad=M1[~y1].mean(0); mean_good=M1[y1].mean(0)
def ids_of(s):
    r=tok.encode(s)
    return list(r.ids) if hasattr(r,"ids") else list(r)
candidates={18:[8],19:[11],22:[59,11,89],23:[22,40,46,94,19,350]}
E=load_original("model.language_model.embed_tokens.weight")
nf=load_original("model.language_model.norm.weight")
print("Final norm (read only) range/median:",float(nf.min()),float(nf.median()),float(nf.max()))
safe_words=[" Wait","Wait"," wait"," But","But"," However"," Therefore"," Thus"," So",
            " again"," check"," repeat"," repeating"," loop"," answer"," response"," write",
            " calculate"," equals"," because"," verify"," correct"," mistake"," Let's",
            " I","1","2","0","+","=","\\","<think>","</think>","<|im_end|>"]
safe_ids=sorted(set(t for s in safe_words for t in ids_of(s)))
safe_labels={t:s for s in safe_words for t in ids_of(s) if len(ids_of(s))==1}
# This is a static weight-to-vocabulary attribution, not a simulated forward pass.
down_cache={}
for l,js in candidates.items():
    W=load_original(f"model.language_model.layers.{l}.mlp.down_proj.weight")
    down_cache[l]=W
    for j in js:
        direction=W[:,j]*float(np.sign(mean_bad[l-1,j]-mean_good[l-1,j]))
        scores=(E[safe_ids]@direction).numpy()
        order=np.argsort(scores)
        print((l,j),"loop/good mean",np.round([mean_bad[l-1,j],mean_good[l-1,j]],3).tolist(),
              "positive",[(safe_labels.get(safe_ids[k],safe_ids[k]),round(float(scores[k]),4)) for k in order[-5:][::-1]],
              "negative",[(safe_labels.get(safe_ids[k],safe_ids[k]),round(float(scores[k]),4)) for k in order[:4]])
notes="""# Math weight-edit experiment
Baseline pass 1: 256 original generations (96 GSM8K,160 MATH), full 32768-token limit.
Accuracy 24/256=9.375%; 232 cutoffs (91.41%). Correct GSM=12/96, MATH=12/160.
Failures enter exact repetitive thinking; 213/232 failed tails contain 'Wait,'.
Repeated 12-token fractions first2048/middle/tail: failures .070/.506/.975, correct .039/.098/.177.
Pass 2 (files fwd_3 after one uncounted attention-instrumentation error): two teacher-forced arithmetic/trigonometric probes, matched identical sentences at first vs repeated occurrences.
Most consistent within-content repetition-associated MLP neurons:
23:22 (+3.26),40 (-4.19),46 (+1.80),94 (+2.18),19 (+1.97);
19:11 (-1.57);18:8 (-1.51);22:59 (+1.13).
These signs agree across five distinct repetitive phrases and with failed-vs-correct generation averages.
Caveat: teacher-forced chat template inserted empty thinking before supplied think; use raw templated text for follow-up.
No edits tested yet. Eight forward passes remain. Need causal dose tests and independent paired validation.
"""
Path("/work/notes.md").write_text(notes)
print("Notes saved. Existing edits:",edited())
```
