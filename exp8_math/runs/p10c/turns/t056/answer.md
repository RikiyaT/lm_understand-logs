The protected direction remains strong after removing the average directions of ordinary reasoning, first self-checks, prompts, and answer transitions. Across the four repeated phrases, its late-layer projection is roughly six to eight times the variation seen during ordinary reasoning. This is still observational evidence: it could drive repetition or represent an attempted response to repetition. I will check its association with natural failures and its output-word associations before choosing the direction of intervention.

```python
# Hypothesis: the protected residual direction generalizes from matched probes to natural loops and favors repeated reconsideration rather than mathematical progress.
from sklearn.metrics import roc_auc_score
index1=json.load(open("/work/gen_1_acts/index.json"))
HG=np.load("/work/gen_1_acts/hidden_mean_answer.npy",mmap_mode="r")
base1=[json.loads(s) for s in open("/work/gen_1.jsonl")]
samples1=flat_records(base1)
bad1=np.array([not s["correct"] for s in samples1])
print("Natural-generation direction association:")
for l in [8,11,15,16,19,20,21,22,23]:
    u=rep_protected[l]
    hg=np.asarray(HG[:,index1["layers"].index(l)],dtype=np.float32)
    vals=hg@u
    print(l,"correct/failed means",np.round([vals[~bad1].mean(),vals[bad1].mean()],3).tolist(),
          "failure AUC",round(float(roc_auc_score(bad1,vals)),3))
if "E" not in globals(): E=load_original("model.language_model.embed_tokens.weight")
nfrozen=load_original("model.language_model.norm.weight")
safe_probe_words=["Wait"," Wait","Hmm","Therefore"," Therefore","So"," So","But","Let's","Again","Finally",
                  "The","Thus","Check","Calculate","Next","Yes","No","1","2","+","="]
u23=torch.from_numpy(rep_protected[23])
logit_axis=(1+nfrozen)*u23
print("Safe selected-token weight associations (not a forward pass):")
for word in safe_probe_words:
    ids=tok.encode(word).ids
    if len(ids)==1:
        print(repr(word),"id",ids[0],"association",round(float(E[ids[0]]@logit_axis),4))
print("Adjacent protected-direction cosines:",
      [(l,round(float(rep_protected[l]@rep_protected[l+1]),3)) for l in range(15,23)])
```
