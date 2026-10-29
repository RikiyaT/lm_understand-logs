The repeat was essentially neutral: +0.10 percentage points with a 1.41-point standard error. Pooling it with the first run gives a slightly negative estimate, but this remains the best tested non-original candidate. I will now freeze the single-column, two-times edit and use all three remaining passes solely for HOLDOUT confirmation. The fixed schedule gives every HOLDOUT problem 12 samples per model, and no HOLDOUT result will change the submitted weights.

```python
import hashlib
from collections import defaultdict

def pooled_evaluation(pass_numbers):
    by_problem=defaultdict(lambda: {"original":[], "edited":[]})
    for k in pass_numbers:
        for model in ["original","edited"]:
            for line in (work/f"gen_{k}_{model}.jsonl").read_text().splitlines():
                rr=json.loads(line)
                by_problem[rr["id"]][model].extend(int(s["correct"]) for s in rr["samples"])
    problem_differences=[]
    originals=[]; edits=[]
    for ident, rr in by_problem.items():
        assert len(rr["original"])==len(rr["edited"])
        originals.extend(rr["original"]); edits.extend(rr["edited"])
        problem_differences.append(np.mean(rr["edited"])-np.mean(rr["original"]))
    differences=np.asarray(edits,dtype=float)-np.asarray(originals,dtype=float)
    return {"passes":pass_numbers,"problems":len(by_problem),"answers_per_model":len(edits),
            "original_accuracy":float(np.mean(originals)),"edited_accuracy":float(np.mean(edits)),
            "gain":float(np.mean(differences)),
            "problem_cluster_se":float(np.std(problem_differences,ddof=1)/np.sqrt(len(by_problem))),
            "sample_paired_se":float(np.std(differences,ddof=1)/np.sqrt(len(differences))),
            "samples_wrong_to_right":int(np.sum(differences==1)),
            "samples_right_to_wrong":int(np.sum(differences==-1))}

dev_pooled=pooled_evaluation([18,22])
print("Pooled DEV:",dev_pooled)
assert edited()==[fname]
final_tensor=load(fname)
orig_tensor=load_original(fname)
assert torch.equal(final_tensor[:,:920],orig_tensor[:,:920])
assert torch.equal(final_tensor[:,921:],orig_tensor[:,921:])
assert torch.equal(final_tensor[:,920],2*orig_tensor[:,920])
frozen_hash=hashlib.sha256(final_tensor.numpy().tobytes()).hexdigest()
frozen={"candidate":"F, L17 neuron920 x2","edit":fcurrent,"tensor_sha256":frozen_hash,
        "selection":"Best pooled DEV estimate among tested non-original candidates",
        "dev":dev_pooled,"no_further_tuning":True,
        "holdout_plan":{"23":{"subset":"HOLDOUT_A.json","n":8},
                        "24":{"subset":"HOLDOUT_B.json","n":8},
                        "25":{"subset":"HOLDOUT.json","n":4}}}
(work/"FROZEN.json").write_text(json.dumps(frozen,indent=2))
shutil.copytree(work/"edited",work/"frozen_submission",dirs_exist_ok=True)

holdout=json.loads((work/"HOLDOUT.json").read_text())
assert len(holdout)==256 and not set(holdout)&set(dev)
holdout_a,holdout_b=holdout[:128],holdout[128:]
(work/"HOLDOUT_A.json").write_text(json.dumps(holdout_a))
(work/"HOLDOUT_B.json").write_text(json.dumps(holdout_b))
with (work/"notes.md").open("a") as f:
    f.write("\nPass22 L17n920x2 replication: DEV35.52%->35.62%, gain+0.10pp, reported SE1.41pp.\n")
    f.write("Pooled DEV18+22: "+json.dumps(dev_pooled)+"\n")
    f.write("\nCANDIDATE FROZEN BEFORE HOLDOUT. Exact edit L17 MLP down_proj column920x2; all other weights original. No proven DEV gain; best tested non-original candidate. SHA256 "+frozen_hash+". Copy /work/frozen_submission. No further weight changes or HOLDOUT tuning.\n")
    f.write("Confirmation schedule fixed now: pass23 first128 HOLDOUT IDs n8; pass24 remaining128 n8; pass25 all256 n4. Both models,max_tokens8192 each; total3072 answers/model,12/problem. Pool all confirmation results with problem-cluster standard error.\n")
    f.write("\n## Pass23 planned\nFrozen edit "+json.dumps(fcurrent)+"; /work/HOLDOUT_A.json (first128 original HOLDOUT IDs),both,n8,max_tokens8192.\n")
print("Frozen tensor hash:",frozen_hash)
print("HOLDOUT_A IDs:",json.dumps(holdout_a))
```
