The final HOLDOUT pass was also essentially neutral, at −0.10 percentage points with a 0.86-point standard error. I will submit the same frozen one-neuron edit, which was selected on DEV before HOLDOUT was examined. I will now calculate the pooled uncertainty, verify that only the intended column differs from the original, and save the final report. The evidence so far does not justify claiming a successful mathematical repair.

```python
final_result=pooled_evaluation([23,24,25])
assert final_result["problems"]==256
assert final_result["answers_per_model"]==3072
assert edited()==[fname]
submitted=load(fname)
original=load_original(fname)
assert submitted.shape==original.shape
assert torch.isfinite(submitted).all()
assert torch.equal(submitted[:,:920],original[:,:920])
assert torch.equal(submitted[:,921:],original[:,921:])
assert torch.equal(submitted[:,920],original[:,920]*2)
assert hashlib.sha256(submitted.numpy().tobytes()).hexdigest()==frozen_hash

secondary={}
for model in ["original","edited"]:
    samples=[]
    for k in [23,24,25]:
        for line in (work/f"gen_{k}_{model}.jsonl").read_text().splitlines():
            samples.extend(json.loads(line)["samples"])
    secondary[model]={"cut_share":float(np.mean([s["cut"] for s in samples])),
                      "mean_tokens":float(np.mean([s["tokens"] for s in samples]))}
ci=[final_result["gain"]-1.96*final_result["problem_cluster_se"],
    final_result["gain"]+1.96*final_result["problem_cluster_se"]]
report={"submitted_edit":fcurrent,"tensor_sha256":frozen_hash,
        "changed_entries":int(torch.count_nonzero(submitted-original)),
        "dev":dev_pooled,"holdout":final_result,"holdout_95pct_ci":ci,
        "secondary_holdout":secondary,
        "conclusion":"No demonstrated accuracy improvement. Best tested non-original DEV candidate retained unchanged through HOLDOUT.",
        "selection_path":"Rejected broad failure-neuron, repetition-direction, and attention-head edits; selected numerically selective L17 MLP neuron920 using DEV associations and static digit alignment, tested removal and doubling, and replicated doubling before freezing."}
(work/"final_report.json").write_text(json.dumps(report,indent=2))
with (work/"notes.md").open("a") as f:
    f.write("\nPass25 HOLDOUT all256,n4: original7.42%,edited7.32%, gain-0.10pp, reported SE0.86pp. Frozen weights unchanged.\n")
    f.write("\n## FINAL SUBMISSION\nOnly model.layers.17.mlp.down_proj.weight[:,920] multiplied2. All other values original; embedding/output and final norm untouched. Selected on DEV, no HOLDOUT tuning. All25 forward passes used.\n")
    f.write(json.dumps(report)+"\n")
print("Final accuracy report:",json.dumps(final_result))
print("95% interval, percentage points:",[round(100*v,4) for v in ci])
print("Secondary outcomes:",secondary)
print("Final audit:",{"edited_tensors":edited(),"changed_entries":report["changed_entries"],
                      "matches_frozen":True,"submission_directory":"/work/edited"})
```
