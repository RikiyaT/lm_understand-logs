Amplifying neuron 1302 slightly reduced cutoffs but again lowered accuracy, by 1.79 points with a 1.66-point standard error. I therefore have no evidence that improving termination alone repairs the model’s mathematics. I will shift to components associated with correct solutions within the same problem, using the saved teacher-forced activations to distinguish answer-relevant features from generic fluency. The next candidate must have evidence tied to mathematical answer production, not just shorter or cleaner responses.

```python
s17 = pair_summary(17)[0]
with (work / "notes.md").open("a") as f:
    f.write("\nPass17 L0 neuron1302x1.5: DEV36.31%->34.52%, gain-1.79pp, reported SE1.66pp; cutoffs3.77%->3.08%. Better termination did not improve math. CandidateE not accepted.\n"+json.dumps(s17)+"\n")
print("Pass17 behavioral summary:", s17)

assoc["correct_stable"] = np.minimum(-assoc.half1, -assoc.half2)
print("Strong stable correct-solution associations:")
print(assoc.sort_values("correct_stable",ascending=False)
      .head(20).round(4).to_string(index=False))
print("Late-layer correct associations:")
print(assoc[assoc.layer >= 13].sort_values("correct_stable",ascending=False)
      .head(12).round(4).to_string(index=False))
for folder in ["fwd_3", "fwd_4"]:
    print(folder, [p.name for p in (work/folder).iterdir()])
    meta = json.loads((work/folder/"meta.json").read_text())
    print("Metadata keys:", list(meta))
    print("Metadata, excluding text-bearing fields:",
          {k:v for k,v in meta.items() if k not in ["texts","text","tokens","samples"] and len(str(v))<1500})
print("Compact-info example:", json.loads((work/"tf_compact_info.json").read_text())[0])
```
