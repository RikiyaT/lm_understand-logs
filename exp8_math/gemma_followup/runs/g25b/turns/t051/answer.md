Ablating neuron 1302 increased cutoffs from 4.46% to 14.58% and missing finished answers from 1 to 24, while lowering accuracy by 2.18 points. This is evidence against it causing repetition; it appears to help keep generation well formed. I will test a 50% increase in its outgoing contribution instead. That intervention must improve accuracy—not just termination—to count as a useful repair.

```python
s16 = pair_summary(16)
print("Behavior summary:", s16)
with (work / "notes.md").open("a") as f:
    f.write("\nPass16 L0 neuron1302 removed: DEV36.01%->33.83%, gain-2.18pp, reported SE1.84pp; cutoffs4.46%->14.58%, missing finished answers1->24. Ablation contradicts harmful-neuron hypothesis; suggests stabilizing role.\n")
    f.write(json.dumps(s16)+"\n")

ew = load_original(ename).clone()
ew[:,1302] *= 1.5
save_edit(ename, ew)
ecurrent = {ename: {"columns":[1302], "removed_fraction":-0.5, "column_multiplier":1.5}}
(work / "current_edit.json").write_text(json.dumps(ecurrent))
shutil.copytree(work / "edited", work / "candidate_E_neuron1302_1p5", dirs_exist_ok=True)
with (work / "notes.md").open("a") as f:
    f.write("\n## Pass17 planned\nOnly L0 MLP down_proj column1302 multiplied1.5; all other weights original. DEV63,both,n16,max_tokens8192. Reverse causal prediction: better termination and accuracy, accuracy required.\n"+json.dumps(ecurrent)+"\n")
print("Pass17 edit:", ecurrent)
```
