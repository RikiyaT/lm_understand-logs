The 25% head increase also failed, with a −2.28-point accuracy difference and a 1.70-point standard error. None of the tested edits has demonstrated an improvement. The least-negative DEV estimate belongs to doubling layer 17 neuron 920, so I will repeat that candidate with the problem order reversed to check sensitivity to sampling and batching. I will then freeze the best DEV choice before any HOLDOUT evaluation, and report negative or inconclusive results honestly.

```python
s21=pair_summary(21)[0]
with (work/"notes.md").open("a") as f:
    f.write("\nPass21 L11H0x1.25: DEV35.62%->33.33%, gain-2.28pp, reported SE1.70pp; cutoffs3.97%->3.67%. No head-reinforcement strength improved accuracy.\n"+json.dumps(s21)+"\n")
candidate_labels={5:"A remove",6:"A half",7:"A amplify",10:"H0 remove",11:"H0 x1.5",
                  12:"H2 x1.5",13:"H2 x0.5",14:"rank1 remove",15:"rank1 amplify",
                  16:"L0n1302 remove",17:"L0n1302 x1.5",18:"L17n920 x2",
                  19:"L17n920 remove",20:"H0 x1.10",21:"H0 x1.25"}
ranking=[]
for k,label in candidate_labels.items():
    sm=pair_summary(k)[0]
    ranking.append({"pass":k,"candidate":label,"gain_pp":100*sm["gain"]})
print("DEV candidate ranking:",pd.DataFrame(ranking).sort_values("gain_pp",ascending=False).round(3).to_string(index=False))

for tensor_name in list(edited()):
    drop_edit(tensor_name)
fw=load_original(fname).clone()
fw[:,920]*=2.0
save_edit(fname,fw)
fcurrent={fname:{"columns":[920],"column_multiplier":2.0,"removed_fraction":-1.0}}
(work/"current_edit.json").write_text(json.dumps(fcurrent))
shutil.copytree(work/"edited",work/"best_1",dirs_exist_ok=True)
dev_retest=list(reversed(dev))
(work/"DEV_RETEST.json").write_text(json.dumps(dev_retest))
with (work/"notes.md").open("a") as f:
    f.write("\n## Pass22 planned\nDEV-selected candidate: L17 MLP down_proj column920x2, all other weights original. Same63 DEV IDs in reverse order (/work/DEV_RETEST.json), both,n16,max_tokens8192. Independent batching-sensitive replication; pool passes18+22 for this candidate. HOLDOUT still unused. Candidate copied /work/best_1.\n"+json.dumps(fcurrent)+"\n")
print("Retest problem order:",dev_retest)
```
