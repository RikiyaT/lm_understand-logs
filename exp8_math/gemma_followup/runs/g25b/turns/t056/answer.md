Neuron 920 has positive correct-digit alignment on about 70% of DEV problems, whereas neuron 1062 does not. Together with its numerical selectivity and within-problem correctness association, this makes 920 the more defensible target. I will double only its outgoing contribution, leaving its activation gate and every other neuron unchanged. Improved accuracy would support an under-strength numerical feature; deterioration would show that this static alignment is not a useful guide to intervention.

```python
for tensor_name in list(edited()):
    drop_edit(tensor_name)
fname="model.layers.17.mlp.down_proj.weight"
fw=load_original(fname).clone()
fw[:,920]*=2.0
save_edit(fname,fw)
fcurrent={fname:{"columns":[920],"column_multiplier":2.0,"removed_fraction":-1.0}}
(work/"current_edit.json").write_text(json.dumps(fcurrent))
shutil.copytree(work/"edited",work/"candidate_F_neuron920_2x",dirs_exist_ok=True)
row920=assoc[(assoc.layer==17)&(assoc.neuron==920)].to_dict("records")[0]
with (work/"notes.md").open("a") as f:
    f.write("\nNew numerical-output hypothesis: L17 MLP neuron920 has stable within-problem correct-solution association, numerical/prose activity ratio1.548, answer activity0.3375 vs formatting0.1677. Static post-FFN-scaled output direction aligns with correct-digit contrast on69.84% of63 DEV problems. Neuron1062 rejected: alignment negative despite numerical selectivity. No network simulation or training.\n"+json.dumps(row920)+"\n")
    f.write("\n## Pass18 planned\nCandidateF: restore all other tensors; double only L17 MLP down_proj column920. DEV63,both,n16,max_tokens8192. Test mathematical accuracy; no unconditional output bias or embedding/norm edits.\n"+json.dumps(fcurrent)+"\n")
print("Candidate F:", fcurrent)
print("Neuron920 association:",row920)
print("Relative tensor change:",float((fw-load_original(fname)).norm()/load_original(fname).norm()))
```
