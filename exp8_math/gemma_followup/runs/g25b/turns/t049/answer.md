Neuron 1302 is more active in repeated answers, whether correct or incorrect, but does not distinguish correct from incorrect non-repeating answers. That supports a repetition-related role rather than a general “wrong-answer neuron,” although its strong prompt activity is a warning that it may also be useful. I will isolate it by zeroing just its outgoing column and restoring all attention weights. A reduction in repetition accompanied by better accuracy would support the repair; impaired instruction following or lower accuracy would refute it.

```python
for j in top_local:
    meta = idx["samples"][int(sel[j])]
    ss = lookup_records[meta["item"]]["samples"][meta["sample"]]
    print({"id":meta["item"],"correct":ss["correct"],"tokens":ss["tokens"],
           "tail":ss["text"][-140:].encode("ascii","backslashreplace").decode()})

for tensor_name in list(edited()):
    drop_edit(tensor_name)
ename = "model.layers.0.mlp.down_proj.weight"
ew = load_original(ename).clone()
ew[:,1302] = 0
save_edit(ename, ew)
ecurrent = {ename: {"columns":[1302], "removed_fraction":1.0}}
(work / "current_edit.json").write_text(json.dumps(ecurrent))
shutil.copytree(work / "edited", work / "candidate_E_neuron1302_removed", dirs_exist_ok=True)
with (work / "notes.md").open("a") as f:
    f.write("\nRepetition-specific MLP screen: within-problem associations controlling loglength,cut,correctness. L0 neuron1302 correlation0.334; split halves0.318/0.352. Nonrepeat means correct3.50/wrong3.43, repeat correct4.28/wrong4.51; prompt mean28.28 suggests possible useful syntax role. Observational only.\n")
    f.write("\n## Pass16 planned\nCandidateE: restore all other weights; zero only L0 MLP down_proj column1302. DEV63,both,n16,max_tokens8192. Test accuracy and repetition, not brevity alone.\n"+json.dumps(ecurrent)+"\n")
print("Candidate E saved. Relative tensor change:",
      float((ew-load_original(ename)).norm()/load_original(ename).norm()))
```
