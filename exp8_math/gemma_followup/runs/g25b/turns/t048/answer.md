The repetition-specific screen found substantially stronger associations than the original general-failure screen. Layer 0 neuron 1302 stands out: its correlation is consistent across both halves of DEV, and its output variation is much larger than that of the other leading candidates. I will inspect when it activates and whether that association is specific to repeated answers rather than all incorrect answers. If it primarily tracks useful computation or prompt content, that would weaken the case for ablation.

```python
assoc = pd.DataFrame(json.loads((work / "neuron_associations.json").read_text()))
merged = repscreen.merge(assoc[["layer","neuron","corr","raw","activity","output_norm"]],
                         on=["layer","neuron"], how="left")
print("Strong stable candidates, ordered by output variation:")
print(merged[merged.stable > 0.25].sort_values("output_effect_sd", ascending=False)
      .head(12).round(4).to_string(index=False))

xp = np.load(work / "gen_1_acts" / "mlp_last_prompt.npy", mmap_mode="r")
for layer, neuron in [(0,1302),(9,1853),(10,1628),(5,1353)]:
    vals = np.asarray(Xall[sel,layer,neuron])
    stats = {"layer":layer,"neuron":neuron,
             "prompt_mean":float(np.mean(xp[sel,layer,neuron])),
             "answer_mean":float(vals.mean()),"answer_sd":float(vals.std())}
    for label,m in [("correct_nonrepeat",(y==0)&(rep==0)),
                    ("wrong_nonrepeat",(y==1)&(rep==0)),
                    ("wrong_repeat",(y==1)&(rep==1)),
                    ("correct_repeat",(y==0)&(rep==1))]:
        stats[label] = {"n":int(m.sum()),"mean":float(vals[m].mean())}
    print(stats)

print("Activation index sample:", idx["samples"][int(sel[0])])
records = [json.loads(line) for line in (work / "gen_1.jsonl").read_text().splitlines()]
lookup_records = {r["id"]:r for r in records}
print("Sample index fields:", list(idx["samples"][0]))
# Retrieve two high-activation samples using the saved alignment.
top_local = np.argsort(np.asarray(Xall[sel,0,1302]))[-2:][::-1]
for j in top_local:
    meta = idx["samples"][int(sel[j])]
    ident = meta.get("id",meta.get("problem_id"))
    si = meta.get("sample",meta.get("sample_idx",meta.get("sample_index",0)))
    if ident in lookup_records and isinstance(si, int):
        ss = lookup_records[ident]["samples"][si]
        print({"id":ident,"sample":si,"correct":ss["correct"],"tokens":ss["tokens"],
               "tail":ss["text"][-145:].encode("ascii","backslashreplace").decode()})
```
