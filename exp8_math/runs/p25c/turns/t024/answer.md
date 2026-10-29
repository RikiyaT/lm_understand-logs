```python
# Hypothesis: an exactly unchanged 128-sample result indicates either an ineffective edit path or a non-causal neuron; verify the saved edit and token-level effects before interpreting it.
print("Work files:",sorted(os.listdir("/work")))
print("Current edits:",edited())
name="model.language_model.layers.23.mlp.down_proj.weight"
current=load(name)
original=load_original(name)
print("Column22 norms original/current:",float(original[:,22].norm()),float(current[:,22].norm()))
print("Weight difference:",float((current-original).norm()),"changed entries",int((current!=original).sum()))
def read_samples(path):
    with open(path) as f:
        rows=[json.loads(x) for x in f]
    return [dict(s,id=r["id"],source=r.get("source")) for r in rows for s in r["samples"]]
g7o=read_samples("/work/gen_7_original.jsonl")
g7e=read_samples("/work/gen_7_edited.jsonl")
identical=sum(a["text"]==b["text"] for a,b in zip(g7o,g7e))
print("Exactly identical texts:",identical,"of",len(g7o))
if os.path.exists("/work/notes.md"):
    with open("/work/notes.md") as f:
        print("Notes excerpt:",f.read()[-2000:])
print("Earlier generation summaries:")
for file in sorted(os.listdir("/work")):
    if re.fullmatch(r"gen_\d+(?:_edited|_original)?\.jsonl",file):
        s=read_samples("/work/"+file)
        print(file,"n",len(s),"correct",sum(x["correct"] for x in s),"cut",sum(x["cut"] for x in s),
              "mean_tokens",round(np.mean([x["tokens"] for x in s])))
```
