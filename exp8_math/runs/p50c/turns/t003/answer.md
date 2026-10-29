```python
# Hypothesis: the dominant failure is a long-reasoning attractor, not arithmetic errors: 85/96 outputs were cut off and every completed answer was correct.
g1 = [json.loads(s) for s in open("/work/gen_1.jsonl")]
print("Record keys:", list(g1[0]))
for r in g1[:5]:
    print({k:v for k,v in r.items() if k not in ["text","output","prompt","problem","response","completion"] and not isinstance(v, (list,dict)) and len(str(v)) < 150})
    for key in ["text", "output", "response", "completion"]:
        if key in r:
            print("START", str(r[key])[:280].replace("\n"," "))
            print("MIDDLE", str(r[key])[len(str(r[key]))//2:][:280].replace("\n"," "))
            print("END", str(r[key])[-280:].replace("\n"," "))
print("Activation index:", str(json.load(open("/work/gen_1_acts/index.json")))[:2000])
print("Model files:", os.listdir("/model"))
normrows=[]
for n in NAMES:
    if ".mlp.down_proj.weight" in n or ".out_proj.weight" in n or ".o_proj.weight" in n:
        w=load_original(n)
        cs=w.norm(dim=0)
        normrows.append((n, tuple(w.shape), round(float(w.norm()),3), round(float(cs.max()/cs.median()),3), int(cs.argmax())))
print("Output-projection norm statistics:")
for r in normrows: print(r)
Path("/work/notes.md").write_text("# Math weight editing\nPass 1: original on 32 GSM8K + 64 MATH diagnostic problems. Accuracy 11/96 (11.46%); 85/96 cut at 32768; no finished wrong answers. Dominant target: pathological long/repetitive reasoning. No edits yet.\n")
```
