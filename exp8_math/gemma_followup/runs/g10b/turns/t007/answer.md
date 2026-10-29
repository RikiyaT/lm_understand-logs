The result file groups samples by problem, so the first inspection did not yet reach the answer texts. I’ll inspect the nested sample schema and activation index before calculating any statistics. This will ensure that each correctness label is matched to the right activation row. The important question remains whether unfinished mathematical reasoning has a consistent internal signature within the same problem.

```python
print("Sample container:", type(gen1[0]["samples"]).__name__)
sample0 = gen1[0]["samples"][0]
print("Sample fields:", list(sample0))
print("Sample metadata:", {k: v for k, v in sample0.items()
                            if not isinstance(v, (list, dict)) and
                            (not isinstance(v, str) or len(v) < 150)})
with open(wdir / "gen_1_acts" / "index.json") as f:
    act_index1 = json.load(f)
print("Activation index type:", type(act_index1).__name__)
if isinstance(act_index1, list):
    print("Index length and first entries:", len(act_index1), act_index1[:2])
else:
    print("Index keys:", list(act_index1)[:15])
    print("Index preview:", str(act_index1)[:700])
for rec in gen1[:8]:
    print("Problem", rec["id"], "reference:", rec["answer"])
    for s in rec["samples"][:2]:
        short = {k: v for k, v in s.items() if not isinstance(v, (list, dict))
                 and (not isinstance(v, str) or len(v) < 100)}
        print(short)
        for k, v in s.items():
            if isinstance(v, str) and len(v) >= 100:
                print(k, (v[:130] + " ... " + v[-150:]).encode("ascii", "backslashreplace").decode())
```
