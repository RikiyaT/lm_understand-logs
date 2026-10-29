The saved notes confirm that DEV and HOLDOUT were fixed before editing, and that several larger attention interventions were harmful. The combined candidate is now archived, so further experiments will not overwrite our fallback. I will inspect compact result summaries and file schemas next, avoiding the large text fields that truncated the previous output. This will establish whether the apparent gain is broad across DEV or concentrated in a few problems before we refine the mechanism.

```python
def read_jsonl(p):
    with open(p) as f:
        return [json.loads(s) for s in f if s.strip()]
r = read_jsonl(W/"gen_17_original.jsonl")[0]
print("GEN SCHEMA", {k: type(v).__name__ for k,v in r.items()})
print("NON-TEXT FIELDS", {k:v for k,v in r.items()
                         if not isinstance(v,(dict,list)) and len(str(v))<100})
for k in [2,3,11]:
    meta = json.loads((W/f"fwd_{k}"/"meta.json").read_text())
    print("META", k, {key: ({"type":type(v).__name__,"len":len(v)}
                            if isinstance(v,(list,dict)) else v)
                     for key,v in meta.items() if key!="texts"})
    print("FILES", k, [p.name for p in (W/f"fwd_{k}").iterdir()])
print("ACTIVATION FILES", [p.name for p in (W/"gen_1_acts").iterdir()][:12])
print("PROMPT FORMAT", (W/"eval_prompt_format.txt").read_text()[:1400])
print("ARCHIVE EXISTS", (W/"candidate_combo_17").exists())
print("KERNEL HELPERS", sorted(k for k,v in list(globals().items())
                               if callable(v) and not k.startswith("_"))[:70])
```