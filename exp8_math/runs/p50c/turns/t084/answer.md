Strength 2 was independently confirmed: accuracy rose from 21/256 to 110/256, a paired gain of 34.77 percentage points with a 3.53-point standard error. That makes it a solid fallback, not just a promising diagnostic result. I’ll finish recovering the probe’s two early-closing controls and the strength-3 result before testing strength 4. The key prediction is that repetition becomes less likely while useful reasoning and appropriately delayed closure remain intact.

```python
# Hypothesis: the repetition intervention has a selective dose-response rather than merely favoring early closure.
print("LATER RESULTS")
for line in notes_text[9800:].splitlines():
    if not line.startswith("{") and any(s in line.lower() for s in
        ["pass20", "gen29", "alpha3", "alpha4", "failed", "best", "gain"]):
        print(line[:1100])
print("\nCLOSURE CONTROLS")
for i in (12, 13):
    print(i, json.dumps(r["texts"][i], ensure_ascii=True))
print("\nHELPERS")
print([k for k, v in list(globals().items())
       if callable(v) and any(s in k.lower() for s in ["metric", "transfer", "probe", "alpha", "pair"])])
print("\nMETA")
print(Path("/work/fwd_26/meta.json").read_text()[:1700])
print("\nSPLIT SIZES")
spl = json.loads(Path("/work/splits.json").read_text())
print({k: len(v) if isinstance(v, (list, dict)) else v for k, v in spl.items()})
```
