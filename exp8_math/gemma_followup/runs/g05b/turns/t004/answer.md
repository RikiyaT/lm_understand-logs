The baseline accuracy is 16.46%, and many failures are not simply wrong final numbers: 708 responses finished without an extracted answer, while 150 failed at the token limit. That makes answer completion and reasoning breakdown worth investigating alongside arithmetic errors. I’ll inspect representative failures and count mixed-success problems before fixing DEV. I’ll also check the activation indexing so each comparison uses the correct answer’s measurements. If missing-answer responses share a specific behavior rather than merely being harder problems, that would give us a more focused repair hypothesis.

```python
with open("/work/gen_1.jsonl") as f:
    baseline = [json.loads(line) for line in f]

print("Record schema:")
for k, v in baseline[0].items():
    if isinstance(v, str):
        print(k, "str", repr(v[:100]))
    elif isinstance(v, (int, float, bool)) or v is None:
        print(k, type(v).__name__, v)
    elif isinstance(v, dict):
        print(k, "dict keys", sorted(v.keys()))
    else:
        print(k, type(v).__name__)

print("Activation directory files:", sorted(os.listdir("/work/gen_1_acts")))
for filename in sorted(glob.glob("/work/gen_1_acts/*.json")):
    with open(filename) as f:
        metadata = json.load(f)
    print(os.path.basename(filename), str(metadata)[:1000])

text_key = next(
    (k for k in ["text", "response", "answer", "completion", "generated_text"]
     if isinstance(baseline[0].get(k), str)),
    None
)
print("Detected response field:", text_key)
if text_key:
    for i in [0, 1, 16, 32, 128, 256, 512, 1024, 1536, 1792]:
        r = baseline[i]
        s = r[text_key]
        excerpt = s[:170] + (" ... " + s[-100:] if len(s) > 170 else "")
        print("Example", i, ascii(excerpt))
```
