```python
# Hypothesis: failing samples enter a stable repeated state; compare their text and weight statistics after unpacking the per-problem samples.
print("Sample fields:", sorted(gen1[0]["samples"][0]))
print("Sample metadata:", {k:v for k,v in gen1[0]["samples"][0].items() if not isinstance(v,(dict,list)) and len(str(v))<160})
samples1 = [dict(s, id=r["id"], source=r["source"], reference_answer=r["answer"]) for r in gen1 for s in r["samples"]]
text_key = next(k for k in ("text", "output", "response", "generated_text", "completion") if k in samples1[0])
for i in [0,1,2,3,4,5,10,48,96,128,200]:
    r = samples1[i]
    s = r[text_key]
    print("\nSAMPLE", i, {k:v for k,v in r.items() if k!=text_key and isinstance(v,(int,float,bool))})
    print("BEGIN:", repr(s[:240]))
    print("MIDDLE:", repr(s[len(s)//2:len(s)//2+240]))
    print("END:", repr(s[-240:]))
    sentences = [x.strip() for x in re.split(r"[\n.!?]+", s) if len(x.strip())>35]
    print("Most repeated:", [(x[:100],n) for x,n in collections.Counter(sentences).most_common(2)])

torch.set_num_threads(8)
weights = {}
stats = []
for name in NAMES:
    if ".layers." not in name:
        continue
    w = load_original(name)
    weights[name] = w
    stats.append((name, tuple(w.shape), float(w.square().mean().sqrt()), float(w.abs().max())))
print("\nNorm and dynamics summaries by layer:")
for l in range(24):
    prefix = f"model.language_model.layers.{l}."
    entries = []
    for name,shape,rms,mx in stats:
        if name.startswith(prefix) and len(shape)<=1:
            w=weights[name]
            entries.append((name[len(prefix):],round(rms,3),round(float(w.min()),3),round(float(w.max()),3)))
    print(l, entries)
print("\nLargest matrix max/RMS ratios:")
for name,shape,rms,mx in sorted([x for x in stats if len(x[1])>1],key=lambda x:x[3]/max(x[2],1e-12),reverse=True)[:12]:
    print(name.replace("model.language_model.layers.",""),shape,"rms",round(rms,5),"max",round(mx,4),"ratio",round(mx/rms,1))
```
