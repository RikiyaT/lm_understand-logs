```python
# Hypothesis: the 89.8% cutoff rate indicates a shared pathological reasoning loop rather than ordinary mathematical errors; inspect loop onsets and component outliers.
import re
with open("/work/gen_1.jsonl") as f:
    gen1 = [json.loads(line) for line in f]
print("Generation fields:", sorted(gen1[0]))
print("First record metadata:", {k:v for k,v in gen1[0].items() if not isinstance(v, (list,dict)) and len(str(v)) < 200})
with open("/work/gen_1_acts/index.json") as f:
    act_index = json.load(f)
print("Activation index structure:", str(act_index)[:1600])

text_key = next(k for k in ("text", "output", "response", "generated_text") if k in gen1[0])
for i in [0,1,2,3,4,5,48,96,128,200]:
    r = gen1[i]
    s = r[text_key]
    print("\nSAMPLE", i, {k:v for k,v in r.items() if k != text_key and isinstance(v, (int,float,bool))})
    print("BEGIN:", repr(s[:260]))
    print("MIDDLE:", repr(s[len(s)//2:len(s)//2+260]))
    print("END:", repr(s[-260:]))
    sentences = [x.strip() for x in re.split(r"[\n.!?]+", s) if len(x.strip()) > 35]
    print("Repeated fragments:", [(x[:150], n) for x,n in collections.Counter(sentences).most_common(2)])

weights = {}
stats = []
for name in NAMES:
    if ".layers." not in name:
        continue
    w = load_original(name)
    weights[name] = w
    stats.append((name, tuple(w.shape), float(w.square().mean().sqrt()), float(w.abs().max())))
print("\nNorm and dynamics weights:")
for name,shape,rms,mx in stats:
    if len(shape) <= 1:
        w = weights[name]
        print(name.replace("model.language_model.layers.",""), shape,
              "rms",round(rms,4),"min",round(float(w.min()),4),
              "max",round(float(w.max()),4))
print("\nLargest matrix max/RMS ratios:")
for name,shape,rms,mx in sorted([x for x in stats if len(x[1]) > 1], key=lambda x:x[3]/max(x[2],1e-12), reverse=True)[:18]:
    print(name.replace("model.language_model.layers.",""),shape,"rms",round(rms,5),"max",round(mx,4),"ratio",round(mx/rms,1))
```
