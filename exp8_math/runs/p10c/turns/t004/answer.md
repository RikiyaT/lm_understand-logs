```python
# Hypothesis: the dominant failure is a recurrent thinking loop, not missing mathematical knowledge (232/256 answers hit the token limit).
gen1 = [json.loads(s) for s in open("/work/gen_1.jsonl")]
print("Generation fields:", list(gen1[0]))
idx1 = json.load(open("/work/gen_1_acts/index.json"))
print("Activation index:", str(idx1)[:2500])
for j in [0,1,2,3,96,97,98,99]:
    r = gen1[j]
    print("\nSAMPLE", j, {k:v for k,v in r.items() if k not in ["text","output","response","completion","prompt"] and len(str(v)) < 150})
    for k,v in r.items():
        if isinstance(v,str) and len(v)>1000:
            print(k, "BEGIN", repr(v[:280]), "MIDDLE", repr(v[len(v)//2:len(v)//2+260]), "END", repr(v[-280:]))
print("Correct sample indices:", [i for i,r in enumerate(gen1) if r.get("correct",r.get("is_correct",False))])
```
