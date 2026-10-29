Amplification also failed to improve DEV accuracy: the measured change was −1.59 percentage points, with a 1.63-point standard error. Taken together, the three strengths do not support this failure-associated neuron set as a repair target. I’ll retain the results as negative evidence and inspect the opposite association: components consistently more active in correct answers after controlling for problem and length. Any new candidate will need its own causal test rather than inheriting credibility from the correlations.

```python
res7,byp7=pair_summary(7)
shutil.copytree("/work/edited","/work/candidate_A_amplified",dirs_exist_ok=True)
negative=[s for s in stats if s["half1"]<-.05 and s["half2"]<-.05]
negative.sort(key=lambda s:max(s["half1"],s["half2"]))
print("Stable success-associated neurons:")
for s in negative[:25]:
    print({k:round(v,4) if isinstance(v,float) else v for k,v in s.items()})
print("Negative minima per layer:",
      [(l,round(min(s["corr"] for s in stats if s["layer"]==l),3)) for l in range(18)])
with open("/work/notes.md","a") as f:
    f.write("\nPass7 amplification: reported gain -1.59pp, SE1.63pp. "+json.dumps(res7)+"\n"
            "Candidate A rejected as a useful repair: full attenuation harmed, half attenuation and amplification did not improve accuracy. Need a more specific component hypothesis.\n")
```
