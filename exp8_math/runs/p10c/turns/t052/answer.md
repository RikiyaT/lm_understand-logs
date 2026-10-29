Halving the two strongest copying heads produced only a 1.6-point accuracy gain, smaller than its 3.1-point paired standard error. That does not establish a benefit, and competition-math accuracy decreased in this sample. I will measure repetition directly and then fully ablate the same two heads to test a stronger dose. A substantial reduction in loops accompanied by improved accuracy would support the copying-feedback hypothesis; unchanged repetition or worse accuracy would count against it.

```python
# Hypothesis: the two induction heads are redundant enough that partial attenuation is insufficient; full ablation should produce a stronger causal effect.
r6o=[json.loads(s) for s in open("/work/gen_6_original.jsonl")]
r6e=[json.loads(s) for s in open("/work/gen_6_edited.jsonl")]
s6o,s6e=flat_records(r6o),flat_records(r6e)
sum6=compare_samples(s6o,s6e)
print("Half-strength paired result:",sum6)
for tag,ss in [("original",s6o),("half-strength",s6e)]:
    reps=[repeated_fraction(tok.encode(s["text"]).ids[-2048:]) for s in ss]
    print(tag,"tail repetition mean:",round(float(np.mean(reps)),4),
          "tail repetition >0.8:",sum(x>0.8 for x in reps),
          "finished correct:",sum(s["correct"] and not s["cut"] for s in ss))
print("Full-strength edit:",apply_copy_strength(1.0))
with open("/work/notes.md","a") as f:
    f.write("\nPass 5 (gen_6), L11H1/L15H7 output attenuation 50%: "+str(sum6)+
            ". Gain not significant; not accepted. Next test 100% ablation on identical 128 problems. Five passes remain.\n")
```
