On this problem, the reference uses \((a+b)^2=90+2\cdot27=144\), while inspected failures repeatedly return to 90. I’ll use two controlled continuations of the same prompt: one completes that identity, and the other repeats the intermediate equation before giving 90. These are diagnostic probes, not evidence of their own frequency. Attention concentrated on the repeated intermediate expression would motivate a head-level test, but only the intervention could show whether that routing is harmful.

```python
probe_problem=problems_by_id["math_train_algebra_380"]["problem"]
good_probe=("Let the integers be a and b. We know a^2+b^2=90 and ab=27. "
            "We need a+b, not a^2+b^2. The identity is (a+b)^2=a^2+b^2+2ab. "
            "Substituting both given values gives (a+b)^2=90+2*27=144. "
            "Since a and b are positive, a+b=sqrt(144)=12. "
            "Checking the distinction, 90 is the sum of squares, while 12 is the sum. "
            "Final Answer: \\boxed{12}")
bad_probe=("Let the integers be a and b. We know a^2+b^2=90 and ab=27. "
           "We need a+b. We have a^2+b^2=90. Therefore a^2+b^2=90. "
           "The sum of the squares is 90. We know a^2+b^2=90. "
           "Thus the sum is 90. The sum of the two integers is 90. "
           "We have a^2+b^2=90. The sum of the two integers is 90. "
           "Final Answer: \\boxed{90}")
attn_texts=[PREFIX+probe_problem+SUFFIX+x for x in [good_probe,bad_probe]]
json.dump(attn_texts,open("/work/attention_probe_texts.json","w"))
with open("/work/notes.md","a") as f:
    f.write("\n## Pass8 planned\nOriginal model (no edits applied for this pass; disk still contains candidateA 1.5x in three down_proj tensors). "
            "Two controlled algebra380 probes in /work/attention_probe_texts.json: correct square-identity computation versus repeated intermediate-value conclusion. Full-attention layers5,11,17; inspect head routing, no causal claim from attention alone. Attention tool limits probes to2 texts.\n")
print(json.dumps(attn_texts))
```
