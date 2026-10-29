The targeted generation request again failed before inference, so strength 4 still has no accuracy measurement. I’ll use this interruption to separate the two intervention sites and inspect the remaining strength-3 failures. Testing each layer alone can show whether both sites contribute to the repetition effect or whether one is unnecessary. If one site harms productive reasoning without helping repetition, I should remove or weaken it rather than keep increasing both together.

```python
# Hypothesis: layers 15 and 19 make distinct causal contributions to repetitive reasoning.
def repeat_fraction(text, n=8):
    words = text.lower().split()[-2000:]
    grams = [tuple(words[i:i+n]) for i in range(max(0, len(words)-n+1))]
    return 1 - len(set(grams))/len(grams) if grams else 0.0
flat29 = [(row["id"], row["source"], s) for row in gen29 for s in row["samples"]]
for label, pred in [
    ("correct", lambda s: s["correct"]),
    ("wrong_finished", lambda s: not s["correct"] and not s["cut"]),
    ("cut", lambda s: s["cut"])]:
    group = [(pid, src, s) for pid, src, s in flat29 if pred(s)]
    print(label, "n", len(group),
          "median_tokens", float(np.median([s["tokens"] for _, _, s in group])) if group else None,
          "mean_tail_duplicate8", round(float(np.mean([repeat_fraction(s["text"]) for _, _, s in group])), 3) if group else None,
          "think_closed", sum("</think>" in s["text"] for _, _, s in group))
    for pid, src, s in group[:2]:
        print(pid, "tail", repr(s["text"][-260:]))
set_repetition_edit({15: 4, 19: 0})
print("CURRENT", edited())
with open("/work/notes.md", "a") as f:
    f.write("\nAlpha4 paired64x2 generation also failed initialization; no accuracy evidence. "
            "CURRENT L15-only alpha4; next transfer probe isolates the first intervention site.\n")
```
