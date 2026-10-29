This batch gained 46.9 percentage points, with 31 rescues and the first regression. That supports a large net benefit but also shows the edit is not universally harmless. I will inspect the lost case and quantify repetition and thinking length across the confirmation runs; persistent long reasoning in correct answers would argue against a generic “stop thinking early” effect.

```python
# Hypothesis: Gains come from escaping repetitive reasoning, not from indiscriminately terminating thinking.
import numpy as np

register_frozen_run(54)

def behavior_summary(run_ids):
    output = {}
    for model in ["original", "edited"]:
        correct_lengths, cut_repetition, finished_repetition = [], [], []
        counts = Counter()
        for run_id in run_ids:
            with open(f"/work/gen_{run_id}_{model}.jsonl") as f:
                for line in f:
                    r = json.loads(line)
                    for s in r["samples"]:
                        t = s["text"]
                        counts["n"] += 1
                        counts["correct"] += bool(s["correct"])
                        counts["cut"] += bool(s["cut"])
                        counts["whitespace_tail"] += bool(s["cut"] and len(t) >= 400 and not t[-400:].strip())
                        if s["correct"]:
                            correct_lengths.append(s["tokens"])
                            counts["correct_thinking_close"] += "</think>" in t
                            counts["correct_close_after_1000_chars"] += t.find("</think>") > 1000
                        words = t.split()[-2000:]
                        grams = [tuple(words[i:i+8]) for i in range(max(0, len(words)-7))]
                        rep = 1-len(set(grams))/len(grams) if grams else 0.0
                        (cut_repetition if s["cut"] else finished_repetition).append(rep)
        output[model] = dict(counts)
        output[model].update({
            "median_correct_tokens": float(np.median(correct_lengths)) if correct_lengths else None,
            "mean_tail_duplicate8_cut": float(np.mean(cut_repetition)) if cut_repetition else None,
            "mean_tail_duplicate8_finished": float(np.mean(finished_repetition)) if finished_repetition else None
        })
    return output

beh = behavior_summary(confirmed_runs)
print("CONFIRMATION BEHAVIOR", json.dumps(beh))
with open("/work/notes.md", "a") as f:
    f.write("\nBehavior through gen54: " + json.dumps(beh) + "\n")

old54 = {r["id"]: r for r in map(json.loads, Path("/work/gen_54_original.jsonl").read_text().splitlines())}
new54 = {r["id"]: r for r in map(json.loads, Path("/work/gen_54_edited.jsonl").read_text().splitlines())}
for pid in old54:
    a, b = old54[pid]["samples"][0], new54[pid]["samples"][0]
    if a["correct"] and not b["correct"]:
        print("REGRESSION", pid)
        for label, s in [("original", a), ("edited", b)]:
            print(label, {k:s[k] for k in ["pred", "correct", "tokens", "cut"]})
            print("tail:", s["text"][-240:])
```
