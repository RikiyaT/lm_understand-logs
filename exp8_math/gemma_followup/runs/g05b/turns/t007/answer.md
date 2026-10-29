DEV is now fixed, with 217 correct and 295 failed baseline samples. Failed responses are somewhat longer and more repetitive, but the strongest length-controlled neuron–failure correlations are only about 0.22; selecting the largest correlation alone would risk chasing noise. I’ll check whether those associations persist across separate halves of DEV and look for stronger links to repeated, non-progressing reasoning. I’ll also count cases where a response abandons a correct boxed result. These checks will help distinguish a plausible failure mechanism from a general difficulty signal before teacher-forced localization.

```python
def normalized_answer(s):
    if s is None:
        return None
    return re.sub(r"[\s$]", "", str(s)).replace(r"\dfrac", r"\frac")

def boxed_contents(text):
    out = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        start, depth, j = m.end(), 1, m.end()
        while j < len(text) and depth:
            depth += (text[j] == "{") - (text[j] == "}")
            j += 1
        if depth == 0:
            out.append(text[start:j-1])
    return out

box_stats = collections.Counter()
for r in dev_records:
    boxes = boxed_contents(r["text"])
    ref = normalized_answer(r["reference"])
    matches = [normalized_answer(b) == ref for b in boxes]
    box_stats["responses_with_box"] += bool(boxes)
    box_stats["multiple_boxes"] += len(boxes) > 1
    box_stats["correct_box_then_wrong_last_box"] += (
        bool(matches) and any(matches[:-1]) and not matches[-1]
    )
    box_stats["failed_despite_earlier_matching_box"] += (
        not r["correct"] and any(matches[:-1])
    )
print("DEV box diagnostics:", dict(box_stats))

# Stability across disjoint halves of the DEV problems.
for layer in range(18):
    X = np.asarray(mlp_mean[dev_indices, layer, :], dtype=np.float64)
    Xc = within_problem(X)
    for half in [0, 1]:
        mask = groups_dev % 2 == half
        z = loglen[mask]
        Y = Xc[mask] - np.outer(z, z @ Xc[mask] / (z @ z))
        f = failure[mask] - z * (z @ failure[mask]) / (z @ z)
        rr = (f @ Y) / (
            np.linalg.norm(f) * np.sqrt(np.sum(Y*Y, axis=0) + 1e-20)
        )
        for s in neuron_screen:
            if s["layer"] == layer:
                s[f"half{half}_corr"] = float(rr[s["neuron"]])

stable = [
    s for s in neuron_screen
    if s["half0_corr"] > .06 and s["half1_corr"] > .06
    and s["failure_corr"] > .10 and s["mean"] > 0
]
for s in stable:
    s["robust_score"] = min(s["half0_corr"], s["half1_corr"]) * math.sqrt(s["energy"])
stable.sort(key=lambda s: s["robust_score"], reverse=True)

print("Stable, positive-mean failure-associated components:")
for s in stable[:18]:
    print({k: round(s[k], 4) if isinstance(s[k], float) else s[k]
           for k in ["layer","neuron","failure_corr","half0_corr","half1_corr",
                     "repeat_corr","energy","mean"]})

print("Strongest length-controlled repetition associations:")
for s in sorted(neuron_screen, key=lambda s: s["repeat_corr"], reverse=True)[:12]:
    print({k: round(s[k], 4) if isinstance(s[k], float) else s[k]
           for k in ["layer","neuron","repeat_corr","failure_corr","energy","mean"]})

print("Matched failing DEV excerpts:")
for pid in dev_ids[:6]:
    rr = [r for r in dev_records if r["problem_id"] == pid and not r["correct"]]
    r = min(rr, key=lambda x: abs(x["tokens"] - 500))
    print(pid, "reference", ascii(str(practice[pid]["answer"])[:60]),
          "response", ascii(r["text"][:140] + " ... " + r["text"][-140:]))

with open("/work/neuron_screen.json", "w") as f:
    json.dump(neuron_screen, f)
print("No edits; holdout remains untouched.")
```
