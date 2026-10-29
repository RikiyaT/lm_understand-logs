DEV is now fixed at 32 problems, with 49.4% baseline accuracy. The inspected failures include incorrect arithmetic, repeated conclusions, and long chains that lose the original expression—not merely missing boxes. I’ll now read failures across different problems and rank MLP neurons by within-problem differences, checking whether those differences persist after accounting for answer length. A useful candidate should recur across problems rather than be driven by one unusually long failure.

```python
import re

act_index = json.loads((act_dir / "index.json").read_text())
meta_samples = act_index["samples"]
assert len(flat) == len(meta_samples)
print("Activation-label agreement:",
      np.mean([s["correct"] == m["correct"] and s["problem_id"] == m["item"]
               for s, m in zip(flat, meta_samples)]))
dev_indices = np.array([i for i, s in enumerate(flat)
                        if s["problem_id"] in dev_set])
Y = np.array([flat[i]["correct"] for i in dev_indices], dtype=bool)
P = np.array([flat[i]["problem_id"] for i in dev_indices])
L = np.log1p([flat[i]["tokens"] for i in dev_indices])
A = np.load(act_dir / "mlp_mean_answer.npy", mmap_mode="r")
X = np.asarray(A[dev_indices], dtype=np.float64)
mlp_layers = act_index["mlp_layers"]

def repetition(t):
    words = t.split()
    grams = [tuple(words[i:i+6]) for i in range(max(0, len(words)-5))]
    return 1 - len(set(grams))/max(1, len(grams))

print("DEV outcome counts:", dict(Counter(
    "correct" if s["correct"] else "cut" if s["cut"] else
    "no_extracted_answer" if s["pred"] is None else "wrong_answer"
    for s in flat if s["problem_id"] in dev_set)))
for good in (True, False):
    ss = [s for s in flat if s["problem_id"] in dev_set and s["correct"] == good]
    print("correct =", good, "n =", len(ss),
          "median_tokens =", np.median([s["tokens"] for s in ss]),
          "mean_repetition =", round(np.mean([repetition(s["text"]) for s in ss]), 3),
          "multiple_final_answers =", round(np.mean([
              s["text"].lower().count("final answer") > 1 for s in ss]), 3))

for pid in dev_ids[:8]:
    s = next(s for s in flat if s["problem_id"] == pid and not s["correct"])
    print(json.dumps({"id": pid, "problem": by_id[pid]["problem"][:180],
                      "reference": by_id[pid]["answer"],
                      "failure_excerpt": s["text"][:140] + " ... " + s["text"][-140:]},
                     ensure_ascii=True))

Xc = X.copy()
Lc = L.copy()
for pid in dev_ids:
    mask = P == pid
    Xc[mask] -= Xc[mask].mean(axis=0)
    Lc[mask] -= Lc[mask].mean()
beta_len = np.einsum("n,nij->ij", Lc, Xc) / (Lc @ Lc)
Xadj = Xc - Lc[:, None, None] * beta_len[None]
scale = Xc.std(axis=0) + 1e-8
contrasts, adjusted = [], []
for pid in dev_ids:
    mask = P == pid
    contrasts.append((X[mask & ~Y].mean(0)-X[mask & Y].mean(0))/scale)
    adjusted.append((Xadj[mask & ~Y].mean(0)-Xadj[mask & Y].mean(0))/scale)
contrasts = np.stack(contrasts)
adjusted = np.stack(adjusted)
tstat = adjusted.mean(0)/(adjusted.std(0, ddof=1)/np.sqrt(len(dev_ids)) + 1e-8)
ranked = np.argsort(-np.abs(tstat).ravel())[:24]
rank_summary = []
for q in ranked:
    li, neuron = np.unravel_index(q, tstat.shape)
    rank_summary.append({
        "layer": mlp_layers[li], "neuron": int(neuron),
        "t_length_adjusted": round(float(tstat[li, neuron]), 2),
        "effect_raw": round(float(contrasts[:, li, neuron].mean()), 3),
        "effect_adjusted": round(float(adjusted[:, li, neuron].mean()), 3),
        "same_sign_problem_fraction": round(float(np.mean(
            np.sign(adjusted[:, li, neuron]) == np.sign(tstat[li, neuron]))), 3)
    })
print("Top within-problem MLP associations:", json.dumps(rank_summary))
"Observational localization only; no edits."
```