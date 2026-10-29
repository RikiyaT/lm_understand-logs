The examples show misread quantities, incorrect algebra, and repeated conclusions, including some incorrect conclusions stated confidently. They also show why “no extracted answer” is not necessarily an unfinished response: a response can contain a mathematical conclusion that the extractor misses. There are 49 mixed-success MATH problems, enough to fix a 32-problem DEV set without using GSM8K. I’ll now compare successful and failed samples within those problems, controlling separately for response length, and rank individual MLP neurons rather than whole layers. A useful candidate should show a consistent failure association across problems, not just activate on long answers.

```python
import re
mixed_math = [
    g["id"] for g in baseline_groups
    if g["source"] == "math" and 0 < counts[g["id"]] < 16
]
dev_ids = sorted(
    mixed_math,
    key=lambda pid: (-counts[pid] * (16 - counts[pid]), pid)
)[:32]
dev_set = set(dev_ids)
assert len(dev_ids) == 32 and not dev_set & holdout_set
with open("/work/DEV_ids.json", "w") as f:
    json.dump(dev_ids, f)
with open("/work/notes.md", "a") as f:
    f.write("\n## DEV fixed before edits\n"
            "- DEV_ids.json: 32 mixed-success MATH problems, selected by highest "
            "baseline Bernoulli variance, ties by id.\n"
            "- HOLDOUT remains untouched. No weights edited.\n")

for r in records:
    t = r["text"]
    words = re.findall(r"\S+", t)
    grams = list(zip(words, words[1:], words[2:], words[3:]))
    r["repeat4"] = 1 - len(set(grams)) / max(1, len(grams))
    r["final_markers"] = len(re.findall(r"final answer", t, re.I))
    r["failure_type"] = (
        "correct" if r["correct"] else
        "cut" if r["cut"] else
        "no_extracted_answer" if r.get("pred") is None else "wrong_extracted_answer"
    )
print("Recounted outcomes:", dict(collections.Counter(r["failure_type"] for r in records)))
print("DEV success counts:", dict(sorted(collections.Counter(counts[p] for p in dev_ids).items())))
print("DEV levels:", dict(collections.Counter(practice[p]["level"] for p in dev_ids)))
print("DEV sources: MATH only")

for correct in [True, False]:
    rr = [r for r in records if r["problem_id"] in dev_set and r["correct"] == correct]
    print("DEV baseline", "correct" if correct else "failed", "n", len(rr),
          "median tokens", round(float(np.median([r["tokens"] for r in rr]))),
          "mean repetition", round(float(np.mean([r["repeat4"] for r in rr])), 3),
          "mean final markers", round(float(np.mean([r["final_markers"] for r in rr])), 2))

dev_indices = np.array([i for i,r in enumerate(records) if r["problem_id"] in dev_set])
dev_records = [records[i] for i in dev_indices]
groups_dev = np.array([dev_ids.index(r["problem_id"]) for r in dev_records])
def within_problem(x):
    x = np.asarray(x, dtype=np.float64).copy()
    for g in np.unique(groups_dev):
        m = groups_dev == g
        x[m] -= x[m].mean(axis=0)
    return x

failure = within_problem(np.array([not r["correct"] for r in dev_records], dtype=float))
loglen = within_problem(np.log1p([r["tokens"] for r in dev_records]))
failure_len = failure - loglen * (loglen @ failure) / (loglen @ loglen)
repeat = within_problem(np.array([r["repeat4"] for r in dev_records]))
repeat_len = repeat - loglen * (loglen @ repeat) / (loglen @ loglen)

mlp_mean = np.load("/work/gen_1_acts/mlp_mean_answer.npy", mmap_mode="r")
hidden_mean = np.load("/work/gen_1_acts/hidden_mean_answer.npy", mmap_mode="r")
neuron_screen = []
for layer in range(18):
    X = np.asarray(mlp_mean[dev_indices, layer, :], dtype=np.float64)
    Xc = within_problem(X)
    Xl = Xc - np.outer(loglen, loglen @ Xc / (loglen @ loglen))
    sd = np.sqrt(np.sum(Xl * Xl, axis=0) + 1e-20)
    corr = (failure_len @ Xl) / (np.linalg.norm(failure_len) * sd)
    rawcorr = (failure @ Xc) / (
        np.linalg.norm(failure) * np.sqrt(np.sum(Xc*Xc, axis=0) + 1e-20)
    )
    repcorr = (repeat_len @ Xl) / (np.linalg.norm(repeat_len) * sd + 1e-20)
    D = load_original(f"model.layers.{layer}.mlp.down_proj.weight").numpy()
    energy = np.sqrt(np.mean(X * X, axis=0)) * np.linalg.norm(D, axis=0)
    for j in range(2048):
        neuron_screen.append({
            "layer": layer, "neuron": j,
            "failure_corr": float(corr[j]), "raw_failure_corr": float(rawcorr[j]),
            "repeat_corr": float(repcorr[j]), "energy": float(energy[j]),
            "mean": float(X[:, j].mean()),
            "sd": float(X[:, j].std())
        })
    top = int(np.argmax(corr))
    print("Layer", layer, "top failure neuron", top,
          "length-controlled corr", round(float(corr[top]), 3),
          "raw corr", round(float(rawcorr[top]), 3),
          "repeat corr", round(float(repcorr[top]), 3))

neuron_screen.sort(key=lambda x: x["failure_corr"], reverse=True)
print("Top 16 neurons:")
for s in neuron_screen[:16]:
    print({k: round(v, 4) if isinstance(v, float) else v for k,v in s.items()})

print("Short excerpts from DEV failures with no extracted answer:")
shown = 0
for r in dev_records:
    if r["failure_type"] == "no_extracted_answer":
        print(r["problem_id"], "tokens", r["tokens"],
              ascii(r["text"][:140] + " ... " + r["text"][-140:]))
        shown += 1
        if shown == 6:
            break

print("DEV IDs:", json.dumps(dev_ids))
```
