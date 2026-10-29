The complementary twelve also produced only a small, uncertain gain: +0.78 points with a 1.86-point standard error. The combined group remains plausible, but neither subset provides strong localization yet. I’ll test a matched control group of unrelated layer 8 neurons and measure how often the model overwrites an already-correct boxed result. This will help distinguish targeted repair from generic perturbation or formatting changes.

```python
def boxes(text):
    result = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start, depth, j = match.end(), 1, match.end()
        while j < len(text) and depth:
            depth += (text[j] == "{") - (text[j] == "}")
            j += 1
        if depth == 0:
            result.append(text[start:j-1])
    return result

def normalize_answer(s):
    return re.sub(r"\s+", "", str(s)).replace("\\dfrac","\\frac").strip("$.")

def behavior_stats(rows):
    samples = [(r["id"], s) for r in rows for s in r["samples"]]
    overwrite, conflicting = 0, 0
    for pid,s in samples:
        bs = [normalize_answer(b) for b in boxes(s["text"])]
        ref = normalize_answer(by_id[pid]["answer"])
        overwrite += bool(ref in bs and not s["correct"])
        conflicting += len(set(bs)) > 1
    return {"n":len(samples), "correct_box_but_final_failure":overwrite,
            "conflicting_boxes":conflicting,
            "mean_repeat_fraction":round(np.mean([repetition(s["text"])
                                                  for _,s in samples]),4)}
print("Behavior diagnostics:", {
    f"{k}_{m}":behavior_stats(read_gen(k,m))
    for k in (4,5,6) for m in ("original","edited")
})

W8 = load_original("model.layers.8.mlp.down_proj.weight")
norm8 = W8.norm(dim=0).numpy()
rms8 = np.sqrt(np.mean(X[:,li8]**2,axis=0))
available = set(np.where(np.abs(tstat[li8]) < 0.5)[0].tolist()) - set(units8)
control8 = []
for j in units8:
    pool = np.array(sorted(available))
    dist = np.log((norm8[pool]+1e-8)/(norm8[j]+1e-8))**2
    dist += np.log((rms8[pool]+1e-8)/(rms8[j]+1e-8))**2
    q = int(pool[np.argmin(dist)])
    control8.append(q)
    available.remove(q)
candidate = prepare_units(9,8,control8,1.0,
    "Negative control: 16 low-association neurons matched to selected "
    "units by column norm and activation RMS.")
print("Control edit:",candidate)
"Matched-control test prepared."
```