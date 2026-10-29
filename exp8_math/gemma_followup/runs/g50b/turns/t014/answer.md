Full suppression gained 1.56 percentage points on DEV, with a 1.83-point standard error: promising, but not conclusive. It also shortened answers and reduced cut-offs, which I won’t count as success by themselves. I’ll preserve this candidate and test half suppression next, followed by the opposite intervention; a consistent accuracy response would be stronger causal evidence than the activation correlations.

```python
import shutil

def read_gen(k, model):
    with (work / f"gen_{k}_{model}.jsonl").open() as f:
        return [json.loads(line) for line in f]

def summarize_pair(k):
    original, changed = read_gen(k,"original"), read_gen(k,"edited")
    om = {r["id"]:r for r in original}
    diffs, by_problem = [], []
    for r in changed:
        os = om[r["id"]]["samples"]
        ds = [int(e["correct"])-int(o["correct"])
              for o,e in zip(os,r["samples"])]
        diffs.extend(ds)
        by_problem.append(np.mean(ds))
    d = np.array(diffs)
    return {"pass":k, "n":len(d), "gain_pp":100*float(d.mean()),
            "paired_SE_pp":100*float(d.std(ddof=1)/np.sqrt(len(d))),
            "problem_cluster_SE_pp":100*float(
                np.std(by_problem,ddof=1)/np.sqrt(len(by_problem)))}

def checkpoint(label, description):
    path = work / label
    path.mkdir(exist_ok=True)
    shutil.copytree(work / "edited", path / "edited", dirs_exist_ok=True)
    (path / "description.json").write_text(json.dumps(description, indent=2))

checkpoint("best_1", {"candidate":candidate, "DEV_passes":[4],
                      "result":summarize_pair(4)})
experiments = {4:dict(candidate)}
candidate = set_unit_edit(8, units8, 0.5)
experiments[5] = dict(candidate)
with (work / "notes.md").open("a") as f:
    f.write("\nPass 4: +1.56 pp, paired SE 1.83 pp; saved best_1. "
            "Not a significant accuracy gain.\nUpcoming pass 5: BOTH, "
            "DEV_ids.json, n=32, max_tokens=8192. Exact edit: "
            + json.dumps(candidate) + ". All other tensors original.\n")
print("Pass 4 computed summary:", summarize_pair(4))
print("Pass 5 edit:", candidate)
"Half-suppression installed; checkpoint preserved."
```