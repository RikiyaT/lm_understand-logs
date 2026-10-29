The second fresh batch showed no net accuracy gain: seven problems improved and seven regressed. That makes the evidence less uniform than the first batch suggested, especially for competition math. I will pool the paired results and continue with the two larger, preselected holdouts; the edit remains unchanged.

```python
# Hypothesis: any real benefit survives pooling and larger independent paired tests, rather than one favorable batch.
import math
import numpy as np
from scipy.stats import binomtest

def read_paired_generation(file_index):
    originals = {}
    with (work / f"gen_{file_index}_original.jsonl").open() as f:
        for line in f:
            record = json.loads(line)
            assert len(record["samples"]) == 1
            s = record["samples"][0]
            originals[record["id"]] = (record["source"], {
                key: s[key] for key in ("correct", "tokens", "cut")
            })
    rows = []
    with (work / f"gen_{file_index}_edited.jsonl").open() as f:
        for line in f:
            record = json.loads(line)
            assert len(record["samples"]) == 1
            source, original = originals.pop(record["id"])
            assert source == record["source"]
            s = record["samples"][0]
            rows.append({
                "id": record["id"], "source": source, "run": file_index,
                "original_correct": bool(original["correct"]), "edited_correct": bool(s["correct"]),
                "original_tokens": original["tokens"], "edited_tokens": s["tokens"],
                "original_cut": bool(original["cut"]), "edited_cut": bool(s["cut"])
            })
    assert not originals
    return rows

def summarize_pairs(rows):
    assert len({r["id"] for r in rows}) == len(rows)
    n = len(rows)
    d = np.array([int(r["edited_correct"]) - int(r["original_correct"]) for r in rows], dtype=float)
    gain = 100 * float(d.mean())
    se = 100 * float(d.std(ddof=1)) / math.sqrt(n)
    wtr, rtw = int((d == 1).sum()), int((d == -1).sum())
    return {
        "n": n, "original_correct": sum(r["original_correct"] for r in rows),
        "edited_correct": sum(r["edited_correct"] for r in rows),
        "gain_pp": gain, "SE_pp": se, "CI95_pp": [gain - 1.96*se, gain + 1.96*se],
        "wrong_to_right": wtr, "right_to_wrong": rtw,
        "mcnemar_exact_p": float(binomtest(wtr, wtr+rtw).pvalue) if wtr+rtw else 1.0,
        "original_cut": sum(r["original_cut"] for r in rows),
        "edited_cut": sum(r["edited_cut"] for r in rows),
        "mean_tokens_original": float(np.mean([r["original_tokens"] for r in rows])),
        "mean_tokens_edited": float(np.mean([r["edited_tokens"] for r in rows]))
    }

primary_rows = read_paired_generation(30) + read_paired_generation(31)
assert {r["id"] for r in primary_rows} == set(holdout_info["H2"])
h2_results = {"all": summarize_pairs(primary_rows)}
for source in ("gsm8k", "math"):
    h2_results[source] = summarize_pairs([r for r in primary_rows if r["source"] == source])
print("H2 pooled:", json.dumps(h2_results))
(work / "H2_pooled_results.json").write_text(json.dumps(h2_results, indent=2))
primary_runs.append({"set": "H2B", "generation_file_index": 31})
(work / "primary_confirmation_runs.json").write_text(json.dumps(primary_runs))
with (work / "notes.md").open("a") as f:
    f.write("\nPass23/25 gen_31 H2B128:12->12 correct,7 improvements/7 regressions, gain0,SE2.93pp. Frozen candidate unchanged. PooledH2 " + json.dumps(h2_results) + "\n")
    f.write("NEXT PASS24/25: both H3 256 independent problems, n1,max_tokens32768, same frozen L15H7=0. Final pass H4 256 predetermined regardless of result.\n")

h3_request = {"type": "generate", "model": "both", "problems": holdout_info["H3"],
              "n": 1, "max_tokens": 32768}
(work / "pass24_request.json").write_text(json.dumps(h3_request))
assert torch.equal(load(target_name), frozen_w)
print("FORWARD REQUEST:", json.dumps(h3_request))
```
