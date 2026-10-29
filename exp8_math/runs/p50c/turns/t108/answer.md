The strength-6 probe ran, but one repetitive text was accidentally shorter by one repeated sentence. I’ll compare only identical common-prefix tokens for that case and flag the difference, rather than silently treating the spans as identical. I’ll then retry generation with one sample per exploratory problem at the full token limit. That restores the evaluation-length budget while reducing batch size; any successful result will still need fresh-set confirmation.

```python
# Hypothesis: alpha6 on the selected writers strengthens repetition suppression while preserving useful continuations.
def transfer_summary_aligned(folder):
    lp = np.load(str(Path(folder) / "token_logprobs.npy"))
    base = np.load("/work/fwd_24/token_logprobs.npy")
    ids = np.load(str(Path(folder) / "input_ids.npy"))
    base_ids = np.load("/work/fwd_24/input_ids.npy")
    lengths = json.loads(Path(folder, "meta.json").read_text())["lengths"]
    common_lengths = [min(b["length"], lengths[i]) for i, b in enumerate(bounds)]
    for i, end in enumerate(common_lengths):
        assert np.array_equal(ids[i, :end], base_ids[i, :end]), ("nonmatching prefix", i)
    out = {"shortened_cases": [i for i, b in enumerate(bounds) if lengths[i] != b["length"]]}
    for condition in ("productive", "repetitive"):
        ix = [i for i, b in enumerate(bounds) if b["condition"] == condition]
        deltas = [float((lp[i]-base[i])[bounds[i]["start"]-1:common_lengths[i]-1].mean()) for i in ix]
        out[condition] = {"branch_delta": float(np.mean(deltas)), "per_case": np.round(deltas, 4).tolist(),
                          "tail24_delta": float(np.mean([
                              (lp[i]-base[i])[common_lengths[i]-25:common_lengths[i]-1].mean() for i in ix]))}
    out["premature_close_logp"] = [float(lp[i, lengths[i]-2]) for i in (12, 13)]
    return out
subset6_metrics = transfer_summary_aligned("/work/fwd_44")
print("TOP64 alpha6", json.dumps(subset6_metrics))
print("Current edited tensors", edited())
with open("/work/notes.md", "a") as f:
    f.write("\nPass29/fwd44 TOP64alpha6 transfer: " + json.dumps(subset6_metrics) +
            ". Case7 accidentally one repeated sentence shorter; all comparisons restricted "
            "to verified identical common-prefix tokens. 16k generation alpha6 failed initialization (not counted). "
            "Next full32768 exploratory64x1 retry; CURRENT top64alpha6 unchanged.\n")
```
