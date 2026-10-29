The random-control probe completed, so I’ll quantify its continuation effects using exactly the same token spans. I’ll also inspect a few remaining failures from the strength-3 generation run to see whether repetition is still the main limitation or whether completed mathematical errors now dominate. A useful repair should preserve substantive reasoning, not merely replace long loops with unsupported answers. These checks will guide whether to strengthen the directional edit or narrow it to particular MLP neurons.

```python
# Hypothesis: the repetition direction is specific, and stronger edits leave a mixture of loops and genuine math errors.
def transfer_summary(folder):
    lp = np.load(str(Path(folder) / "token_logprobs.npy"))
    base = np.load("/work/fwd_24/token_logprobs.npy")
    out = {}
    for condition in ("productive", "repetitive"):
        ix = [i for i, b in enumerate(bounds) if b["condition"] == condition]
        deltas = [float((lp[i]-base[i])[bounds[i]["start"]-1:bounds[i]["length"]-1].mean())
                  for i in ix]
        out[condition] = {
            "branch_delta": float(np.mean(deltas)),
            "per_case": np.round(deltas, 4).tolist(),
            "tail24_delta": float(np.mean([
                (lp[i]-base[i])[bounds[i]["length"]-25:bounds[i]["length"]-1].mean() for i in ix]))
        }
    lengths = json.loads(Path(folder, "meta.json").read_text())["lengths"]
    out["premature_close_logp"] = [float(lp[i, lengths[i]-2]) for i in (12, 13)]
    return out
random_metrics = transfer_summary("/work/fwd_34")
print("RANDOM CONTROL", json.dumps(random_metrics))
gen29 = [json.loads(line) for line in open("/work/gen_29_edited.jsonl")]
print("\nGENERATION ROW KEYS", list(gen29[0]))
for row in gen29[:2]:
    print({k: v for k, v in row.items()
           if not isinstance(v, (list, dict)) and (not isinstance(v, str) or len(v) < 100)})
print("\nWRITER SUBSET STRUCTURE")
writer_subsets = torch.load("/work/repetition_writer_top64.pt", weights_only=False)
for key, value in writer_subsets.items():
    print(key, {k: tuple(v.shape) if torch.is_tensor(v) else str(v)[:100]
                for k, v in value.items()} if isinstance(value, dict) else
          (tuple(value.shape) if torch.is_tensor(value) else str(value)[:200]))
with open("/work/notes.md", "a") as f:
    f.write("\nPass22/fwd34 matched random control transfer: " + json.dumps(random_metrics) + "\n")
```
