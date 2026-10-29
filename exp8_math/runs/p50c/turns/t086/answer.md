The strength-4 probe completed successfully, but the summary averages include prompt tokens and cannot establish selectivity. I’ll compare only the matched continuation spans against the original and weaker edits, and inspect the explicit premature-closing tokens separately. I’ll also inspect the saved random-direction control before applying it. If a size-matched unrelated direction has the same effect, the proposed repetition mechanism would be less convincing.

```python
# Hypothesis: alpha-4 selectively suppresses repeated continuations more than productive reasoning.
import numpy as np
import torch
bounds = json.loads(Path("/work/transfer_probe_bounds.json").read_text())
probe_dirs = {0: "/work/fwd_24", 1: "/work/fwd_25", 2: "/work/fwd_26", 4: "/work/fwd_32"}
base_lp = np.load(probe_dirs[0] + "/token_logprobs.npy")
transfer_metrics = {}
for alpha, folder in probe_dirs.items():
    lp = np.load(folder + "/token_logprobs.npy")
    result = {}
    for condition in ("productive", "repetitive"):
        indices = [i for i, b in enumerate(bounds) if b["condition"] == condition]
        branch = [float((lp[i] - base_lp[i])[bounds[i]["start"]-1:bounds[i]["length"]-1].mean())
                  for i in indices]
        tail = [float((lp[i] - base_lp[i])[bounds[i]["length"]-25:bounds[i]["length"]-1].mean())
                for i in indices]
        result[condition] = {"branch_delta": float(np.mean(branch)),
                             "tail24_delta": float(np.mean(tail)),
                             "per_case": np.round(branch, 4).tolist()}
    meta = json.loads(Path(folder + "/meta.json").read_text())
    lengths = meta["lengths"]
    result["premature_close_logp"] = [float(lp[i, lengths[i]-2]) for i in (12, 13)]
    transfer_metrics[alpha] = result
    print("alpha", alpha, json.dumps(result))
print("\nRANDOM CONTROL STRUCTURE")
random_control = torch.load("/work/random_control_match_a4.pt", weights_only=False)
print(type(random_control).__name__)
if isinstance(random_control, dict):
    for key, value in random_control.items():
        if isinstance(value, dict):
            print(key, {k: tuple(v.shape) if torch.is_tensor(v) else v for k, v in value.items()})
        else:
            print(key, tuple(value.shape) if torch.is_tensor(value) else str(value)[:300])
with open("/work/notes.md", "a") as f:
    f.write("\nPass21/fwd32 alpha4 transfer: " + json.dumps(transfer_metrics[4]) + "\n")
```
