The layer-19-only probe has finished; I’ll compare it with layer 15 and the joint edit. I’ll then test a more localized version that changes only the 64 previously identified repetition-writing neurons in each MLP, rather than every column’s contribution along the direction. Those neurons were selected from the original diagnostic contexts, not from these transfer results. If this smaller edit preserves the selective effect, it may reduce collateral changes such as whitespace loops.

```python
# Hypothesis: a small set of repetition-writing MLP neurons carries much of the causal effect.
l19_metrics = transfer_summary("/work/fwd_38")
print("L19 ONLY alpha4", json.dumps(l19_metrics))
subset_info = {}
for l in (15, 19):
    w = base_mlp_weights[l]
    u = u_rep[l].float()
    ix = writer_subsets[l].long()
    writer_weights = u @ w
    acts = np.load(f"/work/fwd_24/mlp_act_{l}.npy", mmap_mode="r")
    observed = {}
    for condition in ("productive", "repetitive"):
        rows = [i for i, b in enumerate(bounds) if b["condition"] == condition]
        mean_act = np.mean([acts[i, bounds[i]["start"]:bounds[i]["length"]].mean(axis=0)
                            for i in rows], axis=0)
        attribution = torch.from_numpy(mean_act.copy()) * writer_weights
        observed[condition] = {"all_neurons": float(attribution.sum()),
                               "selected64": float(attribution[ix].sum())}
    masked_writer = torch.zeros_like(writer_weights)
    masked_writer[ix] = writer_weights[ix]
    delta = -4 * torch.outer(u, masked_writer)
    save_edit(f"model.language_model.layers.{l}.mlp.down_proj.weight", w + delta)
    subset_info[l] = {
        "relative_change": float(delta.norm()/w.norm()),
        "writer_norm_fraction": float(masked_writer.norm()/writer_weights.norm()),
        "observed_attribution": observed,
        "selected_first12": ix[:12].tolist()
    }
print("TOP64 alpha4", json.dumps(subset_info))
with open("/work/notes.md", "a") as f:
    f.write("\nPass25/fwd38 L19-only alpha4 transfer: " + json.dumps(l19_metrics) +
            "\nCURRENT top64-writer-column-only alpha4 in L15/L19. "
            "Original fwd4 selection; transfer attributions: " + json.dumps(subset_info) + "\n")
```
