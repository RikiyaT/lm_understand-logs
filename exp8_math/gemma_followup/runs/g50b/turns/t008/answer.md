The top individual neurons have modest contributions, so removing one may be too weak to test the failure mechanism clearly. I’ll also examine a rank-one MLP-output direction formed from the length-adjusted failure-minus-success contrasts, checking agreement between two halves of DEV. This is an analytical projection, not fitted training. For screening, I’ll use compact reference-answer continuations for every DEV problem; their likelihood is only a proxy, so generation accuracy will remain decisive.

```python
directions = {}
direction_summary = []
raw_deltas = []
for pid in dev_ids:
    m = P == pid
    raw_deltas.append(Xadj[m & ~Y].mean(0) - Xadj[m & Y].mean(0))
raw_deltas = np.stack(raw_deltas)
split = np.random.default_rng(1701).permutation(len(dev_ids))

for layer in (6, 8, 9, 10, 11, 12, 14, 16, 17):
    li = mlp_layers.index(layer)
    W = load_original(f"model.layers.{layer}.mlp.down_proj.weight").double()
    delta = torch.from_numpy(raw_deltas[:,li].mean(0))
    v = W @ delta
    u = v / v.norm()
    halves = [W @ torch.from_numpy(raw_deltas[ix,li].mean(0))
              for ix in (split[:16], split[16:])]
    agreement = torch.nn.functional.cosine_similarity(halves[0], halves[1], dim=0)
    fraction = (u @ W).norm() / W.norm()
    directions[layer] = u.float()
    direction_summary.append({
        "layer": layer, "split_half_cosine": round(float(agreement),3),
        "removed_weight_fraction_at_alpha_1": round(float(fraction),4),
        "contrast_norm": round(float(v.norm()),5)
    })
torch.save(directions, work / "DEV_failure_directions.pt")
print("Direction diagnostics:", json.dumps(direction_summary))

compact_texts = [
    PREFIX + by_id[p]["problem"] + SUFFIX
    + "Final Answer: $\\boxed{" + by_id[p]["answer"] + "}$."
    for p in dev_ids
]
(work / "DEV_compact_reference_texts.json").write_text(json.dumps(compact_texts))
print("Remaining DEV prompts and answers:")
for pid in dev_ids[21:]:
    r = by_id[pid]
    print(json.dumps({"id":pid, "problem":r["problem"], "answer":r["answer"]},
                     ensure_ascii=True))
with (work / "notes.md").open("a") as f:
    f.write("\nPass 2 scoring clarification: compact boxed reference-answer "
            "continuations for all 32 DEV prompts, not full reference solutions; "
            "fixed in DEV_compact_reference_texts.json. Original weights. "
            "Return token logprobs, hidden states 8/9/11/12, MLP 8/11. "
            "This tests answer likelihood, not generated reasoning accuracy.\n")
"Failure directions saved analytically; weights remain original."
```