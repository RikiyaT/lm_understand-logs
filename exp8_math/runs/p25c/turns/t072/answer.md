The existing probes contain three productive-versus-repetitive pairs, including a productive calculation with repeated formatting. I will reuse those exact texts and choose a control head that shares head 7’s key/value group, has low repetition-retrieval attention, and has a similar output-weight norm. This tests whether the effect comes from the identified head rather than simply removing a comparable amount of attention output. The control is temporary and will not replace the frozen candidate.

```python
# Hypothesis: a comparable non-copying head will not reproduce L15H7's selective loop suppression.
import numpy as np

layer15_heads = [h for h in loc["heads"] if h["layer"] == 15]
head_norms = [float(original_w[:, 256*h:256*(h+1)].norm()) for h in range(8)]
print("Layer-15 head evidence:", [
    {"head": h["head"], "matching_productive": round(h["ind_good"], 4),
     "matching_loop": round(h["ind_loop"], 4),
     "output_norm": round(head_norms[h["head"]], 3)}
    for h in layer15_heads
])
control_pool = [h["head"] for h in layer15_heads
                if h["head"] in (4, 5, 6) and h["ind_loop"] < 0.2]
assert control_pool
control_head = min(control_pool, key=lambda h: abs(head_norms[h] / head_norms[7] - 1))
control_w = original_w.clone()
control_w[:, 256*control_head:256*(control_head+1)] = 0
save_edit(target_name, control_w)

control_request = {
    "type": "teacher_forced", "model": "edited",
    "texts": broad_request["texts"],
    "return": {"token_logprobs": True, "topk": 20}
}
(work / "control_head_request.json").write_text(json.dumps(control_request))
(work / "control_head_recipe.json").write_text(json.dumps({
    "tensor": target_name, "head": control_head, "multiplier": 0.0,
    "purpose": "temporary same-KV-group negative control"
}))
with (work / "notes.md").open("a") as f:
    f.write(f"\nNEXT PASS20/25: teacher-forced six broad probes; temporary ONLY L15H{control_head}=0, H7 restored to original. Control selected from H4-H6 with matching-loop attention <0.2 by closest original output Frobenius norm to H7. Not a new candidate. Restore frozen H7 ablation after causal controls.\n")
print("Temporary control head:", control_head)
print("FORWARD REQUEST:", json.dumps(control_request))
```
