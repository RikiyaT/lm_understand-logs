The broader probes show that the edit reduces repeated-step likelihood more than productive-reasoning likelihood, but it is not perfectly selective. This supports the repetition hypothesis without establishing an accuracy gain. I will keep the candidate frozen and resume fresh paired validation. First I will verify the saved weights and recover the exact held-out problem IDs.

```python
# Hypothesis: the frozen L15H7 ablation improves accuracy by disrupting unproductive repetition.
import json
from pathlib import Path
import torch

work = Path("/work")
target_name = "model.language_model.layers.15.self_attn.o_proj.weight"
current_w = load(target_name)
original_w = load_original(target_name)
assert torch.equal(current_w[:, :1792], original_w[:, :1792])
assert torch.count_nonzero(current_w[:, 1792:2048]).item() == 0
assert torch.equal(current_w[:, 2048:], original_w[:, 2048:])
print("Edited tensors:", edited())
print("Frozen recipe:", json.loads((work / "frozen_final_candidate_recipe.json").read_text()))
print("Earlier diagnosis and localization:\n", (work / "notes.md").read_text()[:5100])

h2a_ids = json.loads((work / "H2A_ids.json").read_text())
h2b_ids = json.loads((work / "H2B_ids.json").read_text())
print("H2A IDs:", json.dumps(h2a_ids))
print("H2B count:", len(h2b_ids))
holdout_info = json.loads((work / "holdout_sets.json").read_text())
print("Holdout structure:", {
    k: (list(v.keys()) if isinstance(v, dict) else len(v))
    for k, v in holdout_info.items()
})
```
