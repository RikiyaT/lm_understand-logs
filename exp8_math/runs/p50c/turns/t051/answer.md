Strength two improved accuracy by 24.6 ± 4.0 percentage points within its paired run, with 64 rescues and one regression. It also produced more finished wrong answers, so stronger suppression has a real tradeoff rather than simply solving every loop. I’ll test strength three to locate that tradeoff, and I’ll use only within-run paired gains for comparison because batching makes separate runs nonidentical. I’ll also record the exact edit and test set before each pass, as clarified by the experimenters.

```python
# Hypothesis: strength three may release additional repetitive failures, but excessive correction will increase reasoning errors.
import json, shutil
from pathlib import Path
import torch
import numpy as np

print("EXACT EVALUATION PROMPT FORMAT")
print(Path("/work/eval_prompt_format.txt").read_text()[:5000])

shutil.copytree("/work/edited", "/work/candidate_L15_L19_a2", dirs_exist_ok=True)
directions = torch.load("/work/repetition_directions.pt", weights_only=True)
dose_ids = json.loads(Path("/work/splits.json").read_text())["dose_response_64"]

print("\nDIRECTIONAL WRITERS")
for layer, u in directions.items():
    name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    w = load_original(name)
    scores = u @ w
    top = torch.topk(scores.abs(), 8).indices.tolist()
    print("layer", layer, "top neuron ids", top,
          "directional coefficients", [round(float(scores[j]),4) for j in top],
          "column norm median/max", round(float(w.norm(dim=0).median()),4),
          round(float(w.norm(dim=0).max()),4))
    candidate = w - 3.0 * torch.outer(u, scores)
    save_edit(name, candidate)
    print("alpha3 relative change", round(float((candidate-w).norm()/w.norm()),5))

plan = {
    "next_forward_pass": 13,
    "edit": {"tensors": [f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in (15,19)],
             "formula": "W_original - alpha * outer(u, u @ W_original)",
             "alpha": 3.0, "directions": "/work/repetition_directions.pt"},
    "request": {"type":"generate","model":"both","problems":dose_ids,"n":4,"max_tokens":32768}
}
Path("/work/current_plan.json").write_text(json.dumps(plan, indent=2))
with open("/work/notes.md", "a") as f:
    f.write("\nPass12/gen16 alpha2 on dose64x4: original29/256, edited92/256; gain24.61pp SE3.96pp, "
            "64 rescues/1 regression; cutoff88.67->52.34%, wrong finished2->28, no-answer finished0->3. "
            "Saved candidate_L15_L19_a2. Separate generation runs have batching-dependent RNG: "
            "compare only edited-original within each both pass, never treat outputs across passes as seed-paired. "
            "Exact raw prompt is /work/eval_prompt_format.txt; attention outputs now work.\n"
            "NEXT PASS EXACT PLAN:\n" + json.dumps(plan) + "\n")
print("\nRecorded next pass: alpha3, both, 64 problems x4 samples, 32768 tokens.")
```
