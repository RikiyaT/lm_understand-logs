On the larger diagnostic test, the current edit gained 12.5 ± 2.6 percentage points, with 36 rescues and four regressions. This corroborates the held-out result, although the remaining cutoff rate is still high. I’ll now double only the two repetition-direction corrections, leaving all orthogonal weight components unchanged. A further accuracy gain would support the predicted dose-response; a rise in wrong finished answers would indicate excessive suppression.

```python
# Hypothesis: stronger correction of the same two repetition directions reduces persistent loops without suppressing useful reasoning.
import json
from pathlib import Path
import torch

directions = torch.load("/work/repetition_directions.pt", weights_only=True)
for layer, u in directions.items():
    name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    w = load_original(name)
    candidate = w - 2.0 * torch.outer(u, u @ w)
    save_edit(name, candidate)
    print("layer", layer, "alpha", 2.0,
          "relative_change", round(float((candidate-w).norm()/w.norm()), 5))
with open("/work/notes.md", "a") as f:
    f.write("\nPass11/gen14 dose set64x4: alpha1 L15+L19, original27/256=10.55%, edited59/256=23.05%, "
            "gain12.50pp SE2.61pp, 36 rescues/4 losses, cutoff91.41->74.22%. "
            "Now testing alpha2 on identical problems/seeds. Confirmed alpha1 backup retained.\n")
print("Current edited tensors:", edited())
```
