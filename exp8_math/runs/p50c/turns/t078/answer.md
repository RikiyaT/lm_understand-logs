The strength-four request failed during startup, so it still has no accuracy result; the two-sample layout does not eliminate the infrastructure issue. I’ll allow a cleanup interval before one retry, without changing weights or decoding settings. Meanwhile, I’ll prepare—but not install—a random-direction control with the same layer locations and weight-change magnitude. That control will distinguish a repetition-specific repair from a generic perturbation of the network.

```python
# Hypothesis: startup failures may be transient; a norm-matched orthogonal-direction control will test edit specificity.
import json, time
from pathlib import Path
import torch

directions=torch.load("/work/repetition_directions.pt",weights_only=True)
basis=torch.linalg.qr(torch.stack([directions[15],directions[19]],dim=1)).Q
rng=torch.Generator().manual_seed(492698)
control={}
for layer,u in directions.items():
    v=torch.randn(1024,generator=rng)
    v=v-basis@(basis.T@v)
    v=v/v.norm()
    name=f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    w=load_original(name)
    alpha=4*float((u@w).norm()/(v@w).norm())
    control[layer]={"direction":v,"alpha":alpha}
    print("Control layer",layer,"alpha",round(alpha,5),
          "cosine with target",round(float(v@u),7),
          "matched_relative_change",round(float(alpha*(v@w).norm()/w.norm()),5))
torch.save(control,"/work/random_control_match_a4.pt")

plan=json.loads(Path("/work/current_plan.json").read_text())
assert plan["edit"]["alpha"]==4
with open("/work/notes.md","a") as f:
    f.write("\nAlpha4 generation startup failed, no counted pass. Current alpha4 unchanged. "
            "Prepared NOT APPLIED random-direction control in random_control_match_a4.pt: seed492698, "
            "each unit direction orthogonal to both repetition directions; coefficient chosen to match "
            "alpha4 edit Frobenius norm per layer. Waiting90s for possible engine cleanup before retry.\n"
            "RETRY EXACT PLAN:\n"+json.dumps(plan)+"\n")
print("Waiting 90 seconds; no model computation or configuration change.")
time.sleep(90)
print("Cleanup interval complete; alpha4 remains installed.")
```
