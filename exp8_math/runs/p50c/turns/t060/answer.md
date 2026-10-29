Generation startup is still failing, so strength three remains unvalidated on accuracy; an engine reset or fuller startup log from the experimenters would help. I’ll continue with a causal test that the working teacher-forced tool can support. Specifically, I’ll temporarily remove the MLP edits and halve only the outputs of the two copying heads identified by attention. If those heads sustain repetition, this should weaken repeated-token likelihood without broadly damaging the productive continuation.

```python
# Hypothesis: L15 head7 and L19 head5 causally sustain repetition, independently of the MLP-direction edits.
import json, shutil
from pathlib import Path
import torch

shutil.copytree("/work/edited","/work/candidate_L15_L19_a3",dirs_exist_ok=True)
print("Attention output tensor names:",
      [n for n in NAMES if any(f".layers.{l}." in n for l in (15,19))
       and ("o_proj" in n or "q_proj" in n)])
for name in list(edited()):
    drop_edit(name)

head_sites = [(15,7),(19,5)]
head_tensors = []
for layer,head in head_sites:
    name=f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
    assert name in NAMES
    w=load_original(name)
    assert tuple(w.shape)==(1024,2048), tuple(w.shape)
    candidate=w.clone()
    candidate[:,head*256:(head+1)*256] *= 0.5
    save_edit(name,candidate)
    head_tensors.append(name)
    print("layer",layer,"head",head,"output gain",0.5,
          "relative matrix change",round(float((candidate-w).norm()/w.norm()),5))

request=json.loads(Path("/work/attention_probe_request.json").read_text())
request["model"]="edited"
plan={"next_forward_pass":15,
      "edit":{"tensors":head_tensors,"sites":head_sites,"head_output_gain":0.5,
              "formula":"Only columns head*256:(head+1)*256 of original o_proj multiplied by 0.5",
              "mlp_edits":"none"},
      "request":request}
Path("/work/current_plan.json").write_text(json.dumps(plan,indent=2))
with open("/work/notes.md","a") as f:
    f.write("\nAlpha3 generation failed startup a third time; no generation evidence. "
            "TF alpha3 repeated-tail logp -.02175->-.08914; productive-tail -1.31965->-1.29310; "
            "observed think-close probabilities <.001. Saved candidate_L15_L19_a3. "
            "Now isolated causal head test: all MLP edits dropped, L15h7 and L19h5 output gains .5.\n"
            "NEXT PASS EXACT PLAN:\n"+json.dumps(plan)+"\n")
print("Current edits:",edited())
```
