The original-model transfer probe is complete, providing aligned baselines across all six subjects. I’ll now measure the confirmed strength-one edit on exactly the same texts, including the unfinished-reasoning controls. I’ll also consolidate the notebook so the strongest accuracy evidence and the unvalidated experiments remain clearly separated after any restart. A useful result would preserve productive-token likelihood while weakening repeated tails, without a broad increase in premature-close likelihood.

```python
# Hypothesis: the confirmed alpha1 edit selectively weakens repetition across six unseen mathematical contexts.
import json
from pathlib import Path
import numpy as np
import torch

request=json.loads(Path("/work/transfer_probe_request.json").read_text())
labels=json.loads(Path("/work/transfer_probe_labels.json").read_text())
meta=json.loads(Path("/work/fwd_24/meta.json").read_text())
lp=np.load("/work/fwd_24/token_logprobs.npy")
bounds=[]
for i,(text,label,n) in enumerate(zip(request["texts"],labels,meta["lengths"])):
    enc=tok.encode(text)
    assert enc.ids==np.load("/work/fwd_24/input_ids.npy")[i,:n].tolist()
    b=next(j for j,(a,z) in enumerate(enc.offsets) if z>label["branch_char"])
    bounds.append({"start":b,"length":n,**label})
    if label["condition"]=="premature_close":
        print("Original premature-close logp",label["case"],round(float(lp[i,n-2]),5))
print("Original branch logp by condition:",
      {condition:round(float(np.mean([lp[i,b["start"]-1:b["length"]-1].mean()
                                      for i,b in enumerate(bounds) if b["condition"]==condition])),5)
       for condition in ("productive","repetitive")})
directions=torch.load("/work/repetition_directions.pt",weights_only=True)
for layer,u in directions.items():
    h=np.load(f"/work/fwd_24/hidden_{layer+1}.npy",mmap_mode="r")
    tails=np.stack([np.asarray(h[i,n-24:n],dtype=np.float32).mean(0)
                    for i,n in enumerate(meta["lengths"][:12])])
    contrast=torch.tensor((tails[1::2]-tails[::2]).mean(0))
    print("Transfer contrast alignment",layer,round(abs(float(contrast@u/contrast.norm())),4))
Path("/work/transfer_probe_bounds.json").write_text(json.dumps(bounds,indent=2))

request["model"]="edited"
plan={"next_forward_pass":17,
      "edit":{"tensors":[f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in (15,19)],
              "formula":"W0 - outer(u,u@W0)","alpha":1,
              "directions":"/work/repetition_directions.pt"},
      "request_file":"/work/transfer_probe_request.json",
      "request_overrides":{"model":"edited"},
      "problem_set":[request["texts"][i].split("\nPlease reason")[0].split("user\n",1)[1]
                     for i in range(0,12,2)],
      "conditions":"productive + fourfold repetition per problem; premature-close controls for cases1,3",
      "return":request["return"]}
Path("/work/current_plan.json").write_text(json.dumps(plan,indent=2))

old=Path("/work/notes.md").read_text()
with open("/work/experiment_history.md","a") as f:
    f.write(old+"\n\n")
summary="""# Math weight editing: consolidated at 16/50 forward passes, Python run36
BEST CONFIRMED: L15 and L19 mlp.down_proj.weight, W=W0-u(u^T W0), alpha1.
Directions saved /work/repetition_directions.pt (torch dict keys15,19); original-derived unit vectors. Backups best_confirmed_a1 and candidate_L15_L19_a1. Relative matrix changes3.526%/3.375%.
Derivation: four matched productive/repetitive contexts (fwd_4), contrast at layer outputs16,20; current directions cosine .996/.995 with reconstructed contrasts. Initial gen1:11/96 correct,85 cutoff; cut tails ~95% duplicate8-word sequences.
Small early tests gen5 neuronL19#11 ablation +6.25pp(SE4.35); gen7 L19-only rank1 +6.25pp(SE4.35). gen8/9 and10/11 reuse same32problems; duplicates NOT independent.
KEY HELD-OUT RESULT pass10/gen13,64G+64M unseen: alpha1 original11/128 vs edited42/128; +24.22pp SE3.96;32 rescues1 loss. GSM12.5->48.44%,MATH4.69->17.19%; cutoff92.19->66.41%. Correct edited median8072 tokens; not skipping thinking.
Dose set in splits.json:64 diagnostic problems (32G32M), n4=256/model.
pass11/gen14 alpha1:27/256->59/256,+12.50pp SE2.61,36rescues4losses.
pass12/gen16 alpha2:29/256->92/256,+24.61pp SE3.96,64rescues1loss; cut88.67->52.34%,wrong finished2->28,noanswer0->3. Alpha2 backed up candidate_L15_L19_a2, NOT independently confirmed.
Alpha3 backed up candidate_L15_L19_a3, never generated successfully. Generation service repeatedly fails initialization after pass12; teacher-forced works. Do NOT infer accuracy for alpha3/head changes.
Separate generation runs have batching-dependent RNG: only compare original vs edited WITHIN one both pass, no cross-pass seed pairing.
Attention pass13/fwd19 ORIGINAL two-text probe, /work/attention_probe_request.json: L15h7 induction .304,L19h5 .308 (attend previous repeat next-token position). Pass14/fwd20 alpha3 MLP: repeated tail logp -.02175->-.08914, productive -1.31965->-1.29310; observed close probs<.001. Pass15/fwd22 HEAD ONLY L15h7+L19h5 outputs*.5: repeated tail -.03071, productive -1.33975 (less selective); generation failed. Head edits REMOVED.
CURRENT weights restored to confirmed MLP alpha1, no head edits.
pass16/fwd24 ORIGINAL transfer probe:6 new math contexts, productive/repetitive pair each +2 premature-close controls. Exact texts /work/transfer_probe_request.json; labels /work/transfer_probe_labels.json; token bounds /work/transfer_probe_bounds.json. Next: same edited alpha1; explore dose and random-direction negative controls, retry generation if service returns. Preserve final new problems for independent confirmation.
Exact eval prompt in /work/eval_prompt_format.txt; use raw texts, not chats. Attention now works <=2 texts <=2048tokens.
Practice JSONL read physical lines, NOT str.splitlines(). Sources gsm8k/math/aime; exact ids have underscores.
Full prior notebook archived /work/experiment_history.md.
"""
Path("/work/notes.md").write_text(summary+"\nNEXT PASS EXACT PLAN:\n"+json.dumps(plan)+"\n")
print("Recorded alpha1 transfer probe; notebook consolidated.")
```
