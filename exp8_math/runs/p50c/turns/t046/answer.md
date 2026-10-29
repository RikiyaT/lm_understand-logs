The independent test strongly supports the edit: accuracy rose from 8.6% to 32.8%, a paired gain of 24.2 ± 4.0 percentage points, with 32 rescues and one regression. Both GSM8K and MATH improved, so this is more than a reduction in cutoffs. I’ll preserve this confirmed candidate, inspect the remaining failures, and build a fixed dose-response test using 256 answers per model. Stronger suppression should reduce repeated reasoning while preserving substantive calculations; premature endings or lost correct answers would count against it.

```python
# Hypothesis: the confirmed rank-one edits interrupt repetitive thinking, and residual failures may respond to a stronger dose.
import json, re, shutil
from pathlib import Path
from collections import Counter
import numpy as np
import torch

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

original13 = read_jsonl("/work/gen_13_original.jsonl")
edited13 = read_jsonl("/work/gen_13_edited.jsonl")
shutil.copytree("/work/edited", "/work/best_confirmed_a1", dirs_exist_ok=True)

for label, rows in (("original", original13), ("edited", edited13)):
    ss = [s for r in rows for s in r["samples"]]
    correct_ss = [s for s in ss if s["correct"]]
    print(label, "correct =", len(correct_ss),
          "correct-token median =", float(np.median([s["tokens"] for s in correct_ss])),
          "correct-token range =", (min(s["tokens"] for s in correct_ss), max(s["tokens"] for s in correct_ss)),
          "think-close count =", sum("</think>" in s["text"] for s in ss))
    reps = []
    for s in ss:
        words = re.findall(r"\S+", s["text"])[-2500:]
        ngrams = [tuple(words[i:i+8]) for i in range(max(0, len(words)-7))]
        reps.append(1 - len(set(ngrams))/max(1, len(ngrams)))
    print(" tail duplicate 8-word fraction =", round(float(np.mean(reps)), 4),
          "among cut =", round(float(np.mean([v for v,s in zip(reps,ss) if s["cut"]])), 4))

print("\nFAILURE EXCERPTS (edited)")
for r in [r for r in edited13 if r["samples"][0]["cut"]][:5]:
    s = r["samples"][0]
    print(r["id"], "tail:", s["text"][-270:].replace("\n", " "))
print("\nRESCUE EXCERPTS")
rescues = [(a,b) for a,b in zip(original13,edited13)
           if not a["samples"][0]["correct"] and b["samples"][0]["correct"]]
for a,b in rescues[:4]:
    print(b["id"], "tokens =", b["samples"][0]["tokens"],
          "start:", b["samples"][0]["text"][:240].replace("\n", " "))

directions = torch.load("/work/repetition_directions.pt", weights_only=True)
meta4 = json.loads(Path("/work/fwd_4/meta.json").read_text())
alignment = {}
for layer in range(25):
    p = Path(f"/work/fwd_4/hidden_{layer}.npy")
    if not p.exists():
        continue
    h = np.load(p, mmap_mode="r")
    ends = np.stack([np.asarray(h[i, max(0,n-32):n], dtype=np.float32).mean(0)
                     for i,n in enumerate(meta4["lengths"])])
    direction = torch.tensor((ends[1::2] - ends[::2]).mean(0))
    direction /= direction.norm()
    alignment[layer] = {site: round(abs(float(direction @ u)), 4)
                        for site,u in directions.items()}
print("\nMATCHED-CONTRAST ALIGNMENT (top layers)")
for site in (15,19):
    print(site, sorted([(l,v[site]) for l,v in alignment.items()], key=lambda x:-x[1])[:5])

diag = read_jsonl("/work/gen_1.jsonl")
dose_ids = ([r["id"] for r in diag if r["source"] == "gsm8k"][:32]
            + [r["id"] for r in diag if r["source"] == "math"][:32])
assert len(dose_ids) == 64
splits = json.loads(Path("/work/splits.json").read_text())
splits["dose_response_64"] = dose_ids
Path("/work/splits.json").write_text(json.dumps(splits, indent=2))
with open("/work/notes.md", "a") as f:
    f.write("\nPass10/gen13 independent validation (64G+64M): alpha1 L15+L19, "
            "original 11/128=8.59%, edited42/128=32.81%, gain24.22pp SE3.96pp, 32 rescues/1 loss. "
            "GSM12.5->48.44%; MATH4.69->17.19%; cutoff92.19->66.41%; mean tokens30663->24477. "
            "Backed up best_confirmed_a1. Dose-response set:64 diagnostic problems x4 samples=256/model; "
            "final confirmation will use new problems.\n")
print("\nDOSE-RESPONSE BASELINE REQUEST")
print(json.dumps({"type":"generate","model":"both","problems":dose_ids,"n":4,"max_tokens":32768}))
```
