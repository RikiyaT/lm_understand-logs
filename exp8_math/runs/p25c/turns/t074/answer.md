The control pass completed, but I accidentally changed one number in the productive modular-arithmetic text. That row is not an exact matched comparison and must be excluded; the other five texts can still provide valid controls. I will audit text identity, calculate those comparisons, and then temporarily amplify head 7 to test the opposite predicted effect. Increased repeated-continuation likelihood would strengthen the causal interpretation.

```python
# Hypothesis: the same-layer control has weaker loop effects, while amplifying H7 should reinforce copying.
import numpy as np
import json

regions = json.loads((work / "broad_probe_regions.json").read_text())
base_meta = json.loads((work / "fwd_25" / "meta.json").read_text())
ctrl_meta = json.loads((work / "fwd_28" / "meta.json").read_text())
base_lp = np.load(work / "fwd_25" / "token_logprobs.npy")
ctrl_lp = np.load(work / "fwd_28" / "token_logprobs.npy")
abl_lp = np.load(work / "fwd_26" / "token_logprobs.npy")
control_results = []
for i, reg in enumerate(regions):
    same_text = base_meta["texts"][i] == ctrl_meta["texts"][i]
    row = {"case": reg["case"], "kind": reg["kind"], "exact_text_match": same_text}
    if same_text:
        for label, start in (("body", reg["branch"]), ("late", reg["late"])):
            sl = slice(start - 1, reg["length"] - 1)
            row[label + "_H4_delta"] = float(np.mean(ctrl_lp[i, sl] - base_lp[i, sl]))
            row[label + "_H7_delta"] = float(np.mean(abl_lp[i, sl] - base_lp[i, sl]))
    else:
        row["excluded_reason"] = "transcription error: exponent-9 remainder changed from 6 to 4"
    control_results.append(row)
print("Exact-text control comparisons:", json.dumps(control_results, indent=2))
(work / "same_layer_control_results.json").write_text(json.dumps(control_results, indent=2))

amplified_w = original_w.clone()
amplified_w[:, 1792:2048] *= 1.5
save_edit(target_name, amplified_w)
amplify_request = {
    "type": "teacher_forced", "model": "edited",
    "texts": broad_request["texts"],
    "return": {"token_logprobs": True, "topk": 20}
}
(work / "amplification_request.json").write_text(json.dumps(amplify_request))
with (work / "notes.md").open("a") as f:
    f.write("\nPass20/25 fwd_28: temporary L15H4=0 same-layer control. Productive modular text had accidental exponent9 remainder4 instead of6; exclude that row entirely from matched comparisons. Five other texts exact. Results " + json.dumps(control_results) + "\n")
    f.write("NEXT PASS21/25: temporary ONLY L15H7 output columns1792:2048 multiplied by1.5, H4 original. All six ORIGINAL broad texts, including correct exponent9 remainder6. Bidirectional causal control, not candidate selection. Restore H7=0 immediately afterward.\n")
print("Active temporary edit: L15H7 multiplier 1.5; every other value original.")
print("FORWARD REQUEST:", json.dumps(amplify_request))
```
