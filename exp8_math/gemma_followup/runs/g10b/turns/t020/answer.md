The full ablation changes local probabilities only modestly: valid-reference scores are largely preserved, while effects on the eight loop excerpts are mixed. That does not establish a general anti-repetition mechanism, so the hypothesis remains tentative. I’ll compare only matching, unpadded token positions—the manually submitted `+1` excerpts differ in length—and then test full-strength ablation on DEV generation. Improved answer accuracy, not merely changed wording or shorter responses, will determine whether it is useful.

```python
with open(wdir / "fwd_4" / "meta.json") as f:
    meta4 = json.load(f)
lengths4 = np.array(meta4["lengths"])
ids4 = np.load(wdir / "fwd_4" / "input_ids.npy")
lp4 = np.load(wdir / "fwd_4" / "token_logprobs.npy")
tf_delta = []
tf_delta_late = []
for i in range(72):
    common_len = min(lengths3[i], lengths4[i])
    equal_positions = ids3[i, :common_len] == ids4[i, :common_len]
    first_diff = np.where(~equal_positions)[0]
    if len(first_diff):
        common_len = int(first_diff[0])
    assert common_len > 16
    d = lp4[i, :common_len-1].astype(np.float64) - lp3[i, :common_len-1]
    tf_delta.append(float(d.mean()))
    tf_delta_late.append(float(d[min(16, len(d)//2):].mean()))
tf_delta = np.array(tf_delta)
tf_delta_late = np.array(tf_delta_late)
print("Reference mean delta logprob/token:", round(float(tf_delta[:64].mean()), 6),
      "SE across references:", round(float(tf_delta[:64].std(ddof=1)/8), 6))
print("Reference late-token delta:", round(float(tf_delta_late[:64].mean()), 6))
print("Loop mean delta:", round(float(tf_delta[64:].mean()), 6))
print("Loop late-token deltas:", [round(v, 6) for v in tf_delta_late[64:]])
print("References improved:", int((tf_delta[:64] > 0).sum()), "/64")
print("Loop probabilities reduced:", int((tf_delta[64:] < 0).sum()), "/8")
tf_summary = {
    "reference_delta_mean": float(tf_delta[:64].mean()),
    "reference_delta_se": float(tf_delta[:64].std(ddof=1)/8),
    "loop_delta_mean": float(tf_delta[64:].mean()),
    "loop_late_delta": tf_delta_late[64:].tolist(),
    "interpretation": "Small, mixed local effects; no general loop-suppression claim."
}
(wdir / "teacher_forced_comparison.json").write_text(json.dumps(tf_summary, indent=2))
with open(wdir / "notes.md", "a") as f:
    f.write("\n## Pass 4 findings\n" + json.dumps(tf_summary) + "\n"
            "Matching unpadded prefixes only; pass3 and pass4 +1 excerpts have different lengths.\n"
            "\n## Pass 5 planned: full paired DEV generation\n"
            "Exact edit: down_proj columns 8:[699],14:[483,984],17:[1179] zeroed (strength1.0).\n"
            "Set: all 64 DEV.json problems. model=both, n=16 (1024 answers/model), max_tokens8192.\n"
            "Measure accuracy first, then repeated conclusions, repeated lines and cutoffs.\n")
print("Prepared pass5: both, DEV64, n16, max8192; active strength1.0")
```
