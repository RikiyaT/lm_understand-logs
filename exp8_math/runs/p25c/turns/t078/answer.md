The fresh test improved from 18/128 to 34/128 correct: a paired gain of 12.5 percentage points with a 3.84-point standard error. Most of the gain came from grade-school problems; competition-math accuracy was nearly unchanged. This is encouraging independent evidence, but I will finish the preselected validation sets without adjusting the edit.

```python
# Hypothesis: the frozen repetition-head ablation's accuracy gain replicates on independently reserved problems.
first_original = json.loads((work / "gen_30_original.jsonl").open().readline())
first_edited = json.loads((work / "gen_30_edited.jsonl").open().readline())
print("Sample fields:", list(first_original["samples"][0]))
print("Original sample metadata:", {
    k: v for k, v in first_original["samples"][0].items()
    if isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) < 100)
})
print("Edited sample metadata:", {
    k: v for k, v in first_edited["samples"][0].items()
    if isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) < 100)
})

primary_runs = [{"set": "H2A", "generation_file_index": 30}]
(work / "primary_confirmation_runs.json").write_text(json.dumps(primary_runs))
h2a_summary = {
    "n": 128, "original_correct": 18, "edited_correct": 34,
    "gain_pp": 12.5, "SE_pp": 3.84, "wrong_to_right": 21, "right_to_wrong": 5,
    "gsm8k_correct": [10, 25], "math_correct": [8, 9]
}
with (work / "notes.md").open("a") as f:
    f.write("\nPass22/25 gen_30: frozen L15H7=0 on untouched H2A128 succeeded. " +
            json.dumps(h2a_summary) +
            ". Positive primary confirmation, mostly GSM; do not claim established MATH gain.\n")
    f.write("NEXT PASS23/25: unchanged frozen candidate, both H2B128, n1, max_tokens32768. Then H3/H4 256 each regardless of interim result.\n")

next_request = {"type": "generate", "model": "both", "problems": h2b_ids,
                "n": 1, "max_tokens": 32768}
(work / "pass23_request.json").write_text(json.dumps(next_request))
assert torch.equal(load(target_name), frozen_w)
print("FORWARD REQUEST:", json.dumps(next_request))
```
