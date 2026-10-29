The larger holdout improved from 29/256 to 58/256 correct, a paired gain of 11.33 points with a 2.59-point standard error. Unlike the earlier batch, this one improved both grade-school and competition-math accuracy. I will run the final preselected holdout and base the decision on all 768 fresh pairs, including the batch with no gain.

```python
# Hypothesis: the frozen edit's benefit remains positive across the complete preselected confirmation sample.
h3_rows = read_paired_generation(32)
assert {r["id"] for r in h3_rows} == set(holdout_info["H3"])
primary_rows = read_paired_generation(30) + read_paired_generation(31) + h3_rows
pooled_512 = {"all": summarize_pairs(primary_rows)}
for source in ("gsm8k", "math"):
    pooled_512[source] = summarize_pairs([r for r in primary_rows if r["source"] == source])
print("Fresh confirmation through 512 problems:", json.dumps(pooled_512))
(work / "primary_512_results.json").write_text(json.dumps(pooled_512, indent=2))
primary_runs.append({"set": "H3", "generation_file_index": 32})
(work / "primary_confirmation_runs.json").write_text(json.dumps(primary_runs))
with (work / "notes.md").open("a") as f:
    f.write("\nPass24/25 gen_32 H3 256:29->58 correct,38 improvements/9 regressions,paired gain11.328125pp,SE2.59pp. GSM21->39/128;MATH8->19/128. Fresh pooled512 " + json.dumps(pooled_512) + "\n")
    f.write("NEXT FINAL PASS25/25: both H4 256, n1,max_tokens32768. Exact frozen L15H7=0 remains unchanged. Final decision uses all768 H2-H4 pairs, H1 separately supportive. No outcome-driven strength selection.\n")

h4_request = {"type": "generate", "model": "both", "problems": holdout_info["H4"],
              "n": 1, "max_tokens": 32768}
(work / "pass25_request.json").write_text(json.dumps(h4_request))
assert set(h4_request["problems"]).isdisjoint(r["id"] for r in primary_rows)
assert torch.equal(load(target_name), frozen_w)
print("FORWARD REQUEST:", json.dumps(h4_request))
```
