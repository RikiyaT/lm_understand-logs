The full set of 64 DEV reference solutions is small enough to submit explicitly, so I’ll use all of them as local mathematical-continuation probes. I’ll also include eight short excerpts from repetitive failing DEV answers to test whether candidate neurons distinguish useful calculations from loops. These reference-only probes are not evaluation-format answer scores; paired generation will remain the accuracy test. The expected screening result is preservation of reference calculations while reducing support for repetitive continuations.

```python
wrong_rank = sorted(
    [(float(a1594[i]), s) for i, s in enumerate(dev_flat) if not s["correct"]],
    key=lambda x: x[0], reverse=True)
probe_failures = []
used_probe_pids = set()
for activity, s in wrong_rank:
    if s["problem_id"] in used_probe_pids:
        continue
    used_probe_pids.add(s["problem_id"])
    probe_failures.append({"problem_id": s["problem_id"], "sample_idx": s["sample_idx"],
                           "activity": activity, "text": s["text"][-280:]})
    if len(probe_failures) == 8:
        break
literal_probes = reference_only + [r["text"] for r in probe_failures]
(wdir / "literal_probes.json").write_text(json.dumps(literal_probes))
(wdir / "literal_probe_metadata.json").write_text(json.dumps(
    [{"problem_id": pid, "kind": "reference"} for pid in dev_ids] +
    [dict(r, kind="repetitive_failure") for r in probe_failures], indent=2))
with open(wdir / "notes.md", "a") as f:
    f.write("\n## Pass 3 planned: literal teacher-forced localization\n"
            "Exact edit: none; strength 0. Set: fixed DEV only.\n"
            "72 raw local-continuation probes: all 64 full reference solutions plus eight "
            "280-character excerpts from repetitive failures, selected by high L14 neuron1594 activity.\n"
            "No problem prompts: these measure local continuation effects, NOT answer accuracy or "
            "evaluation-format answer likelihood. literal_probes.json records exact texts.\n"
            "Return token logprobs and MLP activations at layers 8,14,17.\n")
print("Literal probe count:", len(literal_probes), "tokens:",
      sum(len(encode_ids(s)) for s in literal_probes))
print("Reference texts 0–23:")
print(json.dumps(literal_probes[:24], ensure_ascii=True))
```
