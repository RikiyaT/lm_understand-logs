There were no DEV cases where an earlier correct boxed answer was replaced by a wrong final box, so I won’t pursue that hypothesis. Repetition-associated neurons also have weak accuracy associations, which cautions against treating repetition reduction as the goal. A few early-layer neurons carry much larger contributions and show consistent, though modest, failure associations across both DEV halves. I’ll use teacher forcing on DEV trajectories and reference solutions to see what those neurons actually respond to, especially whether they distinguish unproductive copying from arithmetic progress. This remains localization, not evidence that suppressing them will help.

```python
PREFIX = "<bos><start_of_turn>user\n"
SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
    "<end_of_turn>\n<start_of_turn>model\n"
)

teacher_sources = []
for i in dev_indices:
    r = records[int(i)]
    teacher_sources.append({
        "kind": "generated", "record_index": int(i), "problem_id": r["problem_id"],
        "correct": bool(r["correct"]),
        "prompt": PREFIX + practice[r["problem_id"]]["problem"] + SUFFIX,
        "answer": r["text"],
    })
for pid in dev_ids:
    teacher_sources.append({
        "kind": "reference", "record_index": None, "problem_id": pid,
        "correct": True, "prompt": PREFIX + practice[pid]["problem"] + SUFFIX,
        "answer": practice[pid]["solution"],
    })

encoded_sources = [
    tok.encode(s["prompt"] + s["answer"], add_special_tokens=False)
    for s in teacher_sources
]
cap = 2048
while sum(min(len(e.ids), cap) for e in encoded_sources) > 520000:
    cap -= 64

teacher_texts, teacher_manifest = [], []
for source, enc in zip(teacher_sources, encoded_sources):
    raw = source["prompt"] + source["answer"]
    if len(enc.ids) > cap:
        raw = raw[:enc.offsets[cap-1][1]]
    actual = tok.encode(raw, add_special_tokens=False)
    assert len(actual.ids) <= cap
    teacher_texts.append(raw)
    teacher_manifest.append({
        k: v for k,v in source.items() if k not in ["prompt", "answer"]
    } | {
        "prompt_chars": len(source["prompt"]),
        "prompt_tokens": len(tok.encode(source["prompt"], add_special_tokens=False).ids),
        "tokens": len(actual.ids),
        "truncated": len(enc.ids) > cap,
    })

teacher_return = {
    "token_logprobs": True,
    "topk": 8,
    "hidden_states": [0,1,6,14,18],
    "mlp_act": [0,1,5,13,17],
}
teacher_request = {
    "type": "teacher_forced", "model": "original",
    "texts": teacher_texts, "return": teacher_return,
}
with open("/work/dev_teacher_texts.json", "w") as f:
    json.dump(teacher_texts, f)
with open("/work/dev_teacher_manifest.json", "w") as f:
    json.dump(teacher_manifest, f)
with open("/work/dev_teacher_request.json", "w") as f:
    json.dump(teacher_request, f)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Baseline localization findings\n"
        "- 337/2048 correct overall. DEV: 217/512 correct.\n"
        "- Wrong quantity binding, invalid arithmetic/algebra, and repeated "
        "non-progressing equalities occur in inspected failures.\n"
        "- No DEV correct-box-to-wrong-box overwrite cases found.\n"
        "- Pure repetition correlations are not strong accuracy evidence.\n"
        "- Length-controlled, within-problem failure correlations are modest.\n"
        "- Early MLP neurons 1405,149,1309,2032 and others in layer 0 "
        "have stable associations across disjoint DEV halves; not yet causal.\n"
        "\n## Pass 2: original teacher-forced localization\n"
        "- Exact edit: none; original weights.\n"
        "- Dataset: all 512 DEV sampled responses plus 32 DEV reference "
        "solutions, truncated only for the forward token budget.\n"
        "- Full request: dev_teacher_request.json; text list: dev_teacher_texts.json; "
        "alignment: dev_teacher_manifest.json.\n"
        f"- Texts: {len(teacher_texts)}; tokens: "
        f"{sum(m['tokens'] for m in teacher_manifest)}; cap: {cap}.\n"
        "- Examine token-local arithmetic progress versus copied/repeated "
        "expressions and component specificity before selecting an edit.\n"
    )
print("Teacher-forced texts:", len(teacher_texts))
print("Total tokens:", sum(m["tokens"] for m in teacher_manifest))
print("Maximum tokens:", max(m["tokens"] for m in teacher_manifest))
print("Truncated trajectories:", sum(m["truncated"] for m in teacher_manifest))
print("Request saved to /work/dev_teacher_request.json.")
print("Weights unchanged.")
```
