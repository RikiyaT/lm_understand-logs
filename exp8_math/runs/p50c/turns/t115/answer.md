The broader control probe completed with matching text lengths, so the original and edited token scores can be compared directly. I’ll separate mathematical reasoning tokens, numeric tokens, and the final closure token for each text. This will test whether the edit preserves repeated calculations and error correction rather than simply discouraging all repetition. After recording those results, I’ll retry the unchanged candidate on the first reserved validation batch.

```python
# Hypothesis: the targeted edit spares useful iterative mathematics and does not broadly force thinking to close.
meta48 = json.loads(Path("/work/fwd_48/meta.json").read_text())
meta49 = json.loads(Path("/work/fwd_49/meta.json").read_text())
ids48 = np.load("/work/fwd_48/input_ids.npy")
ids49 = np.load("/work/fwd_49/input_ids.npy")
assert np.array_equal(ids48, ids49)
lp48 = np.load("/work/fwd_48/token_logprobs.npy")
lp49 = np.load("/work/fwd_49/token_logprobs.npy")
def encoded_ids(text):
    encoding = tok.encode(text)
    return encoding.ids if hasattr(encoding, "ids") else encoding
productive_controls = []
numeric_deltas = []
for i in range(8):
    text_i = meta48["texts"][i]
    prefix = text_i.split("<think>\n", 1)[0] + "<think>\n"
    start = len(encoded_ids(prefix))
    end = meta48["lengths"][i]
    assert encoded_ids(prefix) == ids48[i, :start].tolist()
    delta = lp49[i, start-1:end-2] - lp48[i, start-1:end-2]
    for pos in range(start, end-1):
        word = tok.decode([int(ids48[i, pos])]).strip()
        if word.isdigit():
            numeric_deltas.append(float(lp49[i, pos-1]-lp48[i, pos-1]))
    productive_controls.append({"case": i, "reasoning_delta": float(delta.mean()),
                                "close_original": float(lp48[i, end-2]),
                                "close_edited": float(lp49[i, end-2])})
early_controls = []
for i in range(8, 12):
    end = meta48["lengths"][i]
    early_controls.append({"case": i, "original": float(lp48[i, end-2]),
                           "edited": float(lp49[i, end-2])})
control_result = {"productive_cases": productive_controls,
                  "mean_productive_delta": float(np.mean([x["reasoning_delta"] for x in productive_controls])),
                  "mean_numeric_delta": float(np.mean(numeric_deltas)),
                  "numeric_tokens": len(numeric_deltas),
                  "premature_closure": early_controls}
print(json.dumps(control_result))
with open("/work/notes.md", "a") as f:
    f.write("\nFresh128x1 and first balanced64x1 generation attempts failed initialization, no validation results. "
            "Pass31/fwd48 ORIGINAL and pass32/fwd49 TOP64alpha6 broader productive-repetition controls: "
            + json.dumps(control_result) + "\nCURRENT top64alpha6 unchanged; validation_3 still unseen.\n")
```
