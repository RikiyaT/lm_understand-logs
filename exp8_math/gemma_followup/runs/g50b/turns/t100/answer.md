Round 2 ended with another small positive block result, bringing the balanced comparison to a modest gain rather than the near-zero first-round result. The increased cutoff rate persists, so any benefit must be judged against that cost. I will prepare round 3 and recover the saved neuron-selection metadata for the final mechanistic account. This is documentation only: neither the selected neurons nor their strength will change.

```python
prepare_frozen_holdout_pass(43)
print("BALANCED_TWO_ROUNDS", pooled_pairs(list(range(35, 43))))
print("ROUND2_ONLY", pooled_pairs(list(range(39, 43))))
unit_metadata_path = work / "layer11_matched_failure_units.json"
if unit_metadata_path.exists():
    print("SAVED_NEURON_SELECTION_METADATA",
          unit_metadata_path.read_text()[:2200])
else:
    print("Neuron selection is documented in notes.md; metadata file not present.")
"Pass43 logged; round3 uses the precommitted sixteen-position rotations."
```