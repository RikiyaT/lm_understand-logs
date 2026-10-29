Block 4 also tied exactly, so the last two passes changed individual answers without changing total accuracy. I will summarize the three complete rounds and begin the final precommitted round, with no additional tuning. This preserves an independent confirmation test of the DEV-selected edit. The final result must distinguish a possible small accuracy benefit from the clearly observed increase in generation cost.

```python
prepare_frozen_holdout_pass(47)
print("BALANCED_THREE_ROUNDS", pooled_pairs(list(range(35, 47))))
three_round_behavior = holdout_behavior_summary(list(range(35, 47)))
print("THREE_ROUND_BEHAVIOR", {
    m: {k: round(float(v), 3) if isinstance(v, (float, np.floating)) else v
        for k, v in s.items()}
    for m, s in three_round_behavior.items()
})
"Pass47 logged; final round uses the precommitted twenty-four-position rotations."
```