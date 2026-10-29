This batch reproduced the improvement: 3/64 correct became 36/64, with 33 rescues and no regressions. Its 51.6-point gain is consistent with the preceding batches despite different problems. I will add it to the pooled estimate and prepare the last batch of this reserved split; a substantially weaker result would indicate more variability than seen so far.

```python
# Hypothesis: The frozen edit's paired gain is reproducible across independent batches.
register_frozen_run(55)
```
