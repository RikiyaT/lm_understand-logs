The retry succeeded: accuracy increased from 12.5% to 53.1%, with 27 rescues and one regression. This is a somewhat smaller gain than earlier batches, but still far larger than its 6.6-point standard error. I will record it without changing the edit and continue through the locked final split; the pooled estimate should reflect this variability rather than only the strongest batches.

```python
# Hypothesis: The frozen edit retains a substantial net benefit despite ordinary batch-to-batch variation.
register_frozen_run(58)
with open("/work/notes.md", "a") as f:
    f.write("\nBefore gen58, one identical generation request failed engine initialization; "
            "no counted pass or evaluable outputs. Retry succeeded without changes.\n")
```
