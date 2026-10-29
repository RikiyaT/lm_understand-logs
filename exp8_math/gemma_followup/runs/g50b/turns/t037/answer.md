Halving head 2 caused a large accuracy loss: −26.17 points with a 3.73-point standard error. This head is therefore important for solving these problems, even though excessive amplification promotes replay. I’ll test a modest 20% increase rather than suppressing it. The emerging hypothesis is a trade-off between useful mathematical retrieval and excessive repetition, with the original scale potentially close to the optimum.

```python
candidate = prepare_head(16,11,2,1.2,
    "Dose-response localization: 0.5x caused -26.17 pp, 1.5x caused -2.83 pp "
    "with equation replay. Test modest 1.2x strengthening of essential head.")
print(candidate)
"Intermediate head strength prepared."
```