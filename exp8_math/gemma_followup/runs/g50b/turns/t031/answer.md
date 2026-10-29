Halving this head did not help: accuracy fell by 1.66 points, and answers became substantially longer with more cut-offs. That contradicts the hypothesis that its local focus drives the repetition; the head may instead help maintain a stable local calculation. I’ll test the opposite intervention, increasing its contribution by 50%, to see whether strengthening that function improves accuracy.

```python
candidate = prepare_head(13,11,0,1.5,
    "Opposite-direction test after attenuation increased length/cutoffs and "
    "reduced accuracy: strengthen potentially useful local calculation head.")
print(candidate)
"Head amplification ready."
```