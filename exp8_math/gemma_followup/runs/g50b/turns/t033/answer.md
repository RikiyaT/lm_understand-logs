Increasing the locally focused head shortened answers but also reduced accuracy, by 2.15 points with a 2.44-point standard error. Together, the two interventions show that response length is not a reliable repair target, so I’m rejecting this head edit. I’ll instead test layer 11 head 2, which showed the strongest retrieval of original operands in the probe. Strengthening that head directly addresses the observed loss of quantities and operators during calculation.

```python
candidate = prepare_head(14,11,2,1.5,
    "Strengthen operand-retrieval candidate: probe original-operand attention "
    "3.26%, versus 0.14% for head 0; low recent-8 focus (3.8%). "
    "Head-0 edits rejected because shorter/longer responses did not improve accuracy.")
print(candidate)
"Operand-retrieval head test prepared."
```