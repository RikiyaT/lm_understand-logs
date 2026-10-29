```python
# Hypothesis: the loop is maintained by repeated amplification across late MLPs, so a single-layer repair leaves most of the feedback circuit intact.
print("Late-MLP circuit edit:",mlp_direction_edit([12,13,14,15,16,17,18,19,20,21,22],1.0))
with open("/work/notes.md","a") as f:
    f.write("Pass5 gen_7: L19 rank-one repetition-direction removal, paired32 +6.25pp +/-4.35pp; 4->6 correct, cutoff90.6->81.25%. Extending same specific directional removal to MLPs12-22 to test a distributed feedback circuit. This does not target think/end tokens; all direction data are within thinking.\n")
```
