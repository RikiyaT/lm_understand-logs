The replication added only 0.59 points, so half-suppression remains a modest and uncertain effect rather than a clear repair. I will test one mechanistically distinct combination: the same eight-unit half-suppression plus the previously tested 1.2× layer-11 head-2 output. This asks whether moderating the MLP contribution complements stronger context use, without reviving the length-confounded layer-8 group. A gain larger and more repeatable than either component alone would support the combination; otherwise I will prefer the simpler edit.

```python
manifest28_head=set_head_direction(alpha=0.0,gain=1.2)
new11=D11.clone(); new11[:,robust11]*=0.5
save_edit(mlp11name,new11)
manifest28={"components":[manifest28_head,manifest26]}
log_pass(28,"Combination of modest head2 enhancement and matched layer11 unit half-suppression",manifest28)
print("L11 HALF POOLED",pooled_pairs([26,27]))
print("NEXT EDIT",manifest28)
```