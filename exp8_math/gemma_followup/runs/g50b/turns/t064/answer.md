Doubling the eight units reduced accuracy by only 0.39 points, with a 1.94-point standard error; it also shortened answers. The signs are consistent with suppression helping, but the uncertainty is too large to establish that conclusion. I will test half-suppression as the intermediate dose and then stop exploring this group. A reproducible intermediate benefit would make it competitive; otherwise the attention-based candidate remains the stronger empirical lead.

```python
for name in list(edited()): drop_edit(name)
new11=D11.clone(); new11[:,robust11]*=0.5
save_edit(mlp11name,new11)
manifest26={"tensor":mlp11name,"columns":robust11,"multiplier":0.5,
            "relative_F_change":float((new11-D11).norm()/D11.norm())}
log_pass(26,"Intermediate-dose test for length-matched layer11 failure-associated units",manifest26)
print("PASS25",summarize_pair(25))
print("NEXT EDIT",manifest26)
```