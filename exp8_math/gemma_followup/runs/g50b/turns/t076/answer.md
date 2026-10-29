Head 2 at 1.1× alone gained only 0.59 points, so it did not reproduce the combination’s larger improvement. This is consistent with the MLP edit contributing, but the comparison is still noisy. I will now halve the eight matched low-association control units while retaining the same head adjustment. A much smaller gain than the selected group would support specificity; a similar gain would suggest that a more general local perturbation is responsible.

```python
manifest32_head=set_head_direction(alpha=0.0,gain=1.1)
control_new=D11.clone(); control_new[:,control11]*=0.5
save_edit(mlp11name,control_new)
manifest32={"components":[manifest32_head,
    {"tensor":mlp11name,"columns":control11,"multiplier":0.5,
     "relative_F_change":float((control_new-D11).norm()/D11.norm())}]}
log_pass(32,"Specificity control: head2 1.1x plus matched low-association MLP half-suppression",manifest32)
print("PASS31",summarize_pair(31))
print("NEXT EDIT",manifest32)
```