The combination gained 3.12 points, but the standard error was 2.38 points and cutoff frequency more than doubled. That suggests the stronger attention output may preserve useful computations while also driving excessive continuation. I will reduce head amplification from 1.2× to 1.1×, keeping the eight-unit MLP edit fixed. Preserving the accuracy gain while reducing the extra cutoffs would support a better balance; shorter answers without an accuracy gain would not.

```python
archive_current("candidate_head120_matched11half",manifest28)
manifest29_head=set_head_direction(alpha=0.0,gain=1.1)
new11=D11.clone(); new11[:,robust11]*=0.5
save_edit(mlp11name,new11)
manifest29={"components":[manifest29_head,manifest26]}
log_pass(29,"Lower attention dose with fixed matched-neuron half-suppression; test accuracy/looping tradeoff",manifest29)
print("PASS28",summarize_pair(28))
print("NEXT EDIT",manifest29)
```