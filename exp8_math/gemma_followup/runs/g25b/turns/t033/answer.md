Full removal strongly hurt: accuracy fell by 15.97 percentage points, and cutoffs rose from 4.2% to 28.6%. This reverses the initial interpretation—the head appears to help prevent runaway continuation, not cause it. I’ll test strengthening this same head by 50%, with no other edits. The prediction is fewer repetitive failures and higher accuracy; fewer cutoffs alone will not count as success.

```python
res10,byp10=pair_summary(10)
print("Head removal behavior:",res10)
shutil.copytree("/work/edited","/work/candidate_B_head_removed",dirs_exist_ok=True)
edit_desc=apply_head_edit(11,0,-0.5)
with open("/work/notes.md","a") as f:
    f.write("\nPass10 L11H0 ablation: reported gain-15.97pp,SE2.75pp; cutoffs4.17%->28.57%. "+json.dumps(res10)+"\n"
            "Causal revision: head is protective against runaway continuation, not a harmful copying head. One controlled TF probe did not predict broad behavior.\n"
            "## Pass11 planned\nOnly L11 self_attn.o_proj columns0:256 multiplied by1.5 (head0 removal=-0.5); all other weights original. DEV63,both,n16,max_tokens8192. Predict reduced degeneracy AND improved accuracy; accuracy is primary.\n"
            +json.dumps(edit_desc)+"\n")
print("Current edit:",edit_desc)
```
