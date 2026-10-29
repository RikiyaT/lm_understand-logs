Head 2 amplification reduced accuracy by 6.35 percentage points and raised cutoffs from 4.1% to 19.8%. Unlike head 0, strengthening this head promotes runaway continuation, which is consistent with over-retrieval of already repeated content. I’ll now test 50% attenuation of head 2 while leaving head 0 untouched. Higher accuracy alongside fewer repetitive failures would support a balance between these two functions rather than “more attention” being universally better.

```python
res12,byp12=pair_summary(12)
print("Head2 amplification behavior:",res12)
shutil.copytree("/work/edited","/work/candidate_C_head2_1p5",dirs_exist_ok=True)
edit_desc=apply_head_edit(11,2,0.5)
with open("/work/notes.md","a") as f:
    f.write("\nPass12 L11H2x1.5: reported gain-6.35pp,SE2.20pp; cutoffs4.07%->19.84%. "+json.dumps(res12)+"\n"
            "Causal finding: amplifying head2 promotes runaway continuation, opposite head0. Revise hypothesis toward excessive retrieval/reinforcement of repeated intermediates.\n"
            "## Pass13 planned\nOnly L11 self_attn.o_proj columns512:768 multiplied by0.5 (head2 removal0.5); all other weights original. DEV63,both,n16,max_tokens8192. Predict reduced repetition and improved accuracy; reject if only shortening improves.\n"
            +json.dumps(edit_desc)+"\n")
print("Current edit:",edit_desc)
```
