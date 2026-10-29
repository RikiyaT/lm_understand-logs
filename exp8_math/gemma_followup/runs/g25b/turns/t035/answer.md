Strengthening head 0 halved cutoffs but reduced accuracy by 4.27 percentage points, with a 1.96-point standard error. That directly shows why better termination is not enough to improve mathematics. I’ll leave head 0 at its original strength and test layer 11 head 2, which retrieved numeric intermediate values much more strongly in the probes. The new hypothesis is that strengthening content retrieval, rather than termination control, may help the model use its calculations correctly.

```python
res11,byp11=pair_summary(11)
print("Head0 amplification behavior:",res11)
shutil.copytree("/work/edited","/work/candidate_B_head_1p5",dirs_exist_ok=True)
edit_desc=apply_head_edit(11,2,-0.5)
with open("/work/notes.md","a") as f:
    f.write("\nPass11 L11H0x1.5: reported gain-4.27pp,SE1.96pp; cutoffs4.17%->2.08%. "+json.dumps(res11)+"\n"
            "Conclusion: head0 causally controls repetitive continuation but boosting it does not repair math accuracy. Reject fewer cutoffs as success.\n"
            "## Pass12 planned\nCandidate C: restore head0; only L11 self_attn.o_proj columns512:768 (head2) multiplied by1.5. Head2 is strongest numeric-intermediate retriever in probes (previous90 mass .046 correct/.140 wrong, versus head0 .001/.005); occurrence-count confound acknowledged. Test whether stronger content retrieval improves computation-to-answer use. DEV63,both,n16,max_tokens8192.\n"
            +json.dumps(edit_desc)+"\n")
print("Current edit:",edit_desc)
```
