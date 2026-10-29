The 10% increase still hurt accuracy, by 3.08 points with a 1.33-point standard error, and did not reduce cutoffs. That weakens the case for strengthening this head, even at a modest level. I will complete the already specified 25% test rather than introduce another post-hoc target. If it also fails, this head will not be selected merely because its role in repetition is well established.

```python
s20=pair_summary(20)[0]
print("Pass20:",s20)
with (work/"notes.md").open("a") as f:
    f.write("\nPass20 L11H0x1.10: DEV35.81%->32.74%, gain-3.08pp, reported SE1.33pp; cutoffs3.97%->4.07%. Small reinforcement also failed.\n"+json.dumps(s20)+"\n")
hw=load_original(hname).clone()
hw[:,:256]*=1.25
save_edit(hname,hw)
hcurrent={hname:{"head":0,"columns":[0,256],"column_multiplier":1.25,"removed_fraction":-0.25}}
(work/"current_edit.json").write_text(json.dumps(hcurrent))
shutil.copytree(work/"edited",work/"candidate_B_head_1p25",dirs_exist_ok=True)
with (work/"notes.md").open("a") as f:
    f.write("\n## Pass21 planned\nOnly L11 self_attn.o_proj columns0:256 multiplied1.25. DEV63,both,n16,max_tokens8192. Complete predeclared intermediate-strength test; accuracy primary.\n"+json.dumps(hcurrent)+"\n")
print("Pass21 edit:",hcurrent)
```
