Half-strength attenuation lost 1.88 percentage points, with a 1.94-point standard error—much less than full ablation, though still not a demonstrated improvement. This is consistent with useful computation being weakened as attenuation increases. I’ll now reverse the intervention and multiply only those selected columns by 1.5. An accuracy gain would support an underpowered-computation hypothesis; another loss would suggest their original strength is already near a useful balance.

```python
res6,byp6=pair_summary(6)
shutil.copytree("/work/edited","/work/candidate_A_half",dirs_exist_ok=True)
edit_desc=apply_neuron_edit(-0.5)
json.dump(edit_desc,open("/work/current_edit.json","w"))
with open("/work/notes.md","a") as f:
    f.write("\nPass6 half attenuation: reported gain -1.88pp, SE1.94pp. "+json.dumps(res6)+"\n"
            "## Pass7 planned\nReverse-direction causal test: same selected96 columns in layers4/15/17 down_proj multiplied by1.5 (removal=-0.5). DEV63,both,n16,max_tokens8192. Hypothesis: components are useful but insufficiently recruited, not harmful.\n"
            +json.dumps(edit_desc)+"\")
print("Current edit: 96 selected MLP output columns multiplied by 1.5; no other changes.")
```
