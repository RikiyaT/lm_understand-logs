Doubling neuron 920 produced a small negative estimate: −1.29 percentage points, with a 1.46-point standard error. Thus the numerical-feature analysis has not yet translated into a demonstrated accuracy gain. I will test complete removal of the same neuron to establish whether its normal contribution is helpful or excessive. A substantial loss would support its usefulness but reject amplification; an improvement would instead support attenuation.

```python
s18=pair_summary(18)[0]
print("Pass18:",s18)
with (work/"notes.md").open("a") as f:
    f.write("\nPass18 L17 neuron920x2: DEV36.01%->34.72%, gain-1.29pp, reported SE1.46pp; cutoffs4.27%->5.16%. No demonstrated benefit.\n"+json.dumps(s18)+"\n")
fw=load_original(fname).clone()
fw[:,920]=0
save_edit(fname,fw)
fcurrent={fname:{"columns":[920],"column_multiplier":0.0,"removed_fraction":1.0}}
(work/"current_edit.json").write_text(json.dumps(fcurrent))
shutil.copytree(work/"edited",work/"candidate_F_neuron920_removed",dirs_exist_ok=True)
with (work/"notes.md").open("a") as f:
    f.write("\n## Pass19 planned\nOnly L17 MLP down_proj column920 zeroed. DEV63,both,n16,max_tokens8192. Opposite-strength causal test of the same numerically selective component.\n"+json.dumps(fcurrent)+"\n")
print("Pass19 edit saved:",fcurrent)
```
