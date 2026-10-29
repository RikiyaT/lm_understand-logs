The matched control gained 2.15 points with a 2.00-point standard error, comparable to the targeted layer 8 edit. That weakens the claim that the layer 8 group specifically repairs mathematical errors, so I won’t treat it as established. I’ll examine layer 11’s success-associated neurons next, requiring both cross-problem consistency and greater activity on mathematical answer values than on answer formatting. This tests a different hypothesis: useful calculation-related features may be underexpressed.

```python
li11 = mlp_layers.index(11)
acts11 = np.load(work / "fwd_2/mlp_act_11.npy")
value_activity, format_activity = [], []
for i,text in enumerate(compact_texts):
    enc = tok.encode(text,add_special_tokens=False)
    start = text.index("<start_of_turn>model\n")+len("<start_of_turn>model\n")
    val = text.rfind("\\boxed{")+len("\\boxed{")
    vp = [j for j,(a,b) in enumerate(enc.offsets) if b>val and a<len(text)-2]
    fp = [j for j,(a,b) in enumerate(enc.offsets) if b>start and a<val]
    value_activity.append(np.abs(acts11[i,vp]).mean(0))
    format_activity.append(np.abs(acts11[i,fp]).mean(0))
value_ratio11 = (np.mean(value_activity,0)+1e-6)/(np.mean(format_activity,0)+1e-6)
good11 = np.mean([X[(P==p)&Y,li11].mean(0) for p in dev_ids],0)
bad11 = np.mean([X[(P==p)&~Y,li11].mean(0) for p in dev_ids],0)
half_agree11 = (adjusted[split[:16],li11].mean(0)
                * adjusted[split[16:],li11].mean(0) > 0)
valid11 = (half_agree11 & (np.abs(tstat[li11])>2)
           & (np.abs(good11)>np.abs(bad11))
           & (good11*adjusted[:,li11].mean(0)<0)
           & (value_ratio11>1.2))
units11 = np.where(valid11)[0]
units11 = units11[np.argsort(-np.abs(tstat[li11,units11]))][:16].tolist()
print("Success-associated layer-11 units:",[
    {"unit":j,"t":round(float(tstat[li11,j]),2),
     "value_to_format_activity":round(float(value_ratio11[j]),2),
     "correct_mean":round(float(good11[j]),4),
     "failed_mean":round(float(bad11[j]),4)} for j in units11])
(work/"layer11_success_units.json").write_text(json.dumps(units11))
checkpoint("control_layer8", {"candidate":candidate,"result":summarize_pair(9)})
candidate = prepare_units(10,11,units11,-1.0,
    "Double success-associated, answer-value-selective layer-11 neuron outputs; "
    "test underexpressed mathematical features. This replaces, not combines "
    "with, layer-8 edits.")
print(candidate)
"Layer-11 hypothesis prepared from fixed DEV evidence."
```