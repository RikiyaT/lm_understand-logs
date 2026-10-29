The original full-reference baseline is available. I will now remove only the eight preselected layer-11 MLP output columns, leaving their gates and all attention weights original. This is a strong intervention on a small component set, so a negligible effect will not be attributable to an imperceptibly small edit. The next screen will compare the exact same texts and separate numerical reasoning tokens from general prose.

```python
m22=json.loads((W/"fwd_22"/"meta.json").read_text())
refs_actual=m22["texts"]
print("REFERENCE TRANSCRIPTION DIFFERENCES",
      [i for i,(a,b) in enumerate(zip(refs,refs_actual)) if a!=b])
(W/"DEV_reference_texts_actual22.json").write_text(json.dumps(refs_actual))
for name in list(edited()): drop_edit(name)
mlp11name="model.layers.11.mlp.down_proj.weight"
D11=load_original(mlp11name)
new11=D11.clone(); new11[:,robust11]=0
save_edit(mlp11name,new11)
manifest23={"tensor":mlp11name,"columns":robust11,"multiplier":0.0,
            "relative_F_change":float((new11-D11).norm()/D11.norm())}
with (W/"notes.md").open("a") as f:
    f.write("\nPass 22 full reference baseline complete. Exact submitted text "
            "saved to DEV_reference_texts_actual22.json; one prompt has harmless "
            "math-delimiter transcription difference, held identical in comparison. "
            "Upcoming pass 23: EDITED teacher-forcing same 32 exact pass-22 texts. "
            "Exact edit: "+json.dumps(manifest23)+
            ". All other tensors original. HOLDOUT unused.\n")
print("NEXT EDIT",manifest23)
```