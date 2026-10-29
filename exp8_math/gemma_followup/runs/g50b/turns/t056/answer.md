Eight layer-11 units retain a positive failure association in at least 20 of the 32 length-matched problem contrasts. Their strongest observed activations are not confined to arithmetic results, so I will not label them arithmetic-error neurons yet. I will use all full DEV reference solutions, rather than only boxed answers, as the next likelihood screen. Preserving correct reasoning while reducing failures would support suppression; degrading reference reasoning would be a warning against it.

```python
print("FULL REFERENCE JSON PART2",json.dumps(refs[16:],ensure_ascii=True))
planned11={"tensor":"model.layers.11.mlp.down_proj.weight",
           "columns":robust11,"multiplier":0.0}
(W/"planned_matched11_edit.json").write_text(json.dumps(planned11,indent=2))
print("Planned intervention recorded; pass 22 is ORIGINAL, with no edit applied to its forward pass.")
```