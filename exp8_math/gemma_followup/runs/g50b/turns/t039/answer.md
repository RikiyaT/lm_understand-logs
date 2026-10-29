A 20% head increase gained 1.66 points with a 1.56-point standard error, despite producing longer answers and more cut-offs. That is compatible with improved retrieval offset by increased replay, though the accuracy gain remains uncertain. I’ll test a targeted combination: this modest head increase plus half suppression of the layer 8 group that reduced repetition. The combination must outperform on accuracy to justify keeping both components.

```python
checkpoint("head11_2_120",{"candidate":candidate,"result":summarize_pair(16)})
head_component = prepare_head(17,11,2,1.2,
    "Combination test: strengthen essential retrieval head while damping "
    "the previously identified layer-8 repetition-associated group.")
mlp_component = set_unit_edit(8,units8,0.5)
candidate = {"components":[head_component,mlp_component]}
experiments[17] = candidate
with (work/"notes.md").open("a") as f:
    f.write("PASS 17 EXACT FINAL EDIT OVERRIDES preceding single-component "
            "description: "+json.dumps(candidate)+
            ". Both on locked DEV, n=32, max_tokens=8192; all other weights original.\n")
print(candidate)
"Two-component causal combination ready."
```