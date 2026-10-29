```python
# Hypothesis: attenuating the strongest matched loop neuron will weaken repetitive continuation without globally forcing thinking to end.
candidate_name="model.language_model.layers.23.mlp.down_proj.weight"
candidate_original=weights[candidate_name]
w=candidate_original.clone()
w[:,22]=0
save_edit(candidate_name,w)
print("Edited only layer 23 down-projection column 22.")
print("Relative Frobenius change:",float((w-candidate_original).norm()/candidate_original.norm()))
print("Edits:",edited())
print("Probe encoding consistency:")
for i in range(2):
    encoded=tok.encode(tfmeta["texts"][i]).ids
    observed=ids2[i,:len(encoded)].tolist()
    print(i,"metadata encoded length",len(encoded),"same IDs",encoded==observed,
          "first divergence",next((j for j,(a,b) in enumerate(zip(encoded,observed)) if a!=b),None))
notes = """# Math weight-edit experiment
Baseline: 128 practice problems, 2 samples each; accuracy 27/256=10.55%, cutoff 229/256=89.84%, mean 29994 tokens. All observed failures were cutoffs. Texts commonly solve the arithmetic correctly then enter hundreds of identical self-checking lines.
Localization: 13 same-problem success/failure pairs. L23 MLP neuron22 has mean activation 1.773 cutoff versus 0.135 correct; paired delta 1.516, positive in all 13 pairs, delta/SE 4.59. L23:94, L22:59, L17:20, L15:0 are additional candidates, not established causal.
Controlled repeated-continuation probe: L23:22 rises from -1.231 during shared valid reasoning to 2.899 during repeated self-checking; L23:94 from -0.083 to 2.566. These remain correlational.
Trial A: set only L23 down_proj column22 to zero. No final norm, embeddings, output layer, decoding, or template edits. No accepted edit yet.
Forward passes used: 2/25 (one failed attention request did not count).
"""
with open("/work/notes.md","w") as f:
    f.write(notes)
```
