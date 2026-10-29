```python
# Hypothesis: four repetition-amplifying MLP neurons form a small feedback circuit; removing their output should interrupt loops without suppressing thinking generally.
print(Path("/work/notes.md").read_text()[-4500:] if Path("/work/notes.md").exists() else "No previous notes.")
targets = {18:[8], 19:[11], 23:[22,40]}
def apply_circuit_strength(strength):
    for name in list(edited()):
        drop_edit(name)
    changes=[]
    for l,js in targets.items():
        name=f"model.language_model.layers.{l}.mlp.down_proj.weight"
        w=load_original(name)
        new=w.clone()
        new[:,js]*=(1-strength)
        save_edit(name,new)
        changes.append((l,js,float((new-w).norm()/w.norm())))
    return changes
print("Ablation strength 1.0:",apply_circuit_strength(1.0))
# Development set is fixed before observing intervention outcomes: first 32 GSM8K and first 64 MATH diagnostic items.
dev_ids=diag_ids[:32]+diag_ids[96:160]
print("Development IDs:",json.dumps(dev_ids))
note = """
## Diagnosis and first causal test
Baseline: 256 original generations, accuracy 9.375%, 232/256 cut at 32768 tokens; GSM 12.5%, MATH 7.5%.
Failed answers' last-4096-token repeated 12-gram fraction = 0.975; correct answers = 0.177.
213/232 failures repeat 'Wait,' in their tails. Many have already solved the arithmetic before looping.
Teacher-forced pass (files fwd_3; first failed attention request was not counted) used repeated identical sentences.
Within-text repetition grows in the same direction as failure-vs-success activation contrast at neurons
18:8, 19:11, 23:22, 23:40 across all five repeated contents. These are not selected merely by grades.
Caveat: chat rendering inserted an extra empty think pair; subsequent dense probes should use raw texts.
Current experimental edit: zero only down_proj columns 18:8, 19:11, 23:22 and 23:40.
This is a targeted causal test, not yet a confirmed best edit. No protected tensors changed.
Development set: first 32 GSM + first 64 MATH from diagnostic sample (96 problems).
Budget: 2 successful forward passes consumed; 8 remain.
"""
with open("/work/notes.md","a") as f: f.write(note)
```
