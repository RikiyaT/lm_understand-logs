The direction distinguishes natural failed generations from correct ones well, with AUCs of 0.89–0.95, although answer length remains a possible confound. Its word associations do not identify a simple “Wait” bias, so I will test the distributed state rather than suppress particular words. I will remove this one protected direction from attention and MLP output projections in layers 8–22, leaving the final layer and all forbidden tensors unchanged. The hypothesis predicts fewer loops and higher accuracy; merely shorter answers will not qualify as success.

```python
# Hypothesis: preventing late-layer reinforcement of the matched-content repetition direction will interrupt loops while preserving ordinary reasoning.
rep_edit_cache={}
for l in range(8,23):
    u=torch.from_numpy(rep_protected[l+1].copy()).float()
    for suffix in ["mlp.down_proj.weight",
                   "self_attn.o_proj.weight" if l%4==3 else "linear_attn.out_proj.weight"]:
        name=f"model.language_model.layers.{l}.{suffix}"
        w0=load_original(name)
        component=torch.outer(u,u@w0)
        rep_edit_cache[name]=(w0,component)
def apply_repetition_strength(strength):
    for name in list(edited()): drop_edit(name)
    ratios=[]
    for name,(w0,component) in rep_edit_cache.items():
        save_edit(name,w0-strength*component)
        ratios.append(float(abs(strength)*component.norm()/w0.norm()))
    return dict(strength=strength,tensors=len(ratios),relative_delta_min=float(min(ratios)),
                relative_delta_median=float(np.median(ratios)),relative_delta_max=float(max(ratios)))
print("Rank-one direction edit:",apply_repetition_strength(1.0))
np.savez("/work/repetition_directions.npz",**{str(l):u for l,u in rep_protected.items()})
with open("/work/notes.md","a") as f:
    f.write("\nProtected repetition direction: matched repeated-minus-first content, orthogonalized against first self-check, ordinary thinking, prompt, and answer-transition means. Natural failure AUC .886-.952 (observational/length-confounded). Test rank-one erasure W'=W-u(u^T W) on attention and MLP writers in layers8..22, using post-layer direction. 30 tensors, no layer23 or forbidden tensor changes. Strength1 first; accuracy and repetition determine retention. Four passes remain.\n")
```
