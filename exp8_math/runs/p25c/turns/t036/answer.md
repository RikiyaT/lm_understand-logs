Layer 15, head 7 is the clearest copying candidate: it assigns 88.7% of attention to continuations of earlier matching passages during the loop, versus 8.2% during productive checks. Layer 11, head 1 and layer 23, head 5 show similar but weaker patterns. I will first remove only head 7’s contribution through its output-projection columns, leaving every other head unchanged. The prediction is a selective reduction in repetitive continuation—not a general push to end thinking—and I will test both token likelihoods and paired problem-solving accuracy.
```python
# Hypothesis: removing layer 15 head 7's output weakens self-copying feedback without suppressing useful mathematical reasoning.
def clear_edits():
    for name in list(edited()):
        drop_edit(name)
def install_head_scales(spec):
    clear_edits()
    for layer in sorted(set(x[0] for x in spec)):
        name=f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
        w=load_original(name)
        for l,h,scale in spec:
            if l==layer:
                w[:,h*256:(h+1)*256] *= scale
        save_edit(name,w)
    return [{"tensor":f"model.language_model.layers.{l}.self_attn.o_proj.weight",
             "columns":[h*256,(h+1)*256],"head":h,"multiplier":scale}
            for l,h,scale in spec]
current_recipe=install_head_scales([(15,7,0.0)])
(W/"current_recipe.json").write_text(json.dumps(current_recipe))
dev_ids=[r["id"] for r in gen_data["gen_5_original"]]
(W/"dev_ids.json").write_text(json.dumps(dev_ids))
req8={"type":"teacher_forced","model":"edited","texts":probe_texts,
      "return":{"token_logprobs":True,"topk":20,"hidden_states":[16,24],"mlp_act":[23]}}
(W/"pass08_request.json").write_text(json.dumps(req8))
with (W/"notes.md").open("a") as f:
    f.write("Clean probe localization: L15H7 previous-matching-continuation attention 0.8871 loop vs 0.0824 "
            "productive; L11H1 0.7430 vs 0.1388; L23H5 0.6169 vs 0.0344. Correlational candidates only.\n")
    f.write("NEXT PASS 8/25: teacher-forced edited raw probe, exact texts /work/pass08_request.json. "
            "Only edit: model.language_model.layers.15.self_attn.o_proj.weight columns [1792:2048] "
            "multiplied by 0 (ablate head 7); all other values original. No final norm, embedding, "
            "output-layer or thinking-token intervention. Follow with paired generation on dev64 "
            "in /work/dev_ids.json, not held-out confirmation.\n")
print("Installed:",current_recipe)
print("Development problem IDs:",json.dumps(dev_ids))
```
