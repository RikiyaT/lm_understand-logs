Layer 11 head 0 stands out for placing about 55% of its attention on the last eight tokens at the probed arithmetic positions, versus about 0.1% on the original operands. That could contribute to local continuation overpowering the problem constraints, although the probe alone cannot establish that. I’ll halve this head’s output contribution, leaving its attention computation and all other heads unchanged. A paired accuracy gain would justify testing weaker and stronger attenuation.

```python
def prepare_head(pass_number, layer, head, multiplier, rationale):
    clear_edits()
    name = f"model.layers.{layer}.self_attn.o_proj.weight"
    W0 = load_original(name)
    W = W0.clone()
    W[:,head*256:(head+1)*256] *= multiplier
    save_edit(name,W)
    c = {"tensor":name,"query_head":head,
         "columns":[head*256,(head+1)*256],
         "output_multiplier":multiplier,
         "relative_frobenius_change":float((W-W0).norm()/W0.norm())}
    experiments[pass_number] = c
    with (work/"notes.md").open("a") as f:
        f.write(f"\nUpcoming pass {pass_number}: BOTH, DEV_ids.json, n=32, "
                "max_tokens=8192. "+rationale+" Exact edit: "+json.dumps(c)+
                ". All other tensors original. HOLDOUT unused.\n")
    return c

candidate = prepare_head(12,11,0,0.5,
    "Attenuate locally focused full-context head: probe arithmetic-result "
    "queries have 54.6% attention in last 8 tokens, 0.14% original operands. "
    "Hypothesis: excessive local continuation contributes to operation drift.")
print(candidate)
"Single-head attenuation ready."
```