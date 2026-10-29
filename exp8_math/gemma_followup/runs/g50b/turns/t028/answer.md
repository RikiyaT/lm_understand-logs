The matched traces scored differently, as expected, and the tool returned attention for all twelve full-context heads. I’ll summarize attention at the arithmetic-result positions, separating original operands, generated numbers, structural tokens, and local context. This is only a localization probe: ordinary attention to intermediate results can be useful, so a head will be considered harmful only if a controlled intervention improves accuracy.

```python
probe_meta = json.loads((work/"fwd_11/meta.json").read_text())
attn_summary = []
for layer in (5,11,17):
    attn = np.load(work/f"fwd_11/attn_{layer}.npy")
    for head in range(4):
        records = []
        for i,text in enumerate(attention_texts):
            enc = tok.encode(text,add_special_tokens=False)
            assert len(enc.ids) == probe_meta["lengths"][i]
            ans_start = text.index("<start_of_turn>model\n")+len("<start_of_turn>model\n")
            result_strings = (["=9800","=196$","=9996"] if i==0 else
                              ["=19600","=196$","=20560"])
            numeric = [(m.start(),m.end()) for m in re.finditer(r"\d+",text)]
            number_mask = np.array([any(a<d and b>c for c,d in numeric)
                                    for a,b in enc.offsets])
            original_mask = number_mask & np.array([
                b <= text.index("\nPlease reason") for a,b in enc.offsets])
            generated_mask = number_mask & np.array([a>=ans_start for a,b in enc.offsets])
            for rs in result_strings:
                start = text.index(rs,ans_start)+1
                q = max(j for j,(a,b) in enumerate(enc.offsets) if b<=start and b>0)
                weights = attn[i,head,q,:len(enc.ids)]
                records.append({
                    "original_operands":float(weights[original_mask].sum()),
                    "generated_numbers":float(weights[generated_mask].sum()),
                    "bos":float(weights[0]),
                    "recent_8":float(weights[max(0,q-7):q+1].sum()),
                    "entropy":float(-(weights*np.log(weights+1e-12)).sum())
                })
        summary = {"layer":layer,"head":head}
        summary.update({k:round(np.mean([r[k] for r in records]),4)
                        for k in records[0]})
        W = load_original(f"model.layers.{layer}.self_attn.o_proj.weight")
        summary["output_block_norm"] = round(float(
            W[:,head*256:(head+1)*256].norm()),3)
        attn_summary.append(summary)
print(json.dumps(attn_summary,indent=2))
"Attention localization summarized; no attention edit yet."
```