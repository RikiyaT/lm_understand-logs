The broader probe shows some likelihood reduction on productive reasoning as well as on loops, so the edit is not perfectly selective. I will quantify the numeric-token effects rather than assume that all of the loss is harmless wording variation. The final decision still depends on net mathematical accuracy, and the weights remain frozen. I will now retry the untouched paired generation batch after these diagnostic passes.
```python
# Hypothesis: despite some collateral likelihood loss, the frozen edit yields a positive net accuracy effect by reducing stalled copying.
broad_edited=broad_metrics("fwd_26")
broad_deltas=[]
for a,b in zip(broad_base,broad_edited):
    d={"case":a["case"],"kind":a["kind"]}
    for key in ("body_logprob","late_logprob","numeric_logprob","numeric_top1"):
        d[key+"_delta"]=b[key]-a[key]
    broad_deltas.append(d)
print("Frozen-edit broad-probe differences:")
for r in broad_deltas:
    print({k:round(v,5) if isinstance(v,float) else v for k,v in r.items()})
(W/"broad_probe_ablation_deltas.json").write_text(json.dumps(broad_deltas))
req20=dict(req18)
(W/"pass20_request.json").write_text(json.dumps(req20))
assert req20["problems"]==H2A
with (W/"notes.md").open("a") as f:
    f.write("Pass19/25 fwd_26: frozen L15H7=0 broad-probe deltas "+json.dumps(broad_deltas)+
            ". Some productive likelihood loss exists; do NOT claim perfect selectivity or universal preservation.\n")
    f.write("NEXT PASS20/25: retry previously failed, unobserved H2A128 paired generation, n1 max_tokens32768; "
            "exact /work/pass20_request.json. ONLY L15 o_proj columns [1792:2048] zero, all else original. "
            "No outcome from earlier startup failure; primary fresh confirmation remains H2-H4.\n")
print("Recorded unchanged frozen candidate:",current_recipe)
```
