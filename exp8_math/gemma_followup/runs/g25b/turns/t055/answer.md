The token-level screen rejects neuron 1807 as a numerical specialist: it is more active around answer formatting than numerical content. Two other final-layer neurons, 920 and 1062, are more active before numbers and inside reference answers than before prose. I will check whether their outgoing directions distinguish the correct digit from other digits, rather than merely encouraging numeric text. This is a static weight-alignment analysis, not a simulated forward pass or a training update.

```python
# Compare fixed outgoing weight directions with digit embedding contrasts.
# This does not compute model logits or run any network layer.
embed_name = next(n for n in NAMES if n.endswith("embed_tokens.weight"))
emb = load_original(embed_name)
digit_ids = {}
for ch in "0123456789":
    enc = tok.encode(ch, add_special_tokens=False).ids
    if len(enc)==1:
        digit_ids[ch]=enc[0]
assert len(digit_ids)==10
digit_mean = emb[list(digit_ids.values())].mean(dim=0)
numerical_answer_positions=[]
for i,text in enumerate(tftexts):
    enc = tok.encode(text, add_special_tokens=False)
    b=text.rfind("\\boxed{")
    start=b+len("\\boxed{")
    end=text.rfind("}")
    for j in range(1,len(enc.ids)):
        ch=tok.decode([enc.ids[j]]).strip()
        s0,s1=enc.offsets[j]
        if ch in digit_ids and len(ch)==1 and s1>start and s0<end:
            numerical_answer_positions.append((i,j-1,digit_ids[ch]))

last_act = np.load(work/"fwd_3"/"mlp_act_17.npy",mmap_mode="r")
dw = load_original("model.layers.17.mlp.down_proj.weight")
pg = 1+load_original("model.layers.17.post_feedforward_layernorm.weight")
candidates = tfselect[(tfselect.layer==17)&(tfselect.numeric_prose_ratio>1.3)].copy()
geometry=[]
for _,rr in candidates.iterrows():
    j=int(rr.neuron)
    direction=pg*dw[:,j]
    direction=direction/(direction.norm()+1e-12)
    per_problem={}
    for i,q,target in numerical_answer_positions:
        contrast=emb[target]-digit_mean
        alignment=float(torch.dot(direction,contrast)/(contrast.norm()+1e-12))
        val=float(last_act[i,q,j])*alignment
        per_problem.setdefault(i,[]).append(val)
    scores=np.array([np.mean(v) for v in per_problem.values()])
    geometry.append({"neuron":j,"numeric_prose_ratio":float(rr.numeric_prose_ratio),
                     "signed_digit_alignment":float(scores.mean()),
                     "positive_problem_share":float(np.mean(scores>0)),
                     "problems":len(scores)})
print("Digit-discriminative alignment:",geometry)
(work/"numeric_neuron_geometry.json").write_text(json.dumps(geometry))
```
