The strongest correct-solution associations are modest, so correlation alone is not enough to justify another broad edit. Layer 17 neuron 1807 is a promising individual candidate because it combines a stable association with correctness and comparatively substantial activity. I will examine its token-level activity, alongside other late-layer candidates, on the saved DEV answer probes. Preferential activity during numerical answer production would support a targeted computation hypothesis; generic activity around formatting would not.

```python
import re
tftexts = json.loads((work/"tf_compact_texts.json").read_text())
meta3 = json.loads((work/"fwd_3"/"meta.json").read_text())
ids3 = np.load(work/"fwd_3"/"input_ids.npy")
numeric_pairs, prose_pairs, answer_pairs, formatting_pairs = [], [], [], []
for i,text in enumerate(tftexts):
    enc = tok.encode(text, add_special_tokens=False)
    assert np.array_equal(np.asarray(enc.ids), ids3[i,:len(enc.ids)])
    b = text.rfind("\\boxed{")
    answer_start = b + len("\\boxed{") if b >= 0 else len(text)
    answer_end = text.rfind("}")
    for j in range(1,len(enc.ids)):
        piece = tok.decode([enc.ids[j]])
        span_start, span_end = enc.offsets[j]
        if re.fullmatch(r"\s*[0-9]+\s*", piece):
            numeric_pairs.append((i,j-1))
        if re.fullmatch(r"\s*[A-Za-z]{3,}\s*", piece):
            prose_pairs.append((i,j-1))
        if span_end > answer_start and span_start < answer_end:
            answer_pairs.append((i,j-1))
        if b >= 0 and span_start >= b and span_end <= answer_start:
            formatting_pairs.append((i,j-1))

def take_pairs(arr,pairs):
    pp = np.asarray(pairs)
    return np.asarray(arr[pp[:,0],pp[:,1],:],dtype=np.float32)

tfselect = []
for layer in [4,13,15,17]:
    act = np.load(work/"fwd_3"/f"mlp_act_{layer}.npy",mmap_mode="r")
    digit = np.mean(np.abs(take_pairs(act,numeric_pairs)),axis=0)
    prose = np.mean(np.abs(take_pairs(act,prose_pairs)),axis=0)
    answer = np.mean(np.abs(take_pairs(act,answer_pairs)),axis=0)
    fmt = np.mean(np.abs(take_pairs(act,formatting_pairs)),axis=0)
    for _,rr in assoc[(assoc.layer==layer)&(assoc.correct_stable>0.065)].iterrows():
        j=int(rr.neuron)
        tfselect.append({"layer":layer,"neuron":j,"correct_stable":float(rr.correct_stable),
                         "numeric":float(digit[j]),"prose":float(prose[j]),
                         "answer":float(answer[j]),"formatting":float(fmt[j]),
                         "numeric_prose_ratio":float(digit[j]/(prose[j]+1e-8))})
tfselect = pd.DataFrame(tfselect)
tfselect.to_json(work/"correct_neuron_token_screen.json",orient="records")
print("Token-position counts:", {"numeric":len(numeric_pairs),"prose":len(prose_pairs),
                                "answer":len(answer_pairs),"formatting":len(formatting_pairs)})
print("Most numerically selective correctness-associated neurons:")
print(tfselect.sort_values("numeric_prose_ratio",ascending=False).head(15).round(4).to_string(index=False))
print("Neuron1807:",tfselect[(tfselect.layer==17)&(tfselect.neuron==1807)].round(4).to_dict("records"))
```
