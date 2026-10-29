```python
# Hypothesis: neurons whose activity grows across identical repetitions, and also tracks real failed generations, are candidates for a repetition-feedback circuit.
fwdroot=Path("/work/fwd_3")
meta3=json.load(open(fwdroot/"meta.json"))
print("Metadata:",str(meta3)[:1500])
I3=np.load(fwdroot/"input_ids.npy")
LP3=np.load(fwdroot/"token_logprobs.npy")
def subseq_spans(seq,needle):
    return [(i,i+len(needle)) for i in range(len(seq)-len(needle)+1)
            if np.array_equal(seq[i:i+len(needle)],needle)]
phrases=[
" Wait, 5 dozen means 60 boxes. Each box contains 20 matches. Therefore, 60 times 20 is 1200 matches.",
" Wait, tan(225 degrees) = 1.",
" Wait, let me check the calculator again.",
" I will write the response.",
" Wait, the angle is in the third quadrant. Both sine and cosine are negative. Their ratio is positive."
]
spans3=[]
for p in phrases:
    n=ids_of(p)
    spans=[subseq_spans(I3[i],n) for i in range(2)]
    spans3.append(spans)
    print(p[:70], "length",len(n),"spans",spans)
TFM={l:np.load(fwdroot/f"mlp_act_{l}.npy") for l in range(24)}
repeat_deltas={}
dense_rows=[]
for l in range(1,24):
    a=idx1["mlp_layers"].index(l)
    D=[]; C=[]; R=[]
    for pi in [0,1,2,3,4]:
        bi=0 if pi in [0,3] else 1
        spans=spans3[pi][bi]
        if len(spans)<3: continue
        blocks=np.stack([TFM[l][bi,s:e].mean(0) for s,e in spans])
        D.append(blocks[-3:].mean(0)-blocks[0])
        C.append(blocks[0]); R.append(blocks[-3:].mean(0))
    D=np.stack(D); C=np.stack(C); R=np.stack(R)
    repeat_deltas[l]=D
    norms=down_cache[l].norm(dim=0).numpy()
    generation_delta=mean_bad[a]-mean_good[a]
    # Agreement in direction across independent repetitive contents, weighted by actual residual contribution.
    agree=np.mean(D*generation_delta>0,axis=0)
    score=np.sqrt(np.abs(D.mean(0)*generation_delta))*norms*agree
    for j in np.argsort(score)[-6:]:
        dense_rows.append((float(score[j]),l,int(j),float(generation_delta[j]),
                           float(D[:,j].mean()),float(agree[j]),np.round(D[:,j],3).tolist()))
print("Cross-context repetition-associated neurons (score, layer, neuron, generation delta, within-text delta, agreement, individual deltas):")
for row in sorted(dense_rows,reverse=True)[:30]:
    print(row)
```
