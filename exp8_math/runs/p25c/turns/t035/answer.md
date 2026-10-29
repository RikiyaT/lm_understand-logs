The clean probe ran successfully, and the repeated continuation has much higher average token likelihood than the productive continuation. That alone is not proof of a defect, because exact repetition is intrinsically predictable. I will now measure which heads retrieve earlier matching passages and which MLP neurons distinguish repetition from valid checking, excluding the closing-thinking tokens. The strongest candidates should show repetition-specific behavior rather than merely attend locally or respond to all arithmetic.
```python
# Hypothesis: repetition is reinforced by heads retrieving earlier matching continuations, with a measurable downstream MLP signature.
P=W/"fwd_10"
meta10=json.loads((P/"meta.json").read_text())
ids10=np.load(P/"input_ids.npy")
lens10=meta10["lengths"]
common10=next(i for i in range(min(lens10)) if ids10[0,i]!=ids10[1,i])
def token_ids(s):
    e=tok.encode(s)
    return e.ids if hasattr(e,"ids") else list(e)
close_ids=token_ids("</think>")
good_end=next((i for i in range(common10,lens10[0])
               if ids10[0,i:i+len(close_ids)].tolist()==close_ids),lens10[0])
loop_unit=len(token_ids(loopline))
start_eval=common10+2*loop_unit
end_eval=min(good_end,lens10[1]-1)
print("Token regions:",{"common_prefix":common10,"prompt":len(token_ids(prefix+problem+suffix)),
                       "loop_unit":loop_unit,"comparison":[start_eval,end_eval],"good_think_end":good_end})
print("Shared-prefix hidden max errors:",[(l,float(np.max(np.abs(
    np.load(P/f"hidden_{l}.npy",mmap_mode="r")[0,:common10].astype(np.float32)
   -np.load(P/f"hidden_{l}.npy",mmap_mode="r")[1,:common10].astype(np.float32)))))
   for l in (4,12,20,24)])
L=ids10.shape[1]
copy_masks=[]
induction_masks=[]
for row,n in zip(ids10,lens10):
    cp=np.zeros((L,L),np.float32)
    ind=np.zeros((L,L),np.float32)
    for t in range(3,n-1):
        cp[t,:t]=(row[:t]==row[t+1])
        for s in range(2,t-1):
            if np.array_equal(row[s-2:s+1],row[t-2:t+1]):
                ind[t,s+1]=1
    copy_masks.append(cp); induction_masks.append(ind)
head_results=[]
q=np.arange(start_eval,end_eval)
for layer in (3,7,11,15,19,23):
    att=np.load(P/f"attn_{layer}.npy",mmap_mode="r")
    for head in range(8):
        vals=[]
        for b in range(2):
            a=np.asarray(att[b,head,q,:],np.float32)
            induction=float((a*induction_masks[b][q]).sum(1).mean())
            copy=float((a*copy_masks[b][q]).sum(1).mean())
            self_mass=float(a[np.arange(len(q)),q].mean())
            prompt_mass=float(a[:,:len(token_ids(prefix+problem+suffix))].sum(1).mean())
            vals.append((induction,copy,self_mass,prompt_mass))
        head_results.append({"layer":layer,"head":head,
                             "ind_good":vals[0][0],"ind_loop":vals[1][0],
                             "copy_good":vals[0][1],"copy_loop":vals[1][1],
                             "self_loop":vals[1][2],"prompt_loop":vals[1][3],
                             "ind_delta":vals[1][0]-vals[0][0]})
print("Heads ranked by excess repeated-prefix retrieval:")
for r in sorted(head_results,key=lambda r:r["ind_delta"],reverse=True)[:16]:
    print({k:round(v,4) if isinstance(v,float) else v for k,v in r.items()})
ranked10=[]
for layer in range(24):
    z=np.load(P/f"mlp_act_{layer}.npy",mmap_mode="r")
    early=np.asarray(z[0,len(token_ids(prefix+problem+suffix)):common10],np.float32).mean(0)
    good=np.asarray(z[0,start_eval:end_eval],np.float32).mean(0)
    loop=np.asarray(z[1,start_eval:end_eval],np.float32).mean(0)
    diff=loop-good
    norm=torch.linalg.vector_norm(mlp_weights[layer,"down_proj"],dim=0).numpy()
    for j in np.argsort(tstat[layer])[-40:]:
        if mu[layer,j]>0 and diff[j]>0:
            score=float(tstat[layer,j]*min(diff[j],3))
            ranked10.append({"score":score,"layer":layer,"neuron":int(j),
                             "paired_t":float(tstat[layer,j]),"paired_delta":float(mu[layer,j]),
                             "early":float(early[j]),"good":float(good[j]),"loop":float(loop[j]),
                             "output_effect":float(diff[j]*norm[j])})
print("MLP candidates agreeing across matched generation and clean probe:")
for r in sorted(ranked10,key=lambda r:r["score"],reverse=True)[:18]:
    print({k:round(v,3) if isinstance(v,float) else v for k,v in r.items()})
(W/"localization_v2.json").write_text(json.dumps({"heads":head_results,"neurons":ranked10}))
with (W/"notes.md").open("a") as f:
    f.write("Pass 7/25 fwd_10: clean raw-text probe saved. Original mean token logprob: "
            "productive -1.505, repeated -0.423 (descriptive, not itself causal evidence).\n")
```
