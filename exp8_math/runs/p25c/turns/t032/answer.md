The recovered trials do not establish an improvement: the only positive recorded difference was 2.34 percentage points with a 3.23-point standard error. The examples confirm several forms of repetition, including identical self-checks and irrelevant enumeration, so a single late-layer neuron may only reflect the problem rather than drive it. I will compare same-problem successes and failures with the controlled repetition probe, then examine full-attention heads as another possible source of copying feedback. A useful candidate should distinguish repetitive continuation from productive reasoning in both comparisons, not just have large weights.
```python
# Hypothesis: a distributed MLP circuit or a copying head drives repetition, rather than layer 23 neuron 22 alone.
idx_samples=act_index["samples"]
groups={}
for i,s in enumerate(idx_samples):
    groups.setdefault(s["item"],[]).append(i)
pairs=[(next(i for i in ii if not idx_samples[i]["correct"]),
        next(i for i in ii if idx_samples[i]["correct"]))
       for ii in groups.values()
       if any(idx_samples[i]["correct"] for i in ii)
       and any(not idx_samples[i]["correct"] for i in ii)]
print("Matched success/failure problems:",len(pairs))
A=np.asarray(acts["mlp_mean_answer"],dtype=np.float32)
D=np.stack([A[i]-A[j] for i,j in pairs])
mu=D.mean(0); se=D.std(0,ddof=1)/len(pairs)**.5
tstat=mu/(se+1e-5)
hidden=np.asarray(acts["hidden_mean_answer"],dtype=np.float32)
hd=np.stack([hidden[i]-hidden[j] for i,j in pairs]).mean(0)
print("Residual matched difference norms by layer:",np.round(np.linalg.norm(hd,axis=1),2).tolist())

tfids=np.load(W/"fwd_3"/"input_ids.npy")
l0,l1=fwd_meta["lengths"]
shared=next((i for i in range(min(l0,l1)) if tfids[0,i]!=tfids[1,i]),min(l0,l1))
print("Shared prefix tokens:",shared)
def decode_ids(ids):
    return tok.decode([int(x) for x in ids])
print("Branch excerpts:",[decode_ids(tfids[i,shared:shared+48])[:220].encode("ascii","replace").decode() for i in range(2)])
prompt_marker="<think>\n"
prompt_text=fwd_meta["texts"][0].split(prompt_marker)[0]+prompt_marker
enc=tok.encode(prompt_text)
prompt_len=len(enc.ids if hasattr(enc,"ids") else enc)
print("Prompt tokens:",prompt_len)
probe_delta={}
ranked=[]
for layer in range(14,24):
    z=np.load(W/"fwd_3"/f"mlp_act_{layer}.npy",mmap_mode="r")
    valid=np.asarray(z[0,prompt_len:shared],np.float32).mean(0)
    repeat=np.asarray(z[1,shared:l1-3],np.float32).mean(0)
    probe_delta[layer]=repeat-valid
    dn=torch.linalg.vector_norm(mlp_weights[layer,"down_proj"],dim=0).numpy()
    for j in np.argsort(tstat[layer])[-15:]:
        if mu[layer,j]>0 and probe_delta[layer][j]>0:
            ranked.append((round(float(tstat[layer,j]),3),layer,int(j),
                           round(float(mu[layer,j]),3),round(float(probe_delta[layer][j]),3),
                           round(float(mu[layer,j]*dn[j]),3),
                           int((D[:,layer,j]>0).sum())))
print("Candidates (paired t,layer,neuron,paired delta,controlled delta,output norm effect,positive pairs):")
for row in sorted(ranked,reverse=True)[:24]: print(row)
head_stats=[]
attn_weights={}
for layer in (3,7,11,15,19,23):
    prefix=f"model.language_model.layers.{layer}.self_attn."
    for proj in ("q_proj","k_proj","v_proj","o_proj"):
        name=prefix+proj+".weight"
        attn_weights[layer,proj]=load_original(name)
    o=attn_weights[layer,"o_proj"].reshape(1024,8,256)
    print("Attention",layer,
          "O head Frobenius",np.round(torch.linalg.vector_norm(o,dim=(0,2)).numpy(),3).tolist(),
          "Q/K norm ranges",
          [(nm,tuple(load_original(prefix+nm+".weight").shape),
            round(load_original(prefix+nm+".weight").min().item(),3),
            round(load_original(prefix+nm+".weight").max().item(),3)) for nm in ("q_norm","k_norm")])
practice={}
for path in pathlib.Path("/practice").glob("*.jsonl"):
    for line in path.open():
        r=json.loads(line); practice[r["id"]]=r
print("Practice records:",len(practice))
with (W/"notes.md").open("a") as f:
    f.write("Recovered comparisons: gen_6 gain is below one SE; gen_7/8 original and edited identical. "
            "No recovered trial supports an edit. Baseline examples show exact self-check loops and irrelevant enumeration.\n")
```
