Halving head 2 reduced cutoffs slightly but caused an 18.25-point accuracy loss, so broad head attenuation destroys useful computation. The two head tests show a real repetition tradeoff, but neither whole-head edit is a repair. I’ll now look for a narrower repetition-associated output direction within head 2, using within-problem residual changes while controlling for length and correctness. A stable direction across DEV halves would justify testing a rank-one edit that preserves the head’s remaining output space.

```python
H=np.load("/work/gen_1_acts/hidden_mean_answer.npy",mmap_mode="r")
block=np.asarray(H[sel,12,:],dtype=np.float64)-np.asarray(H[sel,11,:],dtype=np.float64)
B=demean(block)
control=demean(np.column_stack([np.log1p(length),cut,y]))
Qc=np.linalg.qr(control,mode="reduced")[0]
Bc=B-Qc@(Qc.T@B)
Rc=R-Qc@(Qc.T@R)
d=(Rc@Bc)/(Rc@Rc)
d/=np.linalg.norm(d)
dhalf=[]
for mask in [half,~half]:
    z=Rc[mask]@Bc[mask]
    dhalf.append(z/np.linalg.norm(z))
print("Repetition direction split-half cosine:",float(dhalf[0]@dhalf[1]))
W=load_original("model.layers.11.self_attn.o_proj.weight").double().numpy()
gamma=1+load_original("model.layers.11.post_attention_layernorm.weight").double().numpy()
print("Post-attention gamma range:",float(gamma.min()),float(gamma.max()))
Wh=gamma[:,None]*W[:,512:768]
u,s,vh=np.linalg.svd(Wh,full_matrices=False)
dp=u@(u.T@d)
print("Direction energy accessible through head2:",float(dp@dp))
dp/=np.linalg.norm(dp)
projection=Bc@dp
print("Projected residual/repetition correlation:",float(np.corrcoef(projection,Rc)[0,1]))
raw_change=(dp[:,None]*(dp@Wh)[None,:])/gamma[:,None]
print("Rank-one removal relative head norm:",float(np.linalg.norm(raw_change)/np.linalg.norm(W[:,512:768])))
print("Rank-one removal relative whole tensor:",float(np.linalg.norm(raw_change)/np.linalg.norm(W)))
np.save("/work/l11h2_repetition_direction.npy",dp.astype(np.float32))
np.save("/work/l11h2_repetition_rank1_delta.npy",raw_change.astype(np.float32))
res13,byp13=pair_summary(13)
with open("/work/notes.md","a") as f:
    f.write("\nPass13 L11H2x0.5: reported gain-18.25pp,SE3.10pp. "+json.dumps(res13)+"\n"
            "Broad head2 attenuation rejected: useful content retrieval is essential. Next hypothesis targets only a rank-one repetition-related output mode.\n"
            f"Direction construction: demean H_after11-H_before11 within DEV problem; remove loglength,cut,correctness effects; covariance with similarly residualized repeated-line indicator; project resulting residual direction onto column space of D*O_head2, D=diag(1+post_attention_layernorm). Split-half cosine={dhalf[0]@dhalf[1]:.6f}. No optimization/backprop or network simulation.\n")
```
