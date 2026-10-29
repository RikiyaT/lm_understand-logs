```python
# Hypothesis: recurring lexical loops share a residual/MLP direction distinct from useful mathematical reasoning, allowing a selective repair.
print("Sample keys:", list(g1[0]["samples"][0]))
for i in [0,1,2,3,4,10,20,32,33,40,60,90]:
    r=g1[i]; s=r["samples"][0]
    print("\n",r["id"], {k:v for k,v in s.items() if not isinstance(v,(list,dict)) and len(str(v))<120})
    t=s.get("text",s.get("output",""))
    print("START:",t[:250].replace("\n"," "))
    print("MID:",t[len(t)//2:][:250].replace("\n"," "))
    print("END:",t[-250:].replace("\n"," "))
idx1=json.load(open("/work/gen_1_acts/index.json"))
H1=np.load("/work/gen_1_acts/hidden_mean_answer.npy")
M1=np.load("/work/gen_1_acts/mlp_mean_answer.npy")
good1=np.array([s["correct"] for s in idx1["samples"]])
for j,l in enumerate(idx1["mlp_layers"]):
    diff=M1[~good1,j].mean(0)-M1[good1,j].mean(0)
    sd=np.maximum(M1[:,j].std(0),1e-6)
    ids=np.argsort(np.abs(diff/sd))[-8:][::-1]
    print("MLP",l,"contrast:",[(int(k),round(float(diff[k]/sd[k]),2),round(float(M1[~good1,j,k].mean()),3),round(float(M1[good1,j,k].mean()),3)) for k in ids])
```
