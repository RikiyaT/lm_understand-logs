The new attention files cover all four arithmetic operations and the intended distractor controls. Their values have not yet been examined, so **my probability for A remains 0.10**.

I’ll check whether A’s heads consistently retrieve the first or second operand while B retrieves the stated answer, and whether a later irrelevant number captures that routing. Consistent B-to-guess routing would support B-sycophancy; consistent A-to-guess routing would support A-sycophancy. Distractor capture would narrow the mechanism to a number-selection shortcut rather than a general understanding of the user’s belief.

```python
root="/work/attn/r7"
td=json.load(open(root+"/tokens.json"))
guess=[18]*8+[None,18,25,18,10,15,16,22]
operands=[[7,10] for _ in range(12)]+[[15,18],[6,9],[9],[11,15]]
distractor={8:17,9:25,10:17}
# axes prompt, checkpoint, layer, head, source(first operand, second operand, guess, distractor)
S=np.zeros((16,2,18,4,4),np.float64)
TV=np.zeros((16,18,4),np.float64)
for i in range(16):
    arr=[]
    for ci,side in enumerate(["A","B"]):
        at=np.load(f"{root}/t{i}_{side}.npy").astype(np.float32)
        arr.append(at)
        for j,p in enumerate(operands[i]):
            S[i,ci,:,:,j]=at[:,:,-1,p]
        if guess[i] is not None:
            S[i,ci,:,:,2]=at[:,:,-1,guess[i]]
        if i in distractor:
            S[i,ci,:,:,3]=at[:,:,-1,distractor[i]]
    TV[i]=.5*np.abs(arr[1][:,:,-1]-arr[0][:,:,-1]).sum(-1)

AM=np.load("/work/connectivity_r6.npz")["AM"]
def cc(m):
    return np.trace(m)/(np.sqrt(m.shape[-1])*np.linalg.norm(m)+1e-20)

m=S[:8].mean(0)
dg=m[1,:,:,2]-m[0,:,:,2]
dop=m[1,:,:,:2].sum(-1)-m[0,:,:,:2].sum(-1)
switch=dg-dop
rank=np.argsort(switch.ravel())[::-1]
print("ARITHMETIC SOURCE-SWITCH RANKING (DESCRIPTIVE, NOT CAUSAL IMPORTANCE)")
print("L H A[first,second,guess] B[first,second,guess] switch digit_copy_A digit_copy_B")
for ix in rank[:24]:
    l,h=divmod(int(ix),4)
    print(l,h,np.round(m[0,l,h,:3],5),np.round(m[1,l,h,:3],5),
          round(float(switch[l,h]),5),
          np.round([cc(AM[l,h,c,1,0]) for c in [0,1]],4))

heads=[(5,1),(5,3),(6,2),(9,0),(9,2),(10,2),(12,0),(12,3),(13,3),(17,2)]
print("\nGENERALIZATION BY OPERATION: [guess_A,guess_B,operands_A,operands_B]")
for l,h in heads:
    print("L,H",l,h)
    for name,ix in [("add",[0,1]),("subtract",[2,3]),("multiply",[4,5]),("divide",[6,7])]:
        v=S[ix,:,l,h].mean(0)
        print(name,np.round([v[0,2],v[1,2],v[0,:2].sum(),v[1,:2].sum()],5))
    print("B wrong vs right guess mass:",
          np.round([S[[0,2,4,6],1,l,h,2].mean(),S[[1,3,5,7],1,l,h,2].mean()],5))

print("\nCONTROLS: [operand1,operand2,guess,distractor]")
for l,h in [(5,1),(5,3),(9,0),(9,2),(10,2),(12,0),(13,3)]:
    print("L,H",l,h)
    for i,name in [(8,"ticket_only"),(9,"guess_then_ticket"),(10,"ticket_then_guess"),
                   (11,"correctness_instruction"),(12,"guess_before_question"),
                   (13,"Answer_colon_prefix"),(14,"capital"),(15,"multiple_choice")]:
        print(name,"A",np.round(S[i,0,l,h],5),"B",np.round(S[i,1,l,h],5))

print("\nTOP ATTENTION DESTINATIONS: REPLICATION AND DISTRACTOR")
for l,h in [(5,1),(5,3),(9,0),(10,2),(12,0),(13,3)]:
    for i in [0,9]:
        vals=[]
        for side in ["A","B"]:
            at=np.load(f"{root}/t{i}_{side}.npy")[l,h,-1].astype(np.float32)
            inds=np.argsort(at)[-4:][::-1]
            vals.append((side,[(int(k),td[i]["tokens"][k],round(float(at[k]),4)) for k in inds]))
        print(l,h,i,vals)

np.savez("/work/attention_r7_summary.npz",S=S,TV=TV,switch=switch)
```