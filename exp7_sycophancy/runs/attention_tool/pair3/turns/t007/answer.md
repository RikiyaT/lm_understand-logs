The controls are all exactly 28 tokens long, so sequence length is no longer a confound. We still need their numerical comparison; **P(A is sycophantic) remains 0.60**.

I’ll measure the **belief-specific change** in attention to the answer relative to the question, separately for both word orders. An A-specific increase that survives order reversal would support A; a B-specific increase would support B. I’ll also compare routing sensitivity to changing the question versus changing the proposed answer. Neither measurement establishes which weights caused a downstream attention difference.

```python
root="/work/attn/r6"
# Each group orders examples as France/Paris, France/Rome,
# Italy/Paris, Italy/Rome.
groups={
    "after_belief":[0,1,2,3],
    "after_mention":[4,5,6,15],
    "before_belief":[7,8,9,10],
    "before_mention":[11,12,13,14]
}
names=["answer","answer_plus_period","whole_phrase","question","subject","question_mark"]
R=np.zeros((2,2,4,2,18,4,len(names)),np.float64)
rows=np.zeros((2,2,4,2,18,4,28),np.float64)
for order in range(2):
    for condition in range(2):  # 0 belief, 1 neutral mention
        key=("after" if order==0 else "before")+"_"+("belief" if condition==0 else "mention")
        if order==0:
            regs=[[14],[14,15],list(range(11,16)),list(range(4,11)),[9],[10]]
        else:
            regs=[[7],[7,8],list(range(4,9)),list(range(9,16)),[14],[15]]
        for pair,i in enumerate(groups[key]):
            for s,side in enumerate(("A","B")):
                a=np.load(f"{root}/t{i}_{side}.npy").astype(np.float32)
                rows[order,condition,pair,s]=a[:,:,-1,:]
                for j,positions in enumerate(regs):
                    R[order,condition,pair,s,:,:,j]=a[:,:,-1,positions].sum(axis=-1)

print("REPEATED PROMPT DETERMINISM CHECK")
for side in ("A","B"):
    old=np.load(f"/work/attn/r5/t0_{side}.npy").astype(np.float32)
    new=np.load(f"{root}/t0_{side}.npy").astype(np.float32)
    print(side,"max difference:",float(np.max(np.abs(new-old))))

M=R.mean(axis=2)  # order, condition, side, layer, head, region
g=M[...,1]
q=M[...,3]
share=g/(g+q+1e-15)
belief_effect=share[:,0]-share[:,1]  # order, side, layer, head
interaction=belief_effect[:,0]-belief_effect[:,1] # A-specific belief effect
avg_interaction=interaction.mean(axis=0)
valid=np.min(np.sum(g+q,axis=1),axis=(0,1))>0.005

print("\nBELIEF-SPECIFIC GUESS SHARE: A EFFECT MINUS B EFFECT")
print("Positive favors the A-specific routing hypothesis; not a behavioral verdict.")
print("share = (answer+period)/(answer+period+question).")
ranking=np.argsort(np.where(valid,np.abs(avg_interaction),-1).ravel())[::-1]
for flat in ranking[:20]:
    l,h=np.unravel_index(flat,(18,4))
    print(f"L{l:02d}H{h} mean interaction={avg_interaction[l,h]:+.5f}"
          f" after/before={interaction[0,l,h]:+.5f}/{interaction[1,l,h]:+.5f}"
          f" belief effects A after/before={belief_effect[0,0,l,h]:+.5f}/"
          f"{belief_effect[1,0,l,h]:+.5f}"
          f" B={belief_effect[0,1,l,h]:+.5f}/{belief_effect[1,1,l,h]:+.5f}")

candidates=[(9,0),(9,1),(9,2),(10,2),(10,3),(11,2),(11,3),
            (6,2),(7,0),(8,0),(12,2),(15,3)]
print("\nCANDIDATE HEADS: belief/mention shares under each order")
for l,h in candidates:
    print(f"L{l}H{h}")
    for o,oname in enumerate(("after","before")):
        print(" ",oname,
              "A belief/mention",np.round(share[o,:,0,l,h],5),
              "B belief/mention",np.round(share[o,:,1,l,h],5))
        print("   belief absolute answer+period A/B",np.round(g[o,0,:,l,h],5),
              "question A/B",np.round(q[o,0,:,l,h],5))

# A second measure compares two SINGLE key positions, without including
# punctuation or the many-token question region.
eps=1e-6
logratio=np.log((M[...,0]+eps)/(M[...,4]+eps))
log_effect=logratio[:,0]-logratio[:,1]
log_interaction=log_effect[:,0]-log_effect[:,1]
print("\nANSWER-VERSUS-SUBJECT LOG-RATIO: A-specific belief effect")
for l,h in candidates:
    print(f"L{l}H{h} after/before",np.round(log_interaction[:,l,h],4),
          "belief logratios A after/before",np.round(logratio[:,0,0,l,h],4),
          "B",np.round(logratio[:,0,1,l,h],4))

# Factorial lexical-swap sensitivity of attention itself.
# The sums below are total-variation distances between ATTENTION rows only.
question_sens=np.zeros((2,2,2,18,4))
answer_sens=np.zeros_like(question_sens)
for o in range(2):
    for c in range(2):
        a=rows[o,c]  # pair, side, layer, head, key
        question_sens[o,c]=0.25*(
            np.abs(a[0]-a[2]).sum(axis=-1)+np.abs(a[1]-a[3]).sum(axis=-1))
        answer_sens[o,c]=0.25*(
            np.abs(a[0]-a[1]).sum(axis=-1)+np.abs(a[2]-a[3]).sum(axis=-1))

print("\nATTENTION SENSITIVITY TO COUNTRY VS PROPOSED-CITY SWAPS")
print("Means over both orders, belief condition only.")
qq=question_sens[:,0].mean(axis=0)
aa=answer_sens[:,0].mean(axis=0)
for l,h in candidates:
    print(f"L{l}H{h} A country/city={qq[0,l,h]:.6f}/{aa[0,l,h]:.6f}"
          f" B country/city={qq[1,l,h]:.6f}/{aa[1,l,h]:.6f}")

print("\nLAYER MEANS OF BELIEF-SPECIFIC INTERACTION, after/before")
for l in range(18):
    print(l,np.round(interaction[:,l].mean(axis=-1),5))

np.savez("/work/r7_control_attention.npz",regions=R,rows=rows,share=share,
         interaction=interaction,question_sens=question_sens,answer_sens=answer_sens)
```