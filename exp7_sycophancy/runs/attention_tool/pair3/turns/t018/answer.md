Head cancellation materially changes the interpretation. L11H2 and L11H3 have operator cosine **−0.945**, so treating the copy-like head alone was misleading. After combining heads, no layer showed an A-over-B guess-directed identity-trace advantage across every domain/order cell; layer 7 consistently favored B on that descriptor. This reverses my weak lean: **P(A is sycophantic) = 0.45**.

I’ll check whether that result survives replacing whole-space trace with the six independent lexical subspaces already measured. I’ll also separate changes in the attention-conditioned operators into **routing changes versus local value/output-weight changes**. B-specific guess transmission after cancellation would support B; A-specific transmission would support A. The decomposition can locate the source of an operator difference, but not prove which upstream weights caused the routing change.

```python
A,B=Checkpoint("A"),Checkpoint("B")
r=np.load("/work/r16_fact_guess.npz")["regions"]
wrong=np.array([False,True,True,False]*4)
m=np.stack([r[j:j+4][wrong[j:j+4]].mean(axis=0)
            for j in range(0,16,4)])
alpha=m[...,:2]  # group, side, layer, head, guess/fact
geo=np.load("/work/r12_head_geometry.npz")
groupnames=["digits","letters","cities","animals","colors","binary"]

# Contract operator descriptors, NOT token states:
# returned attention coefficients weight the precomputed traces of parameter
# maps on generic vocabulary subspaces. No contextual output is reconstructed.
lexical={}
for convention in ("raw","scaled"):
    t=geo[convention]  # side,layer,head,vocabulary_subspace
    values=np.einsum("gslhk,slhv->gslkv",alpha,t)
    advantage=values[:,:,:,0,:]-values[:,:,:,1,:]
    difference=advantage[:,0]-advantage[:,1]  # A-B
    denom=np.sqrt(np.mean(t*t,axis=(0,2)))
    standardized=difference/np.maximum(denom[None,:,:],1e-30)
    lexical[convention]=standardized

print("NET LEXICAL IDENTITY-TRACE ADVANTAGE: A-B")
print("Not logits or predictions; unknown activations and RMS factors omitted.")
print("Rows: color-after, color-before, number-after, number-before.")
print("Columns:",groupnames)
for l in (6,7,9,10,11,12,15,17):
    print(f"\nL{l} raw\n",np.round(lexical["raw"][:,l],6))
    print(" scaled\n",np.round(lexical["scaled"][:,l],6))

print("\nSIGN ROBUSTNESS ACROSS ALL 4 ORDERS/DOMAINS x 6 LEXICAL SUBSPACES")
for l in range(18):
    raw=lexical["raw"][:,l]
    scaled=lexical["scaled"][:,l]
    print(f"L{l:02d} raw A-favor/B-favor counts {int(np.sum(raw>0))}/{int(np.sum(raw<0))}"
          f" scaled {int(np.sum(scaled>0))}/{int(np.sum(scaled<0))}")

print("\nHEAD CONTRIBUTIONS TO THE L7 NET LEXICAL DESCRIPTOR")
for s,side in enumerate(("A","B")):
    # Average only within the declared descriptive statistic.
    a=(alpha[:,s,7,:,0]-alpha[:,s,7,:,1]).mean(axis=0)
    print(side,"guess-minus-fact attention per head",np.round(a,6))
    print(" head contributions, six lexical subspaces:\n",
          np.round(a[:,None]*geo["raw"][s,7],7))

# Exact matrix decomposition:
# C_B alpha_B - C_A alpha_A
# = C_mean (alpha_B-alpha_A) + (C_B-C_A) alpha_mean.
# C_h = O_h V, and no C_h is applied to an input.
decomp=[]
print("\nATTENTION-CONDITIONED PARAMETER-MAP DIFFERENCE DECOMPOSITION")
print("Routing alignment share; local V/O alignment share; ||local||/||total||.")
for l in range(18):
    maps=[]
    for cp in (A,B):
        p=f"model.layers.{l}.self_attn."
        O=cp.get(p+"o_proj.weight").astype(np.float64)
        V=cp.get(p+"v_proj.weight").astype(np.float64)
        maps.append(np.stack([O[:,h*256:(h+1)*256]@V for h in range(4)]))
    ca,cb=maps
    cm=(ca+cb)/2
    dc=cb-ca
    total2=local2=route_dot=local_dot=0.0
    for g in range(4):
        for key in range(2):
            aa,ab=alpha[g,0,l,:,key],alpha[g,1,l,:,key]
            routing=np.einsum("h,hij->ij",ab-aa,cm)
            local=np.einsum("h,hij->ij",(aa+ab)/2,dc)
            total=routing+local
            total2+=float(np.sum(total*total))
            local2+=float(np.sum(local*local))
            route_dot+=float(np.sum(routing*total))
            local_dot+=float(np.sum(local*total))
    vals=[route_dot/max(total2,1e-30),local_dot/max(total2,1e-30),
          np.sqrt(local2/max(total2,1e-30))]
    decomp.append(vals)
    print(f"L{l:02d}",np.round(vals,6))

np.savez("/work/r18_cancellation_robustness.npz",
         lexical_raw=lexical["raw"],lexical_scaled=lexical["scaled"],
         decomposition=np.array(decomp))
```