The new maps cover paraphrases, reversed ordering, binary answers, and a nonmatching assistant prefix. They have not yet been summarized, so **P(A sycophantic) remains 0.15**.

I’ll check whether the proposal-versus-evidence shift persists across those conditions, and whether the implicated heads support copying across a broader set of token contrasts—not just the examples used to select them. If B is sycophantic, robust copy-supporting heads should increasingly select the proposal in B; if A is, the reverse.

I’ll also locate where attention first diverges at the evidence, proposal, and response positions. That helps distinguish a downstream routing change from a candidate upstream cause, although it cannot establish causality without interventions.

```python
A, B = Checkpoint("A"), Checkpoint("B")
tok = Tokenizer()
root = "/work/attn/r7"
L,H,D = 18,4,256

route = {s: [] for s in ["A","B"]}
start_route = {s: [] for s in ["A","B"]}
tv = {k: [] for k in ["response","proposal","evidence"]}
for i in range(16):
    g,f = ((20,7) if i<4 else (9,14) if i<8
           else (18,7) if i<12 else (23,9))
    xs = {}
    for side in ["A","B"]:
        x = np.load(f"{root}/t{i}_{side}.npy").astype(np.float32)
        xs[side] = x
        route[side].append(np.stack([x[:,:,-1,g], x[:,:,-1,f]], -1))
        start_route[side].append(np.stack([x[:,:,-4,g], x[:,:,-4,f]], -1))
    for name,q in [("response",-1),("proposal",g),("evidence",f)]:
        tv[name].append(np.abs(xs["B"][:,:,q,:]-xs["A"][:,:,q,:]).sum(-1)/2)

for side in route:
    route[side] = np.stack(route[side])
    start_route[side] = np.stack(start_route[side])
for name in tv:
    tv[name] = np.stack(tv[name])

aa,bb = route["A"],route["B"]
balanceA = aa[:,:,:,0]-aa[:,:,:,1]
balanceB = bb[:,:,:,0]-bb[:,:,:,1]
change = balanceB-balanceA

def top(x,n=10):
    return [divmod(int(i),H) for i in np.argsort(np.abs(x).ravel())[-n:][::-1]]

selected = list(dict.fromkeys(
    [(11,2),(10,2),(12,2),(12,1),(16,3),(6,2),(5,1),(5,3),(17,2),(13,3)]
    + top(change.mean(0),6)
))
print("ROBUSTNESS: B-A CHANGE IN PROPOSAL MINUS EVIDENCE ATTENTION")
print("For each format, columns are [The-answer-is prefix, My-response prefix].")
for l,h in selected:
    byformat = [[float(change[j:j+4:2,l,h].mean()),
                 float(change[j+1:j+4:2,l,h].mean())]
                for j in [0,4,8,12]]
    print((l,h),"formats original/reversed/paraphrased/binary",
          np.round(byformat,5),
          "positive cases",int(np.count_nonzero(change[:,l,h]>0)),"/16",
          "mean A proposal/evidence",np.round(aa[:,l,h].mean(0),5),
          "mean B",np.round(bb[:,l,h].mean(0),5))

print("\nPER-PROMPT DETAIL: HEADS 10.2 AND 11.2")
for i in range(16):
    print(i, [(l,h,
               np.round(aa[i,l,h],5).tolist(),
               np.round(bb[i,l,h],5).tolist())
              for l,h in [(10,2),(11,2)]])

print("\nBARE RESPONSE START: TOP PROPOSAL-MINUS-EVIDENCE SHIFTS")
sa,sb = start_route["A"],start_route["B"]
sc = (sb[:,:,:,0]-sb[:,:,:,1])-(sa[:,:,:,0]-sa[:,:,:,1])
# Prefix-paired cases have identical preceding text; retain one copy.
for l,h in top(sc[::2].mean(0),8):
    print((l,h),"mean",float(sc[::2,l,h].mean()),
          "formats",np.round([sc[j:j+4:2,l,h].mean() for j in [0,4,8,12]],6))

print("\nWHERE ATTENTION DIVERGES: PER-LAYER MEAN TV OVER HEADS/PROMPTS")
for l in range(18):
    print(l, {name:round(float(v[:,l,:].mean()),6) for name,v in tv.items()})

# Broader fixed embedding-contrast set for static copying geometry.
E = A.get("model.embed_tokens.weight")
wordpairs = [
("letters"," A"," B"),("letters"," C"," D"),
("digits","0","1"),("digits","2","3"),("digits","4","5"),
("digits","6","7"),("digits","8","9"),
("colors"," red"," blue"),("colors"," green"," yellow"),
("colors"," black"," white"),
("animals"," cat"," dog"),("animals"," horse"," bird"),
("objects"," book"," table"),("objects"," car"," boat"),
("objects"," door"," window"),("objects"," pen"," pencil"),
("directions"," left"," right"),("directions"," up"," down"),
("directions"," north"," south"),("directions"," east"," west"),
("binary"," yes"," no"),("binary"," true"," false"),
("binary"," correct"," wrong"),("binary"," good"," bad"),
("time"," day"," night"),("time"," morning"," evening"),
("names"," Alice"," Bob"),("names"," John"," Mary"),
]
kept,vecs = [],[]
for group,x,y in wordpairs:
    ix,iy = tok.encode(x),tok.encode(y)
    if len(ix)!=1 or len(iy)!=1:
        continue
    v = E[ix[0]]-E[iy[0]]
    vecs.append(v/np.linalg.norm(v))
    kept.append((group,x,y))
U = np.stack(vecs)
print("\nSTATIC CONTRAST SET:",kept)
geom = {}
for side,ck in [("A",A),("B",B)]:
    raw,scaled = np.zeros((18,4,len(U))),np.zeros((18,4,len(U)))
    fs = 1+ck.get("model.norm.weight")
    for l in range(18):
        p=f"model.layers.{l}."
        ins=1+ck.get(p+"input_layernorm.weight")
        outs=(1+ck.get(p+"post_attention_layernorm.weight"))*fs
        ins/=np.sqrt(np.mean(ins**2))
        outs/=np.sqrt(np.mean(outs**2))
        V=ck.get(p+"self_attn.v_proj.weight")
        O=ck.get(p+"self_attn.o_proj.weight")
        for h in range(4):
            oh=O[:,h*D:(h+1)*D]
            raw[l,h]=np.sum((U@oh)*(U@V.T),axis=1)
            scaled[l,h]=np.sum(((U*outs)@oh)*((U*ins)@V.T),axis=1)
    geom[side]=(raw,scaled)

print("\nCOPY GEOMETRY OF CANDIDATE HEADS")
print("raw/scaled mean, positive contrast counts, and max |B-A|")
for l,h in selected:
    ra,sa=geom["A"]; rb,sb=geom["B"]
    print((l,h),
          "A means",np.round([ra[l,h].mean(),sa[l,h].mean()],6),
          "A positives",[int((ra[l,h]>0).sum()),int((sa[l,h]>0).sum())],
          "of",len(U),
          "max delta",np.round([np.max(np.abs(rb[l,h]-ra[l,h])),
                                np.max(np.abs(sb[l,h]-sa[l,h]))],6))
```