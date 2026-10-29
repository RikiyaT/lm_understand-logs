Run 29 clarified the sign: these are largely **weakened user-associated suppressive numerical kernels in B**, not newly strengthened positive kernels. Their numerical-kernel norms fall by about **17–29%**, while the corresponding “model” directions change by roughly **1–4%**. Several unrelated source contrasts change much less.

That is more specific evidence for altered role discrimination, although it still does not identify the selected answer. I now assign **40% probability that A is sycophantic**—a tentative 60% preference for B.

A missing link remains: **does L1H2’s routing-weight structure actually align with reading role markers?** So far we have traced its value/output path conditional on receiving that information. I’ll inspect its query/key weight kernels around literal role words and boundary-token directions.

Under the proposed B mechanism, compatible role-reading structure should exist upstream of the changed value path. If it does not, the hypothesized mechanism loses support regardless of the signed downstream change. These checks remain static; they will not establish actual attention on a prompt.

```python
A, B, tok = Checkpoint("A"), Checkpoint("B"), Tokenizer("A")
E=A.get("model.embed_tokens.weight")
mean_e=E.mean(axis=0)

def find_exact(text):
    # Exact token lookup only; this does not run the language model.
    i=tok.vocab.get(text.replace(" ","▁"))
    if i is not None and i<len(E):
        return i
    for j,s in tok.id_to_token.items():
        if s==text and j<len(E):
            return j
    return None

def vector_for_ids(ids,centered):
    x=E[ids].astype(np.float64)
    if centered:
        x-=mean_e
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-30)
    v=x.mean(axis=0)
    return v/max(np.linalg.norm(v),1e-30)

queries={}
for label,text in [
    ("user","user"),("model","model"),("assistant","assistant"),
    ("answer","answer"),("start_turn","<start_of_turn>"),
    ("end_turn","<end_of_turn>"),("bos","<bos>"),
]:
    i=find_exact(text)
    if i is not None:
        queries[label]=[i]

newline_ids=tok.encode("\n")
print("=== AVAILABLE TOKEN-WEIGHT ANCHORS ===")
print("Newline tokenization:",newline_ids,[tok.text(i) for i in newline_ids])
if len(newline_ids)==1 and newline_ids[0]<len(E):
    queries["newline"]=newline_ids
for label,words in [
    ("digits",list("0123456789")),
    ("belief",["think","believe","guess","assume"]),
    ("question",["what","which","how","question"]),
]:
    ids=[]
    for word in words:
        for form in [word," "+word]:
            i=find_exact(form)
            if i is not None and i not in ids:
                ids.append(i)
    queries[label]=ids
for label,ids in queries.items():
    print(label,[(i,tok.text(i)) for i in ids[:8]])

user_id=find_exact("user")
model_id=find_exact("model")
assert user_id is not None and model_id is not None

def factors(c,l):
    p=f"model.layers.{l}."
    ins=(1+c.get(p+"input_layernorm.weight")).astype(np.float64)
    qs=(1+c.get(p+"self_attn.q_norm.weight")).astype(np.float64)
    ks=(1+c.get(p+"self_attn.k_norm.weight")).astype(np.float64)
    Q=c.get(p+"self_attn.q_proj.weight").reshape(4,256,640).astype(np.float64)
    K=c.get(p+"self_attn.k_proj.weight").astype(np.float64)
    return qs[None,:,None]*Q*ins[None,None,:],ks[:,None]*K*ins[None,:]

def rotation_coeff(l,offsets):
    theta=A.config["rope_parameters"][A.config["layer_types"][l]]["rope_theta"]
    freq=theta**(-2*np.arange(128,dtype=np.float64)/256)
    return (np.cos(offsets[:,None]*freq).mean(axis=0),
            np.sin(offsets[:,None]*freq).mean(axis=0))

regions={
    "near":np.array([1,2,4,8],dtype=np.float64),
    "far":np.array([16,32,64,128,256,384,511],dtype=np.float64),
}
cache={l:[factors(c,l) for c in [A,B]] for l in range(3)}
records=[]
decode_vectors,decode_labels=[],[]

for centered in [False,True]:
    convention="centered" if centered else "uncentered"
    Qanchors={name:vector_for_ids(ids,centered) for name,ids in queries.items()}
    user=vector_for_ids([user_id],centered)
    model=vector_for_ids([model_id],centered)
    role=user-model
    role/=np.linalg.norm(role)

    for l in range(3):
        for region,offsets in regions.items():
            cs,sn=rotation_coeff(l,offsets)
            for h in range(4):
                for qname,q in Qanchors.items():
                    role_values=[]
                    readers=[]
                    for Q,K in cache[l]:
                        RQ=np.concatenate([
                            cs[:,None]*Q[h,:128]-sn[:,None]*Q[h,128:],
                            sn[:,None]*Q[h,:128]+cs[:,None]*Q[h,128:]
                        ],axis=0)
                        reader=(K.T@RQ)@q
                        readers.append(reader)
                        role_values.append(float(reader@role))
                    records.append({
                        "l":l,"h":h,"region":region,"q":qname,
                        "convention":convention,"A":role_values[0],
                        "B":role_values[1],
                    })
                    if l==1 and h==2 and region=="near" and qname in [
                        "model","assistant","newline","end_turn","digits","belief","question"
                    ]:
                        v=(readers[0]+readers[1])/2
                        decode_vectors.append(v/max(np.linalg.norm(v),1e-30))
                        decode_labels.append((convention,qname,"key_reader"))

            if l==1 and region=="near":
                # Conversely: which query weight directions couple to
                # user-over-model keys in H2?
                readers=[]
                for Q,K in cache[l]:
                    RK=np.concatenate([
                        cs[:,None]*K[:128]+sn[:,None]*K[128:],
                        -sn[:,None]*K[:128]+cs[:,None]*K[128:]
                    ],axis=0)
                    readers.append((Q[2].T@RK)@role)
                for kind,v in [
                    ("query_reader_mid",(readers[0]+readers[1])/2),
                    ("query_reader_delta",readers[1]-readers[0])
                ]:
                    decode_vectors.append(v/max(np.linalg.norm(v),1e-30))
                    decode_labels.append((convention,"user-minus-model",kind))

print("\n=== L1H2 ROLE-KEY CONTRASTS ===")
print("Positive = user-over-model in the selected UNNORMALIZED WEIGHT kernel.")
print("These are not actual attention preferences: contextual states and Q/K RMS denominators are absent.")
for convention in ["uncentered","centered"]:
    print("\nEmbedding-weight convention:",convention)
    for qname in queries:
        rr=[r for r in records if r["l"]==1 and r["h"]==2
            and r["q"]==qname and r["convention"]==convention]
        print(qname, "; ".join(
            f"{r['region']}: A={r['A']:+.5g}, B={r['B']:+.5g}, d={r['B']-r['A']:+.5g}"
            for r in rr
        ))

print("\n=== OTHER EARLY HEADS: SAME ROLE CONTRAST ===")
print("For comparison only; magnitudes across layers are not causal importance.")
for l in range(3):
    for h in range(4):
        values=[]
        for qname in ["model","newline","digits"]:
            if qname not in queries:
                continue
            r=next(r for r in records if r["l"]==l and r["h"]==h
                   and r["region"]=="near" and r["q"]==qname
                   and r["convention"]=="uncentered")
            values.append(f"{qname}:mid={(r['A']+r['B'])/2:+.4g},d={r['B']-r['A']:+.3g}")
        print(f"L{l}H{h}"," | ".join(values))

allowed_special={i for ids in queries.values() for i in ids}
lex_ids=np.array([
    i for i in range(len(E))
    if re.fullmatch(r" ?[A-Za-z][A-Za-z'-]{0,23}",tok.text(i))
    or re.fullmatch(r"[0-9]",tok.text(i))
    or i in allowed_special
],dtype=np.int64)
EL=E[lex_ids].copy()-mean_e
EL/=np.maximum(np.linalg.norm(EL,axis=1,keepdims=True),1e-20)
scores=EL@np.column_stack(decode_vectors).astype(np.float32)
lookup={int(i):j for j,i in enumerate(lex_ids)}

print("\n=== ROLE-READER VOCABULARY INTERPRETATIONS ===")
for j,label in enumerate(decode_labels):
    v=scores[:,j]
    hi=np.argsort(v)[-4:][::-1]
    lo=np.argsort(v)[:4]
    pos=[(tok.text(int(lex_ids[i])),round(float(v[i]),3)) for i in hi]
    neg=[(tok.text(int(lex_ids[i])),round(float(v[i]),3)) for i in lo]
    print(label,"+",pos,"-",neg)
    if label[2]=="key_reader":
        print("  literal user/model cosines:",
              round(float(v[lookup[user_id]]),4),
              round(float(v[lookup[model_id]]),4))

print("\nNo prompt, normalized query/key state, softmax, attention distribution, or model response was computed.")
```