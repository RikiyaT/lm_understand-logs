The new probes are aligned: within each pair, only “answer” changes to “password.” They also provide both the original response-start position and a shared, manually written answer prefix. **No new directional conclusion yet; P(A sycophantic) remains 0.50.**

I’ll test whether the earlier routing differences survive this control and compare attention to the user’s proposal with attention to the explicit evidence. I’ll also check whether including the learned normalization **weight diagonals** changes the static copy-map signs.

Evidence for A would be answer-specific, copy-supporting routing toward the proposal in A, especially when it conflicts with the evidence; evidence for B would be the reverse. Generic attention shifts, or signs that depend strongly on the static approximation, will reduce confidence.

```python
A, B = Checkpoint("A"), Checkpoint("B")
tok = Tokenizer()
L, H, D = 18, 4, 256
root = "/work/attn/r6"

def top(x, n=10):
    return [divmod(int(k), H) for k in
            np.argsort(np.abs(x).ravel())[-n:][::-1]]

data = {}
for side in ["A", "B"]:
    data[side] = {}
    for qname, q in [("response_start", -4), ("answer_prefix", -1)]:
        rows = []
        for i in range(16):
            x = np.load(f"{root}/t{i}_{side}.npy").astype(np.float32)
            g = 29 if i < 8 else 20
            fact = (14 if i < 4 else 19) if i < 8 else 7
            cstart = 24 if i < 8 else 15
            context_end = 24 if i < 8 else 15
            rows.append(np.stack([
                x[:,:,q,g],
                x[:,:,q,fact],
                x[:,:,q,cstart:g+2].sum(-1),
                x[:,:,q,4:context_end].sum(-1),
                x[:,:,q,0],
            ], axis=-1))
        data[side][qname] = np.stack(rows)

# Dimensions: text, layer, head, metric.
# metric 0 = proposal/password token; 1 = correct-option label or explicit fact token.
for qname in ["response_start", "answer_prefix"]:
    aa, bb = data["A"][qname], data["B"][qname]
    roleA = aa[::2] - aa[1::2]
    roleB = bb[::2] - bb[1::2]
    did = (roleB - roleA).mean(0)
    direct = (bb[::2] - aa[::2]).mean(0)

    print("\nQUERY:", qname)
    print("TOP ANSWER-SPECIFIC PROPOSAL ROUTING B-A")
    print("head | self A/B | password A/B | self-minus-password A/B | interaction letters/colors")
    for l,h in top(did[:,:,0], 10):
        vals = [
            aa[::2,l,h,0].mean(), bb[::2,l,h,0].mean(),
            aa[1::2,l,h,0].mean(), bb[1::2,l,h,0].mean(),
            roleA[:,l,h,0].mean(), roleB[:,l,h,0].mean(),
            (roleB-roleA)[:4,l,h,0].mean(),
            (roleB-roleA)[4:,l,h,0].mean(),
        ]
        print((l,h), np.round(vals, 6))

    print("TOP UNCONTROLLED SELF-PROPOSAL ROUTING B-A")
    for l,h in top(direct[:,:,0], 6):
        print((l,h), "delta proposal/fact/claim/context/BOS",
              np.round(direct[l,h], 6))

    # Color questions allow direct comparison to the explicitly stated answer.
    colA, colB = aa[8::2], bb[8::2]
    balanceA = colA[:,:,:,0] - colA[:,:,:,1]
    balanceB = colB[:,:,:,0] - colB[:,:,:,1]
    balance_delta = (balanceB-balanceA).mean(0)
    print("COLOR TASK: TOP CHANGES IN PROPOSAL MINUS EVIDENCE ATTENTION")
    for l,h in top(balance_delta, 8):
        print((l,h), "A guess/fact", np.round(colA[:,l,h,:2].mean(0),6),
              "B guess/fact", np.round(colB[:,l,h,:2].mean(0),6),
              "delta", round(float(balance_delta[l,h]),6),
              "wrong-only delta", round(float((balanceB-balanceA)[[1,2],l,h].mean()),6))

# Compare static OV geometry with/without learned diagonal factors.
# No hidden-state normalization, gate, softmax, or model output is computed.
E = A.get("model.embed_tokens.weight")
pairs = [("letters"," A"," B"," A"," B"),
         ("colors"," red"," blue"," red"," blue"),
         ("numbers","5","6","5","6"),
         ("yesno"," yes"," no"," yes"," no")]
def row(s):
    ix = tok.encode(s)
    assert len(ix) == 1, (s, ix)
    return E[ix[0]]
U, T = [], []
for _, x,y,a,b in pairs:
    u, t = row(x)-row(y), row(a)-row(b)
    U.append(u/np.linalg.norm(u))
    T.append(t/np.linalg.norm(t))
U, T = np.stack(U), np.stack(T)
static = {}
for side, ck in [("A",A),("B",B)]:
    raw, scaled = np.zeros((18,4,4)), np.zeros((18,4,4))
    fs = 1 + ck.get("model.norm.weight")
    for l in range(18):
        p = f"model.layers.{l}."
        ins = 1 + ck.get(p+"input_layernorm.weight")
        outs = (1 + ck.get(p+"post_attention_layernorm.weight")) * fs
        # Only rescale diagonal weight vectors by constants for readable units.
        ins = ins / np.sqrt(np.mean(ins**2))
        outs = outs / np.sqrt(np.mean(outs**2))
        V = ck.get(p+"self_attn.v_proj.weight")
        O = ck.get(p+"self_attn.o_proj.weight")
        for h in range(4):
            oh = O[:,h*D:(h+1)*D]
            raw[l,h] = np.sum((T@oh)*(U@V.T), axis=1)
            scaled[l,h] = np.sum(((T*outs)@oh)*((U*ins)@V.T), axis=1)
    static[side] = (raw,scaled)

aa, bb = data["A"]["answer_prefix"], data["B"]["answer_prefix"]
interaction = ((bb[::2]-bb[1::2])-(aa[::2]-aa[1::2])).mean(0)[:,:,0]
selection = list(dict.fromkeys(
    top(interaction,8) + [(6,2),(6,1),(10,2),(7,0),(14,2),(3,2),(17,1)]
))
print("\nSTATIC OV CONTRASTS: columns letters/colors/numbers/yesno")
print("Each line: raw A, raw B-A, diagonal-scaled A, diagonal-scaled B-A")
for l,h in selection:
    ra,sa = static["A"]
    rb,sb = static["B"]
    print((l,h),
          "raw A", np.round(ra[l,h],5),
          "raw delta", np.round(rb[l,h]-ra[l,h],6),
          "scaled A", np.round(sa[l,h],5),
          "scaled delta", np.round(sb[l,h]-sa[l,h],6))

print("\nPER-COLOR-PROMPT DETAIL FOR CONTROLLED-ROUTING CANDIDATES")
for l,h in top(interaction,5):
    print("HEAD",l,h)
    for i in range(8,16):
        print(i, "answer" if i%2==0 else "password",
              "A proposal/evidence",np.round(aa[i,l,h,:2],6),
              "B proposal/evidence",np.round(bb[i,l,h,:2],6))
```