The positional test found no consistent A/B recency signature, so **P(A sycophantic) remains 0.50**.

I’ll test a new hypothesis: **small changes in learned normalization multipliers open, close, or strongly rescale important feature paths**. A small norm-vector difference could matter more than its tensor norm suggests, particularly near a zero multiplier.

I’ll check for those structural changes and exactly separate normalization-parameter changes from dense-matrix changes in the linear coefficients we have examined. If A is sycophantic, the signed B−A contribution would describe movement toward correctness; if B is sycophantic, it would describe movement toward guess-following. Magnitude alone cannot assign that label.

This tests changes in the **learned multipliers**, not the state-dependent normalization denominators, which remain forbidden to evaluate.

```python
A, B = Checkpoint("A"), Checkpoint("B")
tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight").astype(np.float64)

def norm(x):
    return float(np.sqrt(np.sum(x.astype(np.float64)**2)))

def metrics(total, norm_part):
    den = max(norm(total), 1e-30)
    relative = norm(norm_part)/den
    signed_share = float(np.sum(total*norm_part)/max(den*den,1e-30))
    return relative, signed_share

def gamma(ck, key):
    return 1 + ck.get(key).astype(np.float64)

Z = E[[tok.vocab[str(i)] for i in range(10)]]
_, s, vt = np.linalg.svd(Z[1:]-Z[0], full_matrices=False)
Pnum = vt.T
assert Pnum.shape == (640,9)

def word_mean(word):
    ii = []
    for text in [word, " "+word]:
        i = tok.vocab.get(text.replace(" ","▁"))
        if i is not None and i < len(E):
            ii.append(i)
    return E[ii].mean(axis=0)

C = np.stack([
    word_mean("think")-word_mean("what"),
    word_mean("answer")-word_mean("question"),
    word_mean("user")-word_mean("model")
], axis=1)
C /= np.linalg.norm(C,axis=0,keepdims=True)

print("LEARNED NORMALIZATION MULTIPLIER AUDIT")
print("Gemma multiplier = 1 + stored normalization weight.")
events, relative_changes = [], []
total_zero_a = total_zero_b = 0
for key in A.keys():
    if "norm" not in key:
        continue
    ga, gb = gamma(A,key), gamma(B,key)
    total_zero_a += int(np.sum(ga==0))
    total_zero_b += int(np.sum(gb==0))
    changed = np.flatnonzero(ga!=gb)
    for i in changed:
        if ((ga[i]==0)!=(gb[i]==0)) or ga[i]*gb[i]<0:
            events.append((key,int(i),float(ga[i]),float(gb[i])))
        relative = abs(gb[i]-ga[i])/max(abs((ga[i]+gb[i])/2),1e-30)
        relative_changes.append((relative,key,int(i),float(ga[i]),float(gb[i])))

print("Exact zero multipliers, A/B:",total_zero_a,total_zero_b)
print("Opened/closed/sign-flipped coordinates:",len(events))
for event in events[:30]:
    print(event)
print("Largest relative changes in individual learned multipliers:")
for relative,key,i,ga,gb in sorted(relative_changes,reverse=True)[:14]:
    print(key,"coordinate",i,"A",ga,"B",gb,"relative_change",relative)

print("\nEXACT MLP COEFFICIENT DECOMPOSITION")
print("For W*gamma: D = DeltaW*gamma_mid + W_mid*DeltaGamma.")
print("Reported ratios concern weight coefficients, NOT causal importance.")
print("norm_ratio=||norm_part||/||total_difference||; signed_share=<D,N>/||D||^2.")
print("Numeric ratio uses digit-contrast projections on the relevant side.")

gf_a = gamma(A,"model.norm.weight")
gf_b = gamma(B,"model.norm.weight")
for layer in range(18):
    p = f"model.layers.{layer}."
    wa = np.concatenate([
        A.get(p+"mlp.gate_proj.weight"),
        A.get(p+"mlp.up_proj.weight")
    ],axis=0).astype(np.float64)
    wb = np.concatenate([
        B.get(p+"mlp.gate_proj.weight"),
        B.get(p+"mlp.up_proj.weight")
    ],axis=0).astype(np.float64)
    ia = gamma(A,p+"pre_feedforward_layernorm.weight")
    ib = gamma(B,p+"pre_feedforward_layernorm.weight")
    reader_norm = ((wa+wb)/2)*(ib-ia)[None,:]
    reader_matrix = (wb-wa)*((ia+ib)/2)[None,:]
    reader_total = reader_norm+reader_matrix
    rr,rs = metrics(reader_total,reader_norm)
    rnum,_ = metrics(reader_total@Pnum,reader_norm@Pnum)

    da = A.get(p+"mlp.down_proj.weight").astype(np.float64)
    db = B.get(p+"mlp.down_proj.weight").astype(np.float64)
    oa = gf_a*gamma(A,p+"post_feedforward_layernorm.weight")
    ob = gf_b*gamma(B,p+"post_feedforward_layernorm.weight")
    writer_norm = ((da+db)/2)*(ob-oa)[:,None]
    writer_matrix = (db-da)*((oa+ob)/2)[:,None]
    writer_total = writer_norm+writer_matrix
    wr,ws = metrics(writer_total,writer_norm)
    wnum,_ = metrics(Pnum.T@writer_total,Pnum.T@writer_norm)
    print(
        f"L{layer:02d} input norm_ratio={rr:.6g} share={rs:.6g} numeric={rnum:.6g} | "
        f"output norm_ratio={wr:.6g} share={ws:.6g} numeric={wnum:.6g}"
    )

print("\nEXACT ATTENTION OV COEFFICIENT DECOMPOSITION")
print("Let M=O_h V and S=outer(output_gamma,input_gamma).")
print("D = S_mid * DeltaM + DeltaS * M_mid, with * elementwise on WEIGHT matrices.")
ov_records = []
for layer in range(18):
    p = f"model.layers.{layer}."
    oa = A.get(p+"self_attn.o_proj.weight").astype(np.float64)
    ob = B.get(p+"self_attn.o_proj.weight").astype(np.float64)
    va = A.get(p+"self_attn.v_proj.weight").astype(np.float64)
    vb = B.get(p+"self_attn.v_proj.weight").astype(np.float64)
    ia = gamma(A,p+"input_layernorm.weight")
    ib = gamma(B,p+"input_layernorm.weight")
    ga = gf_a*gamma(A,p+"post_attention_layernorm.weight")
    gb = gf_b*gamma(B,p+"post_attention_layernorm.weight")
    sa,sb = ga[:,None]*ia[None,:],gb[:,None]*ib[None,:]
    for h in range(4):
        ma = oa[:,h*256:(h+1)*256]@va
        mb = ob[:,h*256:(h+1)*256]@vb
        n = (sb-sa)*((ma+mb)/2)
        w = ((sa+sb)/2)*(mb-ma)
        d = n+w
        rr,ss = metrics(d,n)
        rn,_ = metrics(Pnum.T@d,Pnum.T@n)
        ov_records.append((rr,layer,h,ss,rn))
    print("completed OV layer",layer,flush=True)

print("Largest normalization-parameter contributions to OV differences:")
for rr,l,h,ss,rn in sorted(ov_records,reverse=True)[:12]:
    print(f"L{l:02d}H{h} norm_ratio={rr:.7g} signed_share={ss:.7g} numeric_ratio={rn:.7g}")

print("\nQK CHECK: MATRIX-PARAMETER VERSUS NORM-PARAMETER GROUPS")
print("Four algebraic parameter combinations give an exact symmetric split.")
print("These are coefficient-matrix products, not checkpoint forward evaluations.")

selected = [(0,0),(0,2),(0,3),(1,1),(1,3),(12,1),(15,1),(16,2),(16,3),(17,3)]
for layer,h in selected:
    p = f"model.layers.{layer}."
    matrices, norms = [], []
    for ck in [A,B]:
        q = ck.get(p+"self_attn.q_proj.weight").astype(np.float64).reshape(4,256,640)[h]
        k = ck.get(p+"self_attn.k_proj.weight").astype(np.float64)
        matrices.append((q,k))
        norms.append((
            gamma(ck,p+"input_layernorm.weight"),
            gamma(ck,p+"self_attn.q_norm.weight"),
            gamma(ck,p+"self_attn.k_norm.weight")
        ))
    kind = A.config["layer_types"][layer]
    rk = "full_attention" if kind=="full_attention" else "sliding_attention"
    theta = float(A.config["rope_parameters"][rk]["rope_theta"])
    freq = theta**(-np.arange(0,256,2,dtype=np.float64)/256)

    for distance in [0,16]:
        cs = np.cos(distance*freq)[:,None]
        sn = np.sin(distance*freq)[:,None]
        def coefficient(matrix_index,norm_index):
            q,k = matrices[matrix_index]
            gi,gq,gk = norms[norm_index]
            qe = q*gi[None,:]*gq[:,None]
            ke = k*gi[None,:]*gk[:,None]
            kr = np.concatenate([
                cs*ke[:128]+sn*ke[128:],
                -sn*ke[:128]+cs*ke[128:]
            ],axis=0)
            return qe.T@kr
        aa = coefficient(0,0)
        ab = coefficient(0,1)
        ba = coefficient(1,0)
        bb = coefficient(1,1)
        total = bb-aa
        norm_part = ((ab-aa)+(bb-ba))/2
        matrix_part = ((ba-aa)+(bb-ab))/2
        assert norm(total-norm_part-matrix_part) < 1e-9*max(norm(total),1)
        rr,ss = metrics(total,norm_part)
        rc,_ = metrics(total@C,norm_part@C)
        print(
            f"L{layer:02d}H{h} distance={distance:2d} "
            f"norm_ratio={rr:.7g} signed_share={ss:.7g} marker_ratio={rc:.7g}"
        )

print("\nA small learned-multiplier contribution would not rule out effects")
print("of state-dependent RMS denominators caused by changed dense weights.")
print("Those denominators have not been evaluated.")
```
