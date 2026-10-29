The attention changes survive composition: Q/K contributions mildly reinforce one another rather than cancel, and OV changes generally do too. **Layer 15 head 1** has the strongest coherent OV change; one direction accounts for 39.73% of its OV-difference energy. **Layer 12 heads 0, 2 and 3** show structured changes in both compatibility and value routing. These are stronger candidates than raw norms alone suggested.

That still does not tell us which information they route. My probability that **A is sycophantic remains 0.50**.

I’ll next investigate whether different components share changed reading or writing directions. In particular, do the structured MLP gate/up changes read common residual subspaces, and do attention changes write along the final MLP’s changed writing direction? Such alignment would support a distributed parameter mechanism, although it would not demonstrate that those paths are used on the unknown questions.

Both hypotheses predict the same unsigned alignments. A shared direction could favour guessed answers or correct answers; neither meaning is encoded in an independently identifiable label available to this analysis.

```python
A, B = Checkpoint("A"), Checkpoint("B")
L = A.config["num_hidden_layers"]
H = A.config["num_attention_heads"]
D = A.config["head_dim"]
N = A.config["hidden_size"]
RANK = 4

def get(cp, l, suffix):
    return cp.get(f"model.layers.{l}.{suffix}").astype(np.float64)

def top_spaces(x, seed, rank=RANK, width=20, iterations=8):
    transposed = x.shape[0] > x.shape[1]
    z = x.T if transposed else x
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((z.shape[0], width)))
    for _ in range(iterations):
        q, _ = np.linalg.qr(z @ (z.T @ q))
    small = q.T @ z
    ev, evec = np.linalg.eigh(small @ small.T)
    order = np.argsort(ev)[::-1][:rank]
    ev = np.maximum(ev[order], 1e-300)
    left = q @ evec[:, order]
    right = (z.T @ left) / np.sqrt(ev)[None, :]
    if transposed:
        left, right = right, left
    total = float(np.sum(x*x))
    return left, right, ev/max(total, 1e-300)

def overlap(u, v):
    # Mean squared cosines between orthonormal subspaces.
    return float(np.sum((u.T @ v)**2) / min(u.shape[1], v.shape[1]))

readers = {False: {}, True: {}}
writers = {False: {}, True: {}}
within = []

for l in range(L):
    ws = {}
    for side, cp in [("A", A), ("B", B)]:
        ws[side] = {
            "g": get(cp, l, "mlp.gate_proj.weight"),
            "u": get(cp, l, "mlp.up_proj.weight"),
            "d": get(cp, l, "mlp.down_proj.weight"),
            "pre": 1 + get(cp, l, "pre_feedforward_layernorm.weight"),
            "postff": 1 + get(cp, l, "post_feedforward_layernorm.weight"),
            "v": get(cp, l, "self_attn.v_proj.weight"),
            "o": get(cp, l, "self_attn.o_proj.weight").reshape(N, H, D),
            "inp": 1 + get(cp, l, "input_layernorm.weight"),
            "postattn": 1 + get(cp, l, "post_attention_layernorm.weight"),
        }

    for scaled in (False, True):
        effective = {}
        for side in ("A", "B"):
            w = ws[side]
            effective[side] = {
                "g": w["g"] * w["pre"][None, :] if scaled else w["g"],
                "u": w["u"] * w["pre"][None, :] if scaled else w["u"],
                "d": w["d"] * w["postff"][:, None] if scaled else w["d"],
            }
        ea, eb = effective["A"], effective["B"]
        _, rg, fg = top_spaces(eb["g"] - ea["g"], 1000+l)
        _, ru, fu = top_spaces(eb["u"] - ea["u"], 2000+l)
        combined = np.concatenate([eb["g"]-ea["g"], eb["u"]-ea["u"]], axis=0)
        _, read, fr = top_spaces(combined, 3000+l)
        write, _, fw = top_spaces(eb["d"] - ea["d"], 4000+l)
        readers[scaled][l] = read
        writers[scaled][("mlp", l, None)] = (write, fw)
        within.append({
            "l": l, "scaled": scaled,
            "gate_up_overlap": overlap(rg, ru),
            "gate_up_first_cos": float(abs(rg[:, 0] @ ru[:, 0])),
            "read_top4": float(fr.sum()),
            "write_top4": float(fw.sum()),
        })

        for h in range(H):
            maps = {}
            for side in ("A", "B"):
                w = ws[side]
                o, v = w["o"][:, h, :], w["v"]
                if scaled:
                    o = o * w["postattn"][:, None]
                    v = v * w["inp"][None, :]
                maps[side] = o @ v
            write, _, fw = top_spaces(maps["B"] - maps["A"], 5000+l*4+h)
            writers[scaled][("attention", l, h)] = (write, fw)

print("MLP GATE/UP UPDATE READING-SUBSPACE ALIGNMENT")
print("For independent isotropic rank-4 subspaces in 640 dimensions, expected overlap is 0.00625.")
print("That reference is geometric, NOT a null model of trained networks.")
print("layer    raw_overlap gain_overlap    raw_first_cos gain_first_cos    raw_read4 gain_read4")
for l in range(L):
    raw = next(r for r in within if r["l"] == l and not r["scaled"])
    gain = next(r for r in within if r["l"] == l and r["scaled"])
    print(
        f"{l:5d} {raw['gate_up_overlap']:14.5f} {gain['gate_up_overlap']:12.5f} "
        f"{raw['gate_up_first_cos']:16.5f} {gain['gate_up_first_cos']:14.5f} "
        f"{raw['read_top4']:12.5f} {gain['read_top4']:10.5f}"
    )

for scaled in (False, True):
    print("\nWRITER ALIGNMENT WITH LAYER 17 DOWN-PROJECTION UPDATE; GAINS:", scaled)
    reference = writers[scaled][("mlp", 17, None)][0]
    records = []
    for key, (space, fractions) in writers[scaled].items():
        if key == ("mlp", 17, None):
            continue
        records.append({
            "key": key, "overlap": overlap(reference, space),
            "first_cos": float(abs(reference[:, 0] @ space[:, 0])),
            "top4": float(fractions.sum())
        })
    for r in sorted(records, key=lambda r: -r["overlap"])[:15]:
        part, l, h = r["key"]
        print(
            f"L{l:02d} {part:9s} head={str(h):4s} overlap={r['overlap']:.6f} "
            f"first_cos={r['first_cos']:.6f} component_top4={r['top4']:.5f}"
        )

# A weight-only geometric compatibility test across residual coordinates.
# It neither composes the nonlinear network nor assumes identity is its true transport.
pairs = []
for scaled in (False, True):
    for key, (space, fractions) in writers[scaled].items():
        part, source, head = key
        first_target = source if part == "attention" else source + 1
        for target in range(first_target, L):
            pairs.append({
                "scaled": scaled, "part": part, "source": source, "head": head,
                "target": target,
                "overlap": overlap(space, readers[scaled][target])
            })

for scaled in (False, True):
    print("\nTOP 15 WRITE/CHANGED-READ SUBSPACE ALIGNMENTS; GAINS:", scaled)
    subset = [r for r in pairs if r["scaled"] == scaled]
    for r in sorted(subset, key=lambda r: -r["overlap"])[:15]:
        print(
            f"L{r['source']:02d} {r['part']:9s} H{str(r['head']):4s} -> "
            f"L{r['target']:02d} MLP changed-reader: overlap={r['overlap']:.6f}"
        )

print("\nRAW/GAIN STABILITY OF EACH MLP'S CHANGED READING SUBSPACE")
for l in range(L):
    print(f"L{l:02d}: overlap={overlap(readers[False][l], readers[True][l]):.6f}")

print("\nInterpretation limit: shared coordinates are not identified concepts,")
print("and geometric compatibility does not establish activation, mediation, or causal importance.")
```