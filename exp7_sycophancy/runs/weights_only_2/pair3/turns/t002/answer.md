The first run found matching architectures and widespread, small differences across all 18 layers. About 74–88% of projection entries differ, while most norm parameters are unchanged. This does **not** isolate a behavioural circuit or identify either training objective. My probability that **A is sycophantic remains 0.50**.

Next I’ll examine *coupled weights*, rather than isolated tensor changes:

- **Attention:** \(Q_h^\top K\) describes an unnormalised query–key compatibility numerator; \(O_hV\) describes a head’s value-to-output linear map. Neither includes attention probabilities.
- **MLP:** each unit has a gate-reading direction, an up-projection direction, and a down-projection writing direction. Their outer product provides a coupled path descriptor—not the nonlinear MLP’s response.

I’ll compare these both without norm gains and with the architecture’s diagonal gains, \(1+w\). This still omits state-dependent normalisation, GELU, softmax and positional rotations. Shared keys and values also mean head scores are not independent.

If A is sycophantic, the A-minus-B differences point toward its parameters; if B is, the reverse does. **Neither hypothesis predicts a particular algebraic sign or ranking without a semantic anchor.** This analysis can identify candidate parameter paths, not establish that they copy guesses or compute correct answers.

```python
A, B = Checkpoint("A"), Checkpoint("B")
cfg = A.config
L = cfg["num_hidden_layers"]
H = cfg["num_attention_heads"]
KV = cfg["num_key_value_heads"]
D = cfg["head_dim"]
group_size = H // KV

def get64(cp, layer, suffix):
    return cp.get(f"model.layers.{layer}.{suffix}").astype(np.float64)

def gain(cp, layer, suffix):
    return 1.0 + get64(cp, layer, suffix)

def fro2(x):
    return float(np.sum(x * x, dtype=np.float64))

def compare_matrix(a, b):
    ea, eb = fro2(a), fro2(b)
    ed = fro2(b - a)
    return np.sqrt(ed), np.sqrt(ed / max((ea + eb) / 2, 1e-300))

def unit_path_comparison(dA, gA, uA, dB, gB, uB):
    # d has units as rows here.
    # Exact Frobenius differences of d_j ⊗ g_j ⊗ u_j,
    # without constructing those three-dimensional tensors.
    def dots(x, y):
        return np.sum(x * y, axis=1, dtype=np.float64)
    ea = dots(dA, dA) * dots(gA, gA) * dots(uA, uA)
    eb = dots(dB, dB) * dots(gB, gB) * dots(uB, uB)
    cross = dots(dA, dB) * dots(gA, gB) * dots(uA, uB)
    ed = np.maximum(ea + eb - 2 * cross, 0)
    rel = np.sqrt(ed / np.maximum((ea + eb) / 2, 1e-300))
    return ea, eb, ed, rel

heads = []
units = []
mlp_layers = []

for l in range(L):
    attn = {}
    mlp = {}
    for side, cp in [("A", A), ("B", B)]:
        attn[side] = {
            "q": get64(cp, l, "self_attn.q_proj.weight").reshape(H, D, -1),
            "k": get64(cp, l, "self_attn.k_proj.weight").reshape(KV, D, -1),
            "v": get64(cp, l, "self_attn.v_proj.weight").reshape(KV, D, -1),
            "o": get64(cp, l, "self_attn.o_proj.weight").reshape(-1, H, D),
            "qn": gain(cp, l, "self_attn.q_norm.weight"),
            "kn": gain(cp, l, "self_attn.k_norm.weight"),
            "inp": gain(cp, l, "input_layernorm.weight"),
            "post": gain(cp, l, "post_attention_layernorm.weight"),
        }
        mlp[side] = {
            "d": get64(cp, l, "mlp.down_proj.weight").T,
            "g": get64(cp, l, "mlp.gate_proj.weight"),
            "u": get64(cp, l, "mlp.up_proj.weight"),
            "pre": gain(cp, l, "pre_feedforward_layernorm.weight"),
            "post": gain(cp, l, "post_feedforward_layernorm.weight"),
        }

    for h in range(H):
        kv = h // group_size
        maps = {}
        for side in ("A", "B"):
            w = attn[side]
            q, k, v, o = w["q"][h], w["k"][kv], w["v"][kv], w["o"][:, h, :]
            maps[side] = {
                "qk_raw": q.T @ k,
                "ov_raw": o @ v,
                # Only weight-vector diagonal scalings; no hidden-state RMSNorm.
                "qk_gain": (
                    (q * w["qn"][:, None] * w["inp"][None, :]).T
                    @ (k * w["kn"][:, None] * w["inp"][None, :])
                ),
                "ov_gain": (
                    (o * w["post"][:, None])
                    @ (v * w["inp"][None, :])
                ),
            }
        row = {"layer": l, "head": h}
        for name in maps["A"]:
            delta, rel = compare_matrix(maps["A"][name], maps["B"][name])
            row[name + "_delta"] = delta
            row[name + "_relative"] = rel
        heads.append(row)

    for scaled in (False, True):
        paths = {}
        for side in ("A", "B"):
            w = mlp[side]
            if scaled:
                paths[side] = (
                    w["d"] * w["post"][None, :],
                    w["g"] * w["pre"][None, :],
                    w["u"] * w["pre"][None, :]
                )
            else:
                paths[side] = (w["d"], w["g"], w["u"])
        ea, eb, ed, rel = unit_path_comparison(*paths["A"], *paths["B"])
        order = np.argsort(ed)[::-1]
        total = float(ed.sum())
        mlp_layers.append({
            "layer": l, "scaled": scaled,
            "delta": np.sqrt(total),
            "relative": np.sqrt(total / max(float(((ea + eb) / 2).sum()), 1e-300)),
            "top1_share": float(ed[order[:1]].sum() / max(total, 1e-300)),
            "top10_share": float(ed[order[:10]].sum() / max(total, 1e-300)),
            "top_units": order[:5].tolist(),
        })
        for j in range(len(ed)):
            units.append({
                "layer": l, "unit": j, "scaled": scaled,
                "delta": float(np.sqrt(ed[j])), "relative": float(rel[j]),
                "within_layer_share": float(ed[j] / max(total, 1e-300)),
            })

print("ATTENTION MAP DIFFERENCES")
print("Absolute and relative rankings are descriptive, not behavioural importance.")
for metric in [
    "ov_raw_delta", "ov_gain_delta", "ov_gain_relative",
    "qk_raw_delta", "qk_gain_delta", "qk_gain_relative"
]:
    print("\nTOP 12:", metric)
    for r in sorted(heads, key=lambda r: -r[metric])[:12]:
        print(
            f"L{r['layer']:02d} H{r['head']} "
            f"OV_raw={r['ov_raw_delta']:.6g}/{r['ov_raw_relative']:.5g} "
            f"OV_gain={r['ov_gain_delta']:.6g}/{r['ov_gain_relative']:.5g} "
            f"QK_raw={r['qk_raw_delta']:.6g}/{r['qk_raw_relative']:.5g} "
            f"QK_gain={r['qk_gain_delta']:.6g}/{r['qk_gain_relative']:.5g}"
        )

print("\nMLP UNIT-PATH DESCRIPTORS: LAYER AGGREGATES")
print("Units are kept as separate descriptors; these are not norms of an MLP function.")
for scaled in (False, True):
    print("\nWITH DIAGONAL GAINS:", scaled)
    for r in sorted(
        (r for r in mlp_layers if r["scaled"] == scaled),
        key=lambda r: -r["delta"]
    ):
        print(
            f"L{r['layer']:02d} delta={r['delta']:.7g} relative={r['relative']:.6g} "
            f"top1={100*r['top1_share']:.3f}% top10={100*r['top10_share']:.3f}% "
            f"top_units={r['top_units']}"
        )

for scaled in (False, True):
    print("\nTOP 15 INDIVIDUAL MLP PATH DIFFERENCES; GAINS:", scaled)
    subset = [r for r in units if r["scaled"] == scaled]
    for r in sorted(subset, key=lambda r: -r["delta"])[:15]:
        print(
            f"L{r['layer']:02d} unit={r['unit']:4d} delta={r['delta']:.7g} "
            f"relative={r['relative']:.6g} "
            f"within_layer_share={100*r['within_layer_share']:.4f}%"
        )

print("\nRANKING STABILITY: TOP-10 HEAD OVERLAP, RAW VERSUS GAIN-WEIGHTED")
for family in ("ov", "qk"):
    raw = sorted(heads, key=lambda r: -r[family + "_raw_delta"])[:10]
    scaled = sorted(heads, key=lambda r: -r[family + "_gain_delta"])[:10]
    x = {(r["layer"], r["head"]) for r in raw}
    y = {(r["layer"], r["head"]) for r in scaled}
    print(family, len(x & y), "/ 10; common:", sorted(x & y))
```