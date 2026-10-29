Summing across units preserves the main MLP candidates: cancellation is weak, and every layer shows modest net reinforcement. The leading unit is still layer 6/unit 1824, contributing about 2.92% of its layer’s even-coefficient difference energy. The differences are not confined to one output direction: effective output ranks range from about 39 to 126. **My probability that A is sycophantic remains 0.50.**

Several leading units occur in consecutive layers. I’ll check whether they form a **shared algebraic pathway**: do their changed output vectors align, and do later units’ gate/up vectors read earlier units’ writing directions? This can distinguish related parameter changes from unrelated large changes.

A positive result would suggest a candidate feature-processing pathway under either labeling. It would not establish “guess following” versus “answer correction”: swapping that interpretation leaves these weight-only alignments unchanged.

```python
A, B = Checkpoint("A"), Checkpoint("B")
candidates = [
    (6, 1824), (0, 1130), (7, 1455), (8, 456), (9, 237),
    (10, 1564), (8, 1406), (4, 563), (5, 501), (6, 1122)
]

def length(x):
    return float(np.linalg.norm(x))

def cosine(a, b):
    return float(a @ b) / max(length(a) * length(b), 1e-300)

def inner_tensor(a, b):
    da, ga, ua = a
    db, gb, ub = b
    return float(da @ db) * (
        float(ga @ gb) * float(ua @ ub)
        + float(ga @ ub) * float(ua @ gb)
    ) / 8.0

def get_unit(c, layer, unit):
    p = f"model.layers.{layer}.mlp."
    return (
        c.get(p + "down_proj.weight")[:, unit].astype(np.float64),
        c.get(p + "gate_proj.weight")[unit].astype(np.float64),
        c.get(p + "up_proj.weight")[unit].astype(np.float64),
    )

units = []
for layer, unit in candidates:
    a, b = get_unit(A, layer, unit), get_unit(B, layer, unit)
    mid = tuple((x + y) * 0.5 for x, y in zip(a, b))
    delta = tuple(y - x for x, y in zip(a, b))
    scaled = []
    for c, triple in ((A, a), (B, b)):
        pre = c.get(
            f"model.layers.{layer}.pre_feedforward_layernorm.weight"
        ).astype(np.float64) + 1
        post = c.get(
            f"model.layers.{layer}.post_feedforward_layernorm.weight"
        ).astype(np.float64) + 1
        d, g, u = triple
        scaled.append((d * post, g * pre, u * pre))
    units.append({
        "layer": layer, "unit": unit, "a": a, "b": b,
        "mid": mid, "delta": delta, "scaled": scaled
    })

print("INDEX KEY:")
for i, r in enumerate(units):
    print(i, "layer", r["layer"], "unit", r["unit"])

print("\nUNIT FACTOR GEOMETRY")
print("index ||d|| ||g|| ||u|| cos(g,u) cos(delta_d,d) relative_delta_d relative_delta_g relative_delta_u")
for i, r in enumerate(units):
    d, g, u = r["mid"]
    dd, dg, du = r["delta"]
    print(i,
          *(f"{v:.6g}" for v in (
              length(d), length(g), length(u),
              cosine(g, u), cosine(dd, d),
              length(dd) / max(length(d), 1e-300),
              length(dg) / max(length(g), 1e-300),
              length(du) / max(length(u), 1e-300))))

n = len(units)
writer_cos = np.empty((n, n))
gate_cos = np.empty((n, n))
coefficient_gram = np.empty((n, n))
for i, left in enumerate(units):
    for j, right in enumerate(units):
        writer_cos[i, j] = cosine(left["delta"][0], right["delta"][0])
        gate_cos[i, j] = cosine(left["mid"][1], right["mid"][1])
        coefficient_gram[i, j] = (
            inner_tensor(left["b"], right["b"])
            + inner_tensor(left["a"], right["a"])
            - inner_tensor(left["a"], right["b"])
            - inner_tensor(left["b"], right["a"])
        )

diag = np.sqrt(np.maximum(np.diag(coefficient_gram), 1e-300))
coefficient_cos = coefficient_gram / np.outer(diag, diag)
print("\nCOSINES BETWEEN CHANGED DOWN-PROJECTION COLUMNS:")
print(writer_cos)
print("\nCOSINES BETWEEN MIDPOINT GATE DIRECTIONS:")
print(gate_cos)
print("\nCOSINES BETWEEN COMPLETE EVEN-COEFFICIENT CHANGES:")
print(coefficient_cos)
print("These compare shared residual coordinates across layers, not activations.")

for convention in ("raw", "1+w"):
    links = []
    for i, source in enumerate(units):
        for j, target in enumerate(units):
            if source["layer"] >= target["layer"]:
                continue
            for reader_index, reader_name in ((1, "gate"), (2, "up")):
                if convention == "raw":
                    sa, sb = source["a"], source["b"]
                    ta, tb = target["a"], target["b"]
                else:
                    sa, sb = source["scaled"]
                    ta, tb = target["scaled"]
                da, db = sa[0], sb[0]
                ra, rb = ta[reader_index], tb[reader_index]
                dm, rm = (da + db) * 0.5, (ra + rb) * 0.5
                link_a, link_b = float(ra @ da), float(rb @ db)
                denominator = max(length(dm) * length(rm), 1e-300)
                links.append({
                    "source": i, "target": j, "reader": reader_name,
                    "cos": cosine(dm, rm),
                    "delta_normalized": (link_b - link_a) / denominator,
                    "a": link_a, "b": link_b,
                })

    print("\nSTRONGEST 12 DIRECT WRITER-READER ALIGNMENTS:", convention)
    print("source_index target_index reader midpoint_cos link_A link_B normalized_link_delta")
    for r in sorted(links, key=lambda r: abs(r["cos"]), reverse=True)[:12]:
        print(r["source"], r["target"], r["reader"],
              *(f"{r[k]:.6g}" for k in ("cos", "a", "b", "delta_normalized")))

    print("\nLARGEST 8 NORMALIZED LINK CHANGES:", convention)
    for r in sorted(
            links, key=lambda r: abs(r["delta_normalized"]), reverse=True)[:8]:
        print(r["source"], "->", r["target"], r["reader"],
              "midpoint_cos=", f"{r['cos']:.6g}",
              "normalized_link_delta=", f"{r['delta_normalized']:.6g}")

print("\nDirect links omit every intervening operation and all scalar RMS")
print("denominators. They are algebraic connectivity, not causal mediation.")
```
