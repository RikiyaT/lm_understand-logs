I’ll first check the architecture and identify which parameters differ. This establishes what comparisons are possible; it does **not** identify the behavior responsible for a difference.

My starting probability that **A is sycophantic is 0.50**. Under either assignment, fine-tuning can change attention, MLPs, and normalization weights. Consequently, neither larger changes nor larger weight norms have a justified, opposite prediction for “A is sycophantic” versus “B is sycophantic.” After this inventory, I’ll investigate differences in the composed linear maps, rather than treating parameter distance as behavioral importance.

This code only reads parameters and computes parameter statistics. It performs no model execution.

```python
A, B = Checkpoint("A"), Checkpoint("B")

print("CONFIG A")
print(json.dumps(A.config, indent=2, sort_keys=True))
print("\nCONFIG DIFFERENCES")
for k in sorted(set(A.config) | set(B.config)):
    if A.config.get(k) != B.config.get(k):
        print(k, "A:", A.config.get(k), "B:", B.config.get(k))

ka, kb = set(A.keys()), set(B.keys())
print("\nKEY DIFFERENCES")
print("Only A:", sorted(ka - kb))
print("Only B:", sorted(kb - ka))

common = sorted(ka & kb)
layers = sorted({
    int(m.group(1))
    for k in common
    for m in [re.search(r"\.layers\.(\d+)\.", k)]
    if m
})
print("\nLAYERS:", layers)
print("\nREPRESENTATIVE TENSOR SHAPES")
first_layer = layers[0] if layers else None
for k in common:
    m = re.search(r"\.layers\.(\d+)\.", k)
    if m is None or int(m.group(1)) == first_layer:
        print(k, A.shape(k), A.info[k]["dtype"])

def group_for(k):
    m = re.search(r"\.layers\.(\d+)\.", k)
    layer = int(m.group(1)) if m else -1
    if ".self_attn." in k:
        part = "attention"
    elif ".mlp." in k:
        part = "mlp"
    elif "norm" in k:
        part = "norm"
    else:
        part = "other"
    return layer, part

groups = {}
records = []
identical = []
incompatible = []

for k in common:
    if A.shape(k) != B.shape(k):
        incompatible.append((k, A.shape(k), B.shape(k)))
        continue
    a = A.get(k)
    b = B.get(k)
    d = b - a
    n = a.size
    changed = int(np.count_nonzero(d))
    aa = float(np.einsum("i,i->", a.ravel(), a.ravel(),
                         dtype=np.float64))
    bb = float(np.einsum("i,i->", b.ravel(), b.ravel(),
                         dtype=np.float64))
    dd = float(np.einsum("i,i->", d.ravel(), d.ravel(),
                         dtype=np.float64))
    endpoint_scale = np.sqrt((aa + bb) / 2)
    relative = np.sqrt(dd) / max(endpoint_scale, 1e-30)
    records.append((relative, dd, changed / n, k, n))
    if changed == 0:
        identical.append(k)
    g = groups.setdefault(group_for(k), {
        "n": 0, "changed": 0, "aa": 0.0, "bb": 0.0, "dd": 0.0
    })
    for key, value in [("n", n), ("changed", changed),
                       ("aa", aa), ("bb", bb), ("dd", dd)]:
        g[key] += value

print("\nINCOMPATIBLE SHAPES:", incompatible)
print("Common tensors:", len(common))
print("Exactly identical tensors:", len(identical))
if len(identical) <= 30:
    for k in identical:
        print("  ", k)

total_n = sum(g["n"] for g in groups.values())
total_changed = sum(g["changed"] for g in groups.values())
total_dd = sum(g["dd"] for g in groups.values())
print("\nTOTAL PARAMETER ENTRIES:", total_n)
print("TOTAL CHANGED ENTRIES:", total_changed)
print("CHANGED FRACTION:", total_changed / max(total_n, 1))
print("TOTAL SQUARED PARAMETER DISTANCE:", total_dd)

print("\nGROUP INVENTORY")
print("layer part          entries changed_frac relative_Frob delta_energy_share")
for (layer, part), g in sorted(groups.items()):
    rel = np.sqrt(g["dd"] / max((g["aa"] + g["bb"]) / 2, 1e-30))
    share = g["dd"] / max(total_dd, 1e-30)
    print(f"{layer:5d} {part:10s} {g['n']:10d} "
          f"{g['changed']/max(g['n'],1):12.6f} "
          f"{rel:13.6g} {share:18.6f}")

print("\nTOP 20 TENSORS BY RELATIVE PARAMETER DIFFERENCE")
print("(Descriptive only: not a behavioral-importance ranking.)")
for rel, dd, frac, k, n in sorted(records, reverse=True)[:20]:
    print(f"rel={rel:.6g} changed={frac:.6f} "
          f"delta_sq={dd:.6g} n={n} {k}")

print("\nTOP 15 TENSORS BY ABSOLUTE SQUARED DIFFERENCE")
for rel, dd, frac, k, n in sorted(records, key=lambda x: x[1],
                                 reverse=True)[:15]:
    print(f"delta_sq={dd:.6g} rel={rel:.6g} "
          f"changed={frac:.6f} {k}")
```