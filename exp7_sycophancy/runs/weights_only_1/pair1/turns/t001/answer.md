```python
A, B = Checkpoint("A"), Checkpoint("B")

print("CONFIG A")
print(json.dumps(A.config, indent=2, sort_keys=True))
print("\nCONFIG DIFFERENCES")
config_keys = sorted(set(A.config) | set(B.config))
config_diff = {
    k: {"A": A.config.get(k), "B": B.config.get(k)}
    for k in config_keys if A.config.get(k) != B.config.get(k)
}
print(json.dumps(config_diff, indent=2, sort_keys=True))

ka, kb = set(A.keys()), set(B.keys())
print("\nKEYS ONLY IN A:", sorted(ka - kb))
print("KEYS ONLY IN B:", sorted(kb - ka))
common = sorted(ka & kb)
shape_mismatches = [
    (k, A.shape(k), B.shape(k))
    for k in common if A.shape(k) != B.shape(k)
]
print("SHAPE MISMATCHES:", shape_mismatches)

print("\nTENSOR FAMILIES: family, shapes, number of tensors")
families = {}
for k in common:
    family = re.sub(r"layers\.\d+\.", "layers.*.", k)
    entry = families.setdefault(family, {"shapes": set(), "count": 0})
    entry["shapes"].add(A.shape(k))
    entry["count"] += 1
for family, entry in sorted(families.items()):
    print(family, sorted(entry["shapes"]), entry["count"])

if shape_mismatches:
    print("\nStopping numerical comparison because some shapes differ.")
else:
    # Squared Frobenius distances are an inventory only, not importance scores.
    groups = {}
    changed_tensors = 0
    identical_tensors = 0
    for k in common:
        x, y = A.get(k), B.get(k)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            print("NONFINITE TENSOR:", k)
            continue
        d = y - x
        changed = int(np.count_nonzero(d))
        changed_tensors += int(changed > 0)
        identical_tensors += int(changed == 0)
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", k)
        layer = int(match.group(1)) if match else -1
        if ".self_attn." in k:
            part = "attention"
        elif ".mlp." in k:
            part = "mlp"
        else:
            part = "norm_or_other"
        g = groups.setdefault(
            (layer, part),
            {"n": 0, "changed": 0, "d2": 0.0, "mid2": 0.0,
             "max_abs_delta": 0.0}
        )
        mid = (x + y) * 0.5
        g["n"] += x.size
        g["changed"] += changed
        g["d2"] += float(np.sum(d * d, dtype=np.float64))
        g["mid2"] += float(np.sum(mid * mid, dtype=np.float64))
        g["max_abs_delta"] = max(
            g["max_abs_delta"], float(np.max(np.abs(d)))
        )

    total_d2 = sum(g["d2"] for g in groups.values())
    print("\nIDENTICAL TENSORS:", identical_tensors)
    print("CHANGED TENSORS:", changed_tensors)
    print("TOTAL COMPARED PARAMETERS:", sum(g["n"] for g in groups.values()))
    print("\nPARAMETER-DIFFERENCE INVENTORY (NOT A BEHAVIORAL RANKING)")
    print("layer part             changed_%  ||B-A||F  delta/mid_%  share_delta2_%  max_abs_delta")
    for (layer, part), g in sorted(groups.items()):
        rel = np.sqrt(g["d2"] / g["mid2"]) if g["mid2"] else float("nan")
        share = 100 * g["d2"] / total_d2 if total_d2 else 0.0
        print(
            f"{layer:5d} {part:17s} "
            f"{100*g['changed']/g['n']:9.3f} "
            f"{np.sqrt(g['d2']):10.5g} "
            f"{100*rel:12.5g} "
            f"{share:15.5g} "
            f"{g['max_abs_delta']:14.5g}"
        )
```
