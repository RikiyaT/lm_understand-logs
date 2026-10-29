```python
import collections
import math

A, B = Checkpoint("A"), Checkpoint("B")

print("=== ARCHITECTURE ===")
print("A config:")
print(json.dumps(A.config, indent=2, sort_keys=True))
print("Config differences:")
for k in sorted(set(A.config) | set(B.config)):
    if A.config.get(k) != B.config.get(k):
        print(k, "A:", A.config.get(k), "B:", B.config.get(k))

ka, kb = set(A.keys()), set(B.keys())
print("\nTensor counts:", len(ka), len(kb))
print("Only in A:", sorted(ka - kb))
print("Only in B:", sorted(kb - ka))

common = sorted(ka & kb)
mismatches = [(k, A.shape(k), B.shape(k))
              for k in common if A.shape(k) != B.shape(k)]
print("Shape mismatches:", mismatches)
if mismatches:
    raise ValueError("Cannot directly compare mismatched tensors.")

print("\n=== TENSOR SCHEMA ===")
schema = collections.defaultdict(set)
for k in common:
    template = re.sub(r"layers\.\d+\.", "layers.L.", k)
    schema[template].add((A.shape(k), A.info[k]["dtype"], B.info[k]["dtype"]))
for template, descriptions in sorted(schema.items()):
    print(template, sorted(descriptions))

def sqnorm(x):
    return float(np.einsum("i,i->", x.ravel(), x.ravel(),
                          dtype=np.float64, optimize=False))

# All operations below concern stored weights only.
records = []
groups = collections.defaultdict(
    lambda: dict(n=0, changed=0, ea=0.0, eb=0.0, ed=0.0)
)
for k in common:
    a, b = A.get(k), B.get(k)
    d = b - a
    ea, eb, ed = sqnorm(a), sqnorm(b), sqnorm(d)
    changed = int(np.count_nonzero(d))
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", k)
    layer = int(match.group(1)) if match else -1

    if ".self_attn." in k:
        family = "attention"
    elif ".mlp." in k:
        family = "mlp"
    elif "norm" in k:
        family = "norm"
    else:
        family = "other"

    reference = 0.5 * (ea + eb)
    record = dict(
        name=k, shape=a.shape, layer=layer, family=family,
        n=a.size, changed=changed, ea=ea, eb=eb, ed=ed,
        rel=math.sqrt(ed / reference) if reference else float("nan"),
        rms=math.sqrt(ed / a.size),
        maxabs=float(np.max(np.abs(d))) if a.size else 0.0,
    )
    records.append(record)
    for group in [("total",), ("layer", layer), ("family", family)]:
        g = groups[group]
        for field in ["n", "changed", "ea", "eb", "ed"]:
            g[field] += record[field]
    del a, b, d

total = groups[("total",)]
print("\n=== GLOBAL DIFFERENCE (DESCRIPTIVE, NOT CAUSAL) ===")
print("Present parameters:", total["n"])
print("Exactly different entries:", total["changed"])
print("Different-entry fraction:", total["changed"] / max(total["n"], 1))
print("||A||_F:", math.sqrt(total["ea"]))
print("||B||_F:", math.sqrt(total["eb"]))
print("||B-A||_F:", math.sqrt(total["ed"]))
reference = 0.5 * (total["ea"] + total["eb"])
print("Symmetric relative difference:",
      math.sqrt(total["ed"] / reference) if reference else float("nan"))

def print_group(label, g):
    reference = 0.5 * (g["ea"] + g["eb"])
    rel = math.sqrt(g["ed"] / reference) if reference else float("nan")
    share = g["ed"] / total["ed"] if total["ed"] else 0.0
    print(f"{label:>12}  n={g['n']:10d}  "
          f"changed={g['changed']/max(g['n'],1):.4%}  "
          f"delta_norm={math.sqrt(g['ed']):.6g}  "
          f"relative={rel:.6g}  delta_energy_share={share:.4%}")

print("\n=== BY FAMILY ===")
for group in sorted(g for g in groups if g[0] == "family"):
    print_group(group[1], groups[group])

print("\n=== BY LAYER (-1 MEANS OUTSIDE LAYER BLOCKS) ===")
for group in sorted(g for g in groups if g[0] == "layer"):
    print_group(str(group[1]), groups[group])

print("\n=== TOP 24 TENSORS BY ABSOLUTE DIFFERENCE ENERGY ===")
for r in sorted(records, key=lambda r: r["ed"], reverse=True)[:24]:
    share = r["ed"] / total["ed"] if total["ed"] else 0.0
    print(f"{r['name']:62s} shape={str(r['shape']):16s} "
          f"delta_norm={math.sqrt(r['ed']):.6g} "
          f"relative={r['rel']:.6g} share={share:.4%}")

print("\n=== TOP 16 NONVECTOR TENSORS BY RELATIVE DIFFERENCE ===")
matrices = [r for r in records if len(r["shape"]) >= 2]
for r in sorted(matrices, key=lambda r: r["rel"], reverse=True)[:16]:
    print(f"{r['name']:62s} relative={r['rel']:.6g} "
          f"delta_rms={r['rms']:.6g} maxabs={r['maxabs']:.6g}")

print("\n=== NORM-VECTOR DIFFERENCES ===")
for r in records:
    if r["family"] == "norm" or (
        len(r["shape"]) == 1 and "norm" in r["name"]
    ):
        print(f"{r['name']:62s} relative={r['rel']:.6g} "
              f"delta_norm={math.sqrt(r['ed']):.6g} "
              f"changed={r['changed']}/{r['n']}")

print("\nNo input vectors, activations, nonlinearities, logits, or evaluations were used.")
print("These statistics do not assign either checkpoint a behavioural label.")
```