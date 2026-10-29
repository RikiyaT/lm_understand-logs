```python
import math
import time
import collections

A, B = Checkpoint("A"), Checkpoint("B")
L = A.config["num_hidden_layers"]
N = A.config["hidden_size"]
specs = [
    ("q", "self_attn.q_proj.weight", "read"),
    ("k", "self_attn.k_proj.weight", "read"),
    ("v", "self_attn.v_proj.weight", "read"),
    ("o", "self_attn.o_proj.weight", "write"),
    ("gate", "mlp.gate_proj.weight", "read"),
    ("up", "mlp.up_proj.weight", "read"),
    ("down", "mlp.down_proj.weight", "write"),
]

records = []
shared = {"read": np.zeros((N,N)), "write": np.zeros((N,N))}
started = time.monotonic()

for l in range(L):
    for short, suffix, kind in specs:
        name = f"model.layers.{l}.{suffix}"
        a = A.get(name).astype(np.float64)
        b = B.get(name).astype(np.float64)
        d = b-a
        midpoint = (a+b)/2
        # Orient the stored-weight difference so residual coordinates are columns.
        x = d if kind == "read" else d.T
        gram = x.T @ x
        energy = float(np.trace(gram))
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        ev = np.maximum(eigenvalues[order], 0)
        directions = eigenvectors[:, order[:8]]
        normalized = gram/max(energy, 1e-30)
        shared[kind] += normalized
        cumulative = np.cumsum(ev)/max(energy, 1e-30)
        midpoint_energy = float(np.sum(midpoint*midpoint))
        radial_cos = float(np.sum(d*midpoint)) / max(
            math.sqrt(energy*midpoint_energy), 1e-30
        )
        records.append(dict(
            layer=l, short=short, kind=kind,
            energy=energy,
            top1=float(cumulative[0]),
            top4=float(cumulative[3]),
            top16=float(cumulative[min(15, len(cumulative)-1)]),
            stable_rank=energy/max(float(ev[0]), 1e-30),
            rank90=int(np.searchsorted(cumulative, 0.9)+1),
            radial_cos=radial_cos,
            directions=directions,
            fractions=ev[:8]/max(energy, 1e-30),
        ))
    if l in [5, 11, 17]:
        print(f"Processed layers 0..{l}; elapsed {time.monotonic()-started:.1f}s",
              flush=True)

print("\n=== RAW DIFFERENCE: TOP-4 SINGULAR-DIRECTION ENERGY FRACTION ===")
print("layer       q       k       v       o    gate      up    down")
for l in range(L):
    by = {r["short"]: r for r in records if r["layer"] == l}
    print(f"{l:2d}", *[f"{by[s]['top4']:7.3%}" for s, _, _ in specs])

print("\n=== FAMILY SPECTRAL SUMMARY ===")
print("family median_top1 median_top16 median_rank90 median_radial_cos")
for s, _, _ in specs:
    rr = [r for r in records if r["short"] == s]
    print(f"{s:5s}",
          f"{np.median([r['top1'] for r in rr]):.3%}",
          f"{np.median([r['top16'] for r in rr]):.3%}",
          f"{np.median([r['rank90'] for r in rr]):.1f}",
          f"{np.median([r['radial_cos'] for r in rr]):+.6f}")

print("\n=== TOP 18 MATRICES BY FRACTION IN LEADING DIFFERENCE DIRECTION ===")
for r in sorted(records, key=lambda r: r["top1"], reverse=True)[:18]:
    v = r["directions"][:, 0]
    ix = np.argsort(v*v)[-3:][::-1]
    print(f"L{r['layer']:02d}/{r['short']:4s} "
          f"top1={r['top1']:.3%} top4={r['top4']:.3%} "
          f"top16={r['top16']:.3%} rank90={r['rank90']} "
          f"radial_cos={r['radial_cos']:+.5f} "
          f"top_residual_coords={[(int(i), round(float(v[i]**2),4)) for i in ix]}")

shared_directions = {}
print("\n=== CROSS-LAYER SHARED CHANGE GEOMETRY ===")
for kind in ["read", "write"]:
    s = shared[kind]
    ev, vec = np.linalg.eigh(s)
    ix = np.argsort(ev)[::-1]
    ev, vec = ev[ix], vec[:, ix]
    frac = ev/max(float(ev.sum()), 1e-30)
    shared_directions[kind] = vec[:, :8]
    print(f"{kind}: leading 8 energy fractions={np.round(frac[:8],6)} "
          f"top16={frac[:16].sum():.3%}")
    for j in range(3):
        coords = np.argsort(vec[:,j]**2)[-4:][::-1]
        print(f"  direction{j}: "
              f"{[(int(i), round(float(vec[i,j]**2),4)) for i in coords]}")

overlap = shared_directions["read"][:, :4].T @ shared_directions["write"][:, :4]
print("Shared read/write top-4 subspace overlap:",
      float(np.sum(overlap*overlap)/4))
print("For orientation only, independent uniformly oriented 4D subspaces")
print(f"would have expected overlap 4/{N}={4/N:.6f}; that is NOT an assumed null model.")

print("\n=== STRONGEST ORDER-COMPATIBLE WRITE/READ CHANGE-DIRECTION ALIGNMENTS ===")
pairs = []
writers = [r for r in records if r["kind"] == "write"]
readers = [r for r in records if r["kind"] == "read"]
for w in writers:
    for r in readers:
        # Same-layer attention output precedes that layer's MLP.
        valid_same = (
            w["layer"] == r["layer"] and
            w["short"] == "o" and r["short"] in ["gate", "up"]
        )
        if not (w["layer"] < r["layer"] or valid_same):
            continue
        dots = w["directions"][:, :4].T @ r["directions"][:, :4]
        overlap = float(np.sum(dots*dots)/4)
        pairs.append((overlap, w, r))

for overlap, w, r in sorted(pairs, key=lambda x: x[0], reverse=True)[:20]:
    print(f"L{w['layer']:02d}/{w['short']} -> L{r['layer']:02d}/{r['short']} "
          f"subspace_overlap={overlap:.5f} "
          f"writer_top4_energy={w['top4']:.3%} "
          f"reader_top4_energy={r['top4']:.3%}")

print(f"\nElapsed: {time.monotonic()-started:.1f}s")
print("SVD/eigendecomposition was applied only to stored-weight differences.")
print("Direction alignment does not show that a message activates those directions,")
print("nor that intermediate nonlinear blocks preserve them.")
```