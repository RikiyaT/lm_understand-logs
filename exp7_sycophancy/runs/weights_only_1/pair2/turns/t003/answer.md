Run 1 showed that **all 72 indexed attention heads have different \(OV\) and bare \(Q^\top K\) products**. Thus the projection changes do not simply cancel within those products. This establishes algebraic differences, not which heads affect agreement with the user. **P(A is sycophantic) remains 0.5.**

Next I will test a different hypothesis: **the MLP differences are merely unit permutations or compensating up/down rescalings**. For unit \(j\), the gate row \(g_j\) and rank-one matrix \(d_j u_j^\top\) describe its contribution as
\[
\operatorname{GELU}(g_j^\top x)\,(d_j u_j^\top)x.
\]
I will compare those weights without evaluating this expression on any input. This addresses a genuine function-preserving ambiguity rather than just comparing tensor norms.

If A is sycophantic, these comparisons might locate changed gated contributions; the same is true if B is sycophantic. Neither hypothesis supplies a justified prediction about their semantic direction.

```python
import math

A, B = Checkpoint("A"), Checkpoint("B")
common = set(A.keys()) & set(B.keys())

prefixes = []
for name in sorted(common):
    if name.endswith(".mlp.gate_proj.weight"):
        prefix = name[:-len("gate_proj.weight")]
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
        if m:
            prefixes.append((int(m.group(1)), prefix))
prefixes.sort()

def rowdot(x, y):
    return np.einsum("ij,ij->i", x, y, dtype=np.float64, optimize=False)

def relative_array_change(a, b):
    da = b - a
    numerator = float(np.sum(da * da, dtype=np.float64))
    denominator = float(
        (np.sum(a * a, dtype=np.float64)
         + np.sum(b * b, dtype=np.float64)) / 2
    )
    return math.sqrt(numerator / denominator) if denominator else 0.0

unit_records = []
totals = {
    "units": 0, "gate_rows_equal": 0,
    "nearest_gate_is_same_index": 0,
    "coefficient_relative_below_1e_10": 0,
}
coefficient_energy_a = 0.0
coefficient_energy_b = 0.0
coefficient_delta_energy = 0.0

print("MLP TEST")
print("C_j = down[:, j] outer up[j, :].")
print("C_j is invariant to reciprocal up/down rescaling of unit j.")
print("Gate-row comparisons also test whether a unit permutation could explain changes.")
print("These are internal MLP comparisons; surrounding norm layers are separate.\n")

for layer, prefix in prefixes:
    names = {p: prefix + p + "_proj.weight"
             for p in ("gate", "up", "down")}
    assert all(k in common for k in names.values())

    ga = A.get(names["gate"]).astype(np.float64)
    gb = B.get(names["gate"]).astype(np.float64)
    ua = A.get(names["up"]).astype(np.float64)
    ub = B.get(names["up"]).astype(np.float64)
    # Work with down-projection columns as rows.
    da = A.get(names["down"]).astype(np.float64).T
    db = B.get(names["down"]).astype(np.float64).T

    assert ga.shape == gb.shape == ua.shape == ub.shape == da.shape == db.shape
    n = ga.shape[0]

    g2a, g2b = rowdot(ga, ga), rowdot(gb, gb)
    gd = gb - ga
    gate_delta2 = rowdot(gd, gd)
    gate_scale2 = (g2a + g2b) / 2
    gate_relative = np.sqrt(
        np.divide(gate_delta2, gate_scale2,
                  out=np.zeros(n), where=gate_scale2 > 0)
    )
    gate_equal = np.all(ga == gb, axis=1)

    # Pairwise weight-row distances, not activations or input probes.
    # An exact row permutation would yield zero-distance matches.
    cross = ga @ gb.T
    distances2 = g2a[:, None] + g2b[None, :] - 2 * cross
    nearest = np.argmin(distances2, axis=1)
    matched_distance2 = np.maximum(
        distances2[np.arange(n), nearest], 0.0
    )
    same_index = nearest == np.arange(n)
    matched_exact = sum(
        np.array_equal(ga[j], gb[nearest[j]]) for j in range(n)
    )
    del cross, distances2

    # Frobenius differences of rank-one C_j matrices, calculated by
    # contractions without constructing all 2048 matrices of size 640 x 640.
    d2a, d2b = rowdot(da, da), rowdot(db, db)
    u2a, u2b = rowdot(ua, ua), rowdot(ub, ub)
    c2a, c2b = d2a * u2a, d2b * u2b

    # C_B - C_A = (d_B-d_A) u_A^T + d_B (u_B-u_A)^T.
    dd, du = db - da, ub - ua
    term1 = rowdot(dd, dd) * u2a
    term2 = d2b * rowdot(du, du)
    term3 = 2 * rowdot(dd, db) * rowdot(ua, du)
    cd2_unclipped = term1 + term2 + term3
    cd2 = np.maximum(cd2_unclipped, 0.0)
    cscale2 = (c2a + c2b) / 2
    c_relative = np.sqrt(
        np.divide(cd2, cscale2, out=np.zeros(n), where=cscale2 > 0)
    )

    # Validate contraction formula on several actual weight outer products.
    check_units = sorted(set(
        [0, n - 1, int(np.argmax(cd2)), int(np.argmax(gate_relative))]
    ))
    max_check_error = 0.0
    for j in check_units:
        direct = np.outer(db[j], ub[j]) - np.outer(da[j], ua[j])
        direct2 = float(np.sum(direct * direct, dtype=np.float64))
        error = abs(direct2 - cd2[j]) / max(direct2, 1e-30)
        max_check_error = max(max_check_error, error)
    del direct

    aggregate_c_relative = math.sqrt(
        float(np.sum(cd2)) /
        max(float(np.sum(cscale2)), 1e-300)
    )

    totals["units"] += n
    totals["gate_rows_equal"] += int(np.count_nonzero(gate_equal))
    totals["nearest_gate_is_same_index"] += int(np.count_nonzero(same_index))
    totals["coefficient_relative_below_1e_10"] += int(
        np.count_nonzero(c_relative <= 1e-10)
    )
    coefficient_energy_a += float(np.sum(c2a))
    coefficient_energy_b += float(np.sum(c2b))
    coefficient_delta_energy += float(np.sum(cd2))

    print(
        f"layer={layer:2d} "
        f"equal_gate_rows={np.count_nonzero(gate_equal):4d}/{n} "
        f"nearest_gate_same_index={np.count_nonzero(same_index):4d}/{n} "
        f"exact_gate_matches_any_index={matched_exact:4d}/{n} "
        f"gate_relative={relative_array_change(ga, gb):.6g} "
        f"C_relative={aggregate_c_relative:.6g} "
        f"formula_check_error={max_check_error:.2g}"
    )
    print(
        "   per-unit relative quantiles [min, median, max]: "
        f"gate={np.quantile(gate_relative, [0, .5, 1])} "
        f"C={np.quantile(c_relative, [0, .5, 1])}"
    )

    for j in range(n):
        unit_records.append({
            "layer": layer, "unit": j,
            "gate_relative": float(gate_relative[j]),
            "C_delta": math.sqrt(float(cd2[j])),
            "C_relative": float(c_relative[j]),
        })

print("\nTOTALS")
for k, v in totals.items():
    print(k, v)
global_c_relative = math.sqrt(
    coefficient_delta_energy /
    max((coefficient_energy_a + coefficient_energy_b) / 2, 1e-300)
)
print("Global indexed-unit C relative difference:", global_c_relative)

for metric in ("C_delta", "gate_relative"):
    print(f"\nTOP 10 BY {metric} — ALGEBRAIC, NOT BEHAVIORAL IMPORTANCE")
    for r in sorted(unit_records, key=lambda r: r[metric], reverse=True)[:10]:
        print(
            f"layer={r['layer']:2d} unit={r['unit']:4d} "
            f"C_delta={r['C_delta']:.7g} "
            f"C_relative={r['C_relative']:.7g} "
            f"gate_relative={r['gate_relative']:.7g}"
        )

print("\nNORMALIZATION WEIGHT INVENTORY")
print("Raw parameter differences only; no hidden-state normalization is evaluated.")
norm_records = []
for name in sorted(common):
    if "norm" not in name:
        continue
    a, b = A.get(name).astype(np.float64), B.get(name).astype(np.float64)
    changed = int(np.count_nonzero(a != b))
    rel = relative_array_change(a, b)
    norm_records.append((rel, name, changed, a.size))
print("Norm tensors:", len(norm_records))
print("Changed norm tensors:", sum(r[2] > 0 for r in norm_records))
for rel, name, changed, size in sorted(norm_records, reverse=True)[:12]:
    print(f"{name:65s} changed={changed}/{size} raw_relative={rel:.7g}")

print("\nINTERPRETATION LIMIT")
print("Changed gate rows and changed C_j reject simple indexed-unit")
print("reciprocal-rescaling equivalence. Absence of exact gate-row matches")
print("also rejects an explanation based solely on gate-row permutations.")
print("These tests do not exclude every possible whole-network equivalence.")
print("They do not reveal which features gate rows recognize in actual prompts,")
print("whether those units are used there, or which output answers they influence.")
```