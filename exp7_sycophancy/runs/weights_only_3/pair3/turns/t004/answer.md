```python
import math
import time

A, B = Checkpoint("A"), Checkpoint("B")
selected_layers = [0, 1, 3, 4, 6, 7, 10, 14]
started = time.monotonic()

def load_factors(c, l):
    p = f"model.layers.{l}."
    d = c.get(p+"mlp.down_proj.weight").astype(np.float64).T
    g = c.get(p+"mlp.gate_proj.weight").astype(np.float64)
    u = c.get(p+"mlp.up_proj.weight").astype(np.float64)
    pre = 1 + c.get(p+"pre_feedforward_layernorm.weight").astype(np.float64)
    post = 1 + c.get(p+"post_feedforward_layernorm.weight").astype(np.float64)
    return d*post[None, :], g*pre[None, :], u*pre[None, :]

def tensor_gram(x, y, same=False):
    # Entry i,j equals the exact inner product of structural coefficient
    # tensors T_i(x) and T_j(y). No input or activation is involved.
    dx, gx, ux = x
    dy, gy, uy = y
    dd = dx @ dy.T
    term = gx @ gy.T
    term *= ux @ uy.T
    cross = gx @ uy.T
    if same:
        term += cross * cross.T
    else:
        cross *= ux @ gy.T
        term += cross
    dd *= 0.5 * term
    return dd

print("=== EXACT CROSS-UNIT ACCOUNTING FOR GAIN-SCALED COEFFICIENTS ===")
print("These coefficients omit activation functions and input-dependent RMS denominators.")
print("No result below is a nonlinear MLP output or a behavioural effect.")
summaries = []

for l in selected_layers:
    t0 = time.monotonic()
    xa, xb = load_factors(A, l), load_factors(B, l)
    aa = tensor_gram(xa, xa, same=True)
    bb = tensor_gram(xb, xb, same=True)
    ab = tensor_gram(xa, xb)

    total_a = float(aa.sum())
    total_b = float(bb.sum())
    reference_total = (total_a + total_b)/2
    reference_diagonal = (np.diag(aa)+np.diag(bb))/2

    # DeltaGram[i,j] = <T_Bi - T_Ai, T_Bj - T_Aj>.
    delta_gram = aa + bb
    delta_gram -= ab
    delta_gram -= ab.T
    del aa, bb, ab, xa, xb

    diagonal = np.diag(delta_gram).copy()
    total_delta = float(delta_gram.sum())
    independent_delta = float(diagonal.sum())
    signed_overlap = delta_gram.sum(axis=1)
    reference_total = max(reference_total, 1e-30)
    denom = max(total_delta, 1e-30)

    if total_delta < -1e-10 * reference_total:
        raise ArithmeticError("Unexpected negative total coefficient difference.")

    raw_order = np.argsort(diagonal)[::-1]
    overlap_order = np.argsort(signed_overlap)[::-1]
    negative_order = np.argsort(signed_overlap)

    relative_total = math.sqrt(max(total_delta, 0)/reference_total)
    coherence = total_delta / max(independent_delta, 1e-30)
    total_strength_ratio = math.sqrt(max(total_b, 0)/max(total_a, 1e-30))
    neg_count = int(np.sum(signed_overlap < 0))
    negative_fraction = float(
        signed_overlap[signed_overlap < 0].sum()/denom
    )

    print(f"\nLAYER {l:02d}: total_relative_change={relative_total:.7f} "
          f"coherence_ratio={coherence:.5f} "
          f"total_coefficient_strength_B/A={total_strength_ratio:.7f}")
    print(f"  Negative signed-overlap units: {neg_count}/2048; "
          f"their combined signed share={negative_fraction:.3%}")

    print("  Top 8 units by signed overlap with the total coefficient difference:")
    for i in overlap_order[:8]:
        # The change in squared coefficient-difference norm if this unit's
        # coefficient difference is algebraically removed:
        removal_fraction = (2*signed_overlap[i]-diagonal[i])/denom
        print(f"    U{i:04d} signed_overlap_share={signed_overlap[i]/denom:+.3%} "
              f"individual_energy/total={diagonal[i]/denom:.3%} "
              f"coefficient_removal_fraction={removal_fraction:+.3%} "
              f"reference_diagonal_share="
              f"{reference_diagonal[i]/max(reference_diagonal.sum(),1e-30):.3%}")

    print("  Top 3 opposing units:")
    for i in negative_order[:3]:
        print(f"    U{i:04d} signed_overlap_share={signed_overlap[i]/denom:+.3%}")

    print("  Group accounting (ranked by individual coefficient-change energy):")
    for k in [1, 4, 20, 100]:
        ix = raw_order[:k]
        group_energy = float(delta_gram[np.ix_(ix, ix)].sum())
        overlap = float(signed_overlap[ix].sum())
        remaining_energy = total_delta - 2*overlap + group_energy
        print(f"    top{k:3d}: group_energy/total={group_energy/denom:.3%} "
              f"signed_overlap/total={overlap/denom:+.3%} "
              f"remaining_energy/total={remaining_energy/denom:.3%}")

    summaries.append((
        l, relative_total, coherence, total_strength_ratio,
        overlap_order[:8].tolist()
    ))
    print(f"  elapsed_for_layer={time.monotonic()-t0:.1f}s", flush=True)
    del delta_gram
    if time.monotonic()-started > 470:
        print("Stopping before the execution limit; remaining layers were not analyzed.")
        break

print("\n=== SUMMARY: RANKED BY RELATIVE TOTAL COEFFICIENT CHANGE ===")
for l, rel, coh, ratio, units in sorted(summaries, key=lambda x: x[1], reverse=True):
    print(f"L{l:02d} relative={rel:.7f} coherence={coh:.5f} "
          f"B/A_strength={ratio:.7f} leading_units={units}")
print(f"\nTotal elapsed: {time.monotonic()-started:.1f}s")
print("Coefficient-removal quantities are algebraic tensor identities,")
print("not model ablations, evaluations, or causal behavioural measurements.")
```