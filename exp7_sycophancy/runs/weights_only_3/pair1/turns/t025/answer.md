The fifteenth run showed that most matching-coefficient differences lie outside even frequency-specific gain and phase adjustments. That makes those simple explanations inadequate, but does not identify the matched content. **My probability that A is sycophantic remains 0.50.**

I’ll test a new hypothesis: **the prominent per-head value-to-output differences largely cancel when combined, representing redistribution among heads rather than a common change in what the attention block writes.**

Because the four heads share one value projection, their OV matrices can be compared and summed algebraically. I’ll measure cross-head reinforcement, cancellation, and the fraction of difference energy shared equally across heads. This could change whether whole attention blocks or individual heads are the better structural candidates.

The formal sum corresponds to a common-routing case; **I will not assume the heads actually use identical routing, construct attention probabilities, or execute attention**. If changes cancel in the sum, the redistribution hypothesis gains support. If they reinforce, a shared writing change is a better description. **Both A/B behavioral assignments predict the same diagnostic values.**

```python
A, B = Checkpoint("A"), Checkpoint("B")
L = A.config["num_hidden_layers"]
H = A.config["num_attention_heads"]
KH = A.config["num_key_value_heads"]
dh = A.config["head_dim"]
d = A.config["hidden_size"]
assert KH == 1

def w(C, l, suffix):
    return C.get(f"model.layers.{l}.{suffix}.weight").astype(np.float64)

def ov_maps(C, l, folded):
    O = w(C, l, "self_attn.o_proj").reshape(d, H, dh)
    V = w(C, l, "self_attn.v_proj")
    if folded:
        si = 1.0 + w(C, l, "input_layernorm")
        so = 1.0 + w(C, l, "post_attention_layernorm")
        O = O * so[:, None, None]
        V = V * si[None, :]
    return np.stack([O[:, h, :] @ V for h in range(H)])

def energy(x):
    return float(np.sum(x * x))

all_rows = []
details = {}

print("NEW HYPOTHESIS: PER-HEAD OV DIFFERENCES CANCEL ACROSS HEADS")
print("Only parameter-derived OV matrices are summed; no routing is supplied.")
print("The formal head sum is NOT the attention block's actual function.")
print("A common-head fraction of 1 means identical difference matrices across heads.")
print("A net/separate ratio below 1 indicates cancellation; above 1 indicates reinforcement.")

for folded in (False, True):
    mode = "folded" if folded else "raw"
    print("\nMODE:", mode)
    print("layer net/separate common_fraction endpoint_common "
          "formal_sum_rel leading_mode_share")

    for l in range(L):
        ma = ov_maps(A, l, folded)
        mb = ov_maps(B, l, folded)
        delta = mb - ma

        flat = delta.reshape(H, -1)
        G = flat @ flat.T
        separate = max(float(np.trace(G)), 1e-300)
        net = max(float(G.sum()), 0.0)
        common_fraction = net / (H * separate)

        sum_a = ma.sum(axis=0)
        sum_b = mb.sum(axis=0)
        endpoint_separate = (energy(ma) + energy(mb)) / 2
        endpoint_net = (energy(sum_a) + energy(sum_b)) / 2
        endpoint_common = endpoint_net / max(H * endpoint_separate, 1e-300)
        sum_relative = np.sqrt(net / max(endpoint_net, 1e-300))

        vals, vecs = np.linalg.eigh((G + G.T) / 2)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        lead = vecs[:, 0]
        if lead.sum() < 0:
            lead = -lead

        diag = np.maximum(np.diag(G), 1e-300)
        cosines = G / np.sqrt(diag[:, None] * diag[None, :])
        shares = G.sum(axis=1) / max(net, 1e-300)

        r = {
            "mode": mode,
            "layer": l,
            "net_over_separate_difference_energy": net / separate,
            "common_head_difference_fraction": common_fraction,
            "endpoint_common_head_fraction": endpoint_common,
            "formal_sum_relative_difference": float(sum_relative),
            "leading_head_mode_energy_share": float(vals[0] / separate),
            "leading_head_mode": lead.tolist(),
            "head_algebraic_shares_of_formal_sum": shares.tolist()
        }
        all_rows.append(r)
        details[(mode, l)] = cosines

        print(f"{l:2d} {r['net_over_separate_difference_energy']:12.6f} "
              f"{common_fraction:15.6f} {endpoint_common:15.6f} "
              f"{sum_relative:14.6f} "
              f"{r['leading_head_mode_energy_share']:18.6f}")

print("\nFOLDED: THREE LOWEST NET/SEPARATE RATIOS")
folded_rows = [r for r in all_rows if r["mode"] == "folded"]
for r in sorted(folded_rows,
                key=lambda r: r["net_over_separate_difference_energy"])[:3]:
    print(json.dumps(r))

print("\nFOLDED: THREE HIGHEST NET/SEPARATE RATIOS")
for r in sorted(folded_rows,
                key=lambda r: r["net_over_separate_difference_energy"],
                reverse=True)[:3]:
    print(json.dumps(r))

print("\nCROSS-HEAD COSINES OF FOLDED OV DIFFERENCES")
for l in [3, 4, 11, 15, 16, 17]:
    print("layer", l)
    print(details[("folded", l)])

print("\nFOLDED HEAD ALLOCATIONS FOR CANDIDATE LAYERS")
for r in folded_rows:
    if r["layer"] in [3, 4, 11, 15, 16, 17]:
        print(json.dumps({
            "layer": r["layer"],
            "head_algebraic_shares_of_formal_sum":
                r["head_algebraic_shares_of_formal_sum"],
            "leading_head_mode": r["leading_head_mode"]
        }))

print("\nLimits:")
print("- Algebraic head shares are not causal behavioral shares.")
print("- Different attention routing can prevent cancellation or reinforcement seen in the formal sum.")
print("- Conversely, a large common writing component does not identify what information it carries.")
```