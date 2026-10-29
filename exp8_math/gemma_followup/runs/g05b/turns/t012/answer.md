Mechanical checks found an invalid numeric equality in 82 of 295 failed DEV responses, versus 27 of 217 correct ones. However, several prominent early-layer candidates correlate more strongly with digit density than with arithmetic errors, so suppressing them could damage number processing rather than repair it. I’ll exclude that shortcut by controlling for numeric density and common formatting as well as length. The next screen will favor later-layer components associated with both failure and invalid arithmetic after those controls. This gives the intervention a narrower prediction: fewer incorrect transformations, with improved final-answer accuracy.

```python
# Statistical residualization only: this does not train or run the model.
control_names = [
    "digit_density", "latex_density", "equality_density",
    "final_density", "let_density", "therefore_density"
]
C = np.column_stack(
    [loglen] + [Fc[:, feature_names.index(k)] for k in control_names]
)
Q, _ = np.linalg.qr(C, mode="reduced")
def remove_controls(x, mask=None):
    if mask is None:
        return x - Q @ (Q.T @ x)
    c = C[mask]
    q, _ = np.linalg.qr(c, mode="reduced")
    z = x[mask]
    return z - q @ (q.T @ z)

f_adj = remove_controls(failure)
a_adj = remove_controls(Fc[:, feature_names.index("arithmetic_error")])
p_adj = remove_controls(Fc[:, feature_names.index("arithmetic_progress")])
i_adj = remove_controls(Fc[:, feature_names.index("identity_fraction")])
r_adj = remove_controls(repeat)
controlled_screen = []

for layer in range(18):
    Xc = within_problem(np.asarray(mlp_mean[dev_indices, layer, :], dtype=np.float64))
    Xa = remove_controls(Xc)
    normx = np.sqrt(np.sum(Xa * Xa, axis=0) + 1e-20)
    def correlations(y):
        return y @ Xa / (np.linalg.norm(y) * normx + 1e-20)
    cf, ca, cp, ci, cr = [correlations(y) for y in [f_adj,a_adj,p_adj,i_adj,r_adj]]
    ch = []
    for half in [0,1]:
        mask = groups_dev % 2 == half
        xx = remove_controls(Xc, mask)
        yy = remove_controls(failure, mask)
        ch.append(yy @ xx / (
            np.linalg.norm(yy) * np.sqrt(np.sum(xx*xx, axis=0) + 1e-20)
        ))
    prior = {s["neuron"]:s for s in neuron_screen if s["layer"] == layer}
    for j in range(2048):
        controlled_screen.append({
            **prior[j],
            "adjusted_failure": float(cf[j]),
            "adjusted_arithmetic_error": float(ca[j]),
            "adjusted_progress": float(cp[j]),
            "adjusted_identity": float(ci[j]),
            "adjusted_repeat": float(cr[j]),
            "adjusted_half0": float(ch[0][j]),
            "adjusted_half1": float(ch[1][j]),
        })

controlled_candidates = [
    s for s in controlled_screen
    if s["layer"] >= 6
    and s["mean"] > 0
    and s["adjusted_failure"] > .10
    and min(s["adjusted_half0"], s["adjusted_half1"]) > .045
    and s["adjusted_arithmetic_error"] > .04
]
controlled_candidates.sort(
    key=lambda s: (
        min(s["adjusted_half0"], s["adjusted_half1"])
        * math.sqrt(max(0, s["adjusted_arithmetic_error"]))
    ), reverse=True
)
print("Late-layer candidates after surface-form controls:", len(controlled_candidates))
for s in controlled_candidates[:24]:
    print({
        k: round(s[k], 3) if isinstance(s[k], float) else s[k]
        for k in [
            "layer","neuron","adjusted_failure","adjusted_arithmetic_error",
            "adjusted_half0","adjusted_half1","adjusted_progress",
            "adjusted_identity","energy"
        ]
    })

print("Top adjusted failure correlations by layer:")
for layer in range(18):
    ss = [s for s in controlled_screen if s["layer"] == layer]
    s = max(ss, key=lambda s:s["adjusted_failure"])
    print(layer, s["neuron"], round(s["adjusted_failure"],3),
          "halves", round(s["adjusted_half0"],3), round(s["adjusted_half1"],3),
          "arith", round(s["adjusted_arithmetic_error"],3))

with open("/work/controlled_neuron_screen.json", "w") as f:
    json.dump(controlled_screen, f)
print("No edit yet.")
```
