```python
import math

A, B = Checkpoint("A"), Checkpoint("B")
L = A.config["num_hidden_layers"]

def rows(x, y):
    return np.einsum("ij,ij->i", x, y, optimize=False)

def load(c, l):
    p = f"model.layers.{l}."
    d = c.get(p+"mlp.down_proj.weight").astype(np.float64).T
    g = c.get(p+"mlp.gate_proj.weight").astype(np.float64)
    u = c.get(p+"mlp.up_proj.weight").astype(np.float64)
    pre = 1+c.get(p+"pre_feedforward_layernorm.weight").astype(np.float64)
    post = 1+c.get(p+"post_feedforward_layernorm.weight").astype(np.float64)
    raw = (d,g,u)
    scaled = (d*post[None,:], g*pre[None,:], u*pre[None,:])
    return raw, scaled

def coefficient_inner(x, y):
    dx,gx,ux = x
    dy,gy,uy = y
    return 0.5*rows(dx,dy)*(
        rows(gx,gy)*rows(ux,uy) + rows(gx,uy)*rows(ux,gy)
    )

def radial_coefficient_decomposition(x, y):
    # Per-unit structural tensor:
    # T_i = d_i outer sym(g_i outer u_i).
    #
    # Compute projection of delta(T_i) onto midpoint(T_i), using only
    # coefficient inner products. No hidden/input vector is constructed.
    aa = coefficient_inner(x,x)
    bb = coefficient_inner(y,y)
    ab = coefficient_inner(x,y)
    delta_energy = np.maximum(aa+bb-2*ab,0)
    midpoint_energy = np.maximum((aa+bb+2*ab)/4,0)
    delta_dot_midpoint = (bb-aa)/2

    scalar = delta_dot_midpoint/np.maximum(midpoint_energy,1e-300)
    radial_energy = scalar*scalar*midpoint_energy
    error = float(np.max(
        np.maximum(radial_energy-delta_energy,0) /
        np.maximum((aa+bb)/2,1e-300)
    ))
    if error > 1e-10:
        raise ArithmeticError("Unexpected radial projection error.")

    radial_energy = np.minimum(radial_energy,delta_energy)
    perpendicular_energy = delta_energy-radial_energy
    return dict(
        delta=delta_energy,
        radial=radial_energy,
        perpendicular=perpendicular_energy,
        scalar=scalar,
        relative=np.sqrt(delta_energy/np.maximum((aa+bb)/2,1e-300))
    )

def up_down_decomposition(x, y):
    da,_,ua = x
    db,_,ub = y
    dm,um = (da+db)/2,(ua+ub)/2
    dd,du = db-da,ub-ua
    dn,un = rows(dm,dm),rows(um,um)
    ad = rows(dd,dm)/np.maximum(dn,1e-300)
    au = rows(du,um)/np.maximum(un,1e-300)

    # Orthogonal residuals of stored-weight differences.
    dd_perp = dd-ad[:,None]*dm
    du_perp = du-au[:,None]*um
    dp,up = rows(dd_perp,dd_perp),rows(du_perp,du_perp)
    weight = dn*un

    # Exact midpoint identity:
    # delta(d outer u) =
    # (ad+au) dm outer um
    # + dd_perp outer um + dm outer du_perp.
    # The three displayed coefficient terms are mutually orthogonal.
    strength = (ad+au)**2*weight
    down_direction = dp*un
    up_direction = dn*up
    invariant_delta = strength+down_direction+up_direction

    direct_delta = (
        rows(da,da)*rows(ua,ua)
        + rows(db,db)*rows(ub,ub)
        - 2*rows(da,db)*rows(ua,ub)
    )
    reference = 0.5*(
        rows(da,da)*rows(ua,ua)+rows(db,db)*rows(ub,ub)
    )
    reconstruction_error = float(np.max(
        np.abs(invariant_delta-direct_delta)/np.maximum(reference,1e-300)
    ))

    # In the product-weighted metric on the two separate parameter changes,
    # opposite scalar changes form the reciprocal-rescaling direction.
    # This component cancels exactly from delta(d outer u).
    gauge_energy = 0.5*(ad-au)**2*weight
    separate_piece_energy = (
        rows(dd,dd)*un + dn*rows(du,du)
    )
    return dict(
        gauge=gauge_energy,
        separate=separate_piece_energy,
        strength=strength,
        down_direction=down_direction,
        up_direction=up_direction,
        invariant_delta=invariant_delta,
        error=reconstruction_error
    )

selected = {
    0:[1405,1130],
    1:[1725,372],
    3:[1635],
    4:[13,1044,20],
    6:[1824],
    7:[1455],
    10:[1363,1564,1529],
    14:[641],
    17:[1026,1916]
}
details=[]
summary=[]
all_units=[]

print("=== FEATURE-STRENGTH VERSUS DIRECTION CHANGE ===")
print("Fractions sum over individual coefficient tensors, not over their")
print("cross-unit sum. These are structural coefficients, not MLP outputs.")
print("layer raw_T_radial scaled_T_radial raw_UD_gauge "
      "scaled_UD_gauge scaled_UD_strength scaled_UD_down_dir scaled_UD_up_dir")

for l in range(L):
    ra,sa = load(A,l)
    rb,sb = load(B,l)
    tr = radial_coefficient_decomposition(ra,rb)
    ts = radial_coefficient_decomposition(sa,sb)
    ur = up_down_decomposition(ra,rb)
    us = up_down_decomposition(sa,sb)

    def fraction(num,den):
        return float(np.sum(num)/max(float(np.sum(den)),1e-300))

    raw_radial = fraction(tr["radial"],tr["delta"])
    scaled_radial = fraction(ts["radial"],ts["delta"])
    raw_gauge = fraction(ur["gauge"],ur["separate"])
    scaled_gauge = fraction(us["gauge"],us["separate"])
    strength = fraction(us["strength"],us["invariant_delta"])
    down_dir = fraction(us["down_direction"],us["invariant_delta"])
    up_dir = fraction(us["up_direction"],us["invariant_delta"])
    print(f"{l:2d} {raw_radial:9.3%} {scaled_radial:9.3%} "
          f"{raw_gauge:9.3%} {scaled_gauge:9.3%} "
          f"{strength:9.3%} {down_dir:9.3%} {up_dir:9.3%}")

    summary.append((l,raw_radial,scaled_radial,raw_gauge,scaled_gauge))
    for i in range(len(ts["delta"])):
        all_units.append(dict(
            layer=l,unit=i,
            directional_share=float(
                ts["perpendicular"][i]/
                max(float(ts["perpendicular"].sum()),1e-300)
            ),
            total_share=float(
                ts["delta"][i]/max(float(ts["delta"].sum()),1e-300)
            ),
            radial_fraction=float(
                ts["radial"][i]/max(float(ts["delta"][i]),1e-300)
            ),
            scalar=float(ts["scalar"][i])
        ))

    for i in selected.get(l,[]):
        details.append(dict(
            layer=l,unit=i,
            raw_radial=float(tr["radial"][i]/max(float(tr["delta"][i]),1e-300)),
            scaled_radial=float(ts["radial"][i]/max(float(ts["delta"][i]),1e-300)),
            scalar=float(ts["scalar"][i]),
            scaled_rel=float(ts["relative"][i]),
            gauge=float(us["gauge"][i]/max(float(us["separate"][i]),1e-300)),
            ud_strength=float(us["strength"][i]/max(float(us["invariant_delta"][i]),1e-300)),
            ud_down=float(us["down_direction"][i]/max(float(us["invariant_delta"][i]),1e-300)),
            ud_up=float(us["up_direction"][i]/max(float(us["invariant_delta"][i]),1e-300))
        ))

    if max(ur["error"],us["error"]) > 1e-10:
        raise ArithmeticError("Up/down coefficient reconstruction failed.")

print("\n=== PREVIOUS CANDIDATES: SCALAR REWEIGHTING OR DIRECTION CHANGE? ===")
print("Signed scalar is the projection of T_B-T_A onto (T_A+T_B)/2.")
for r in details:
    print(f"L{r['layer']:02d}U{r['unit']:04d} "
          f"T_rel={r['scaled_rel']:.5f} "
          f"raw_radial={r['raw_radial']:.2%} "
          f"scaled_radial={r['scaled_radial']:.2%} "
          f"signed_scalar={r['scalar']:+.6f} "
          f"UD_gauge={r['gauge']:.2%} "
          f"UD_real_change(strength,down_dir,up_dir)="
          f"({r['ud_strength']:.2%},{r['ud_down']:.2%},{r['ud_up']:.2%})")

print("\n=== TOP 16 UNITS BY SHARE OF THEIR LAYER'S DIRECTIONAL T DIFFERENCE ===")
for r in sorted(all_units,key=lambda r:r["directional_share"],reverse=True)[:16]:
    print(f"L{r['layer']:02d}U{r['unit']:04d} "
          f"directional_share={r['directional_share']:.3%} "
          f"original_total_share={r['total_share']:.3%} "
          f"radial_fraction={r['radial_fraction']:.2%} "
          f"signed_scalar={r['scalar']:+.6f}")

print("\nInterpretation limits:")
print("- Reciprocal up/down rescaling is an exact parameter symmetry when gate weights are unchanged.")
print("- The gauge statistic measures that cancelling direction, not proof of complete model equivalence.")
print("- Radial structural-tensor change is not proof of scalar reweighting of the nonlinear MLP function.")
print("- All computations concern stored weights and tensor contractions only; no inputs were evaluated.")
```