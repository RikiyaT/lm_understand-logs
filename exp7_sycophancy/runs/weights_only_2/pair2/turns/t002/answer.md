The checkpoints have identical architectures, and differences are spread across all 18 layers: about **79.4% of retained parameters differ**. Raw differences are largest in MLP input projections, while relative differences are largest in MLP output projections. Neither observation identifies sycophancy; my probability that A is sycophantic remains **0.50**.

One important complication is that total parameter norms grow enormously in later layers. I’ll check whether learned normalization scales explain that, and whether the differences have concentrated directions rather than being diffuse. This matters before comparing composed pathways. Under either assignment—A sycophantic or B sycophantic—large scales and concentrated changes are possible; these tests have no justified semantic sign.

```python
A, B = Checkpoint("A"), Checkpoint("B")
from collections import Counter

print("STORAGE DTYPES")
print("A:", dict(Counter(v["dtype"] for v in A.info.values())))
print("B:", dict(Counter(v["dtype"] for v in B.info.values())))

norm_names = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
]

print("\nNORMALIZATION PARAMETERS")
print("Each entry: median(A), max_abs(A), ||B-A||, unequal_count")
for l in range(18):
    print("layer", l)
    for suffix in norm_names:
        k = f"model.layers.{l}.{suffix}"
        a, b = A.get(k).astype(np.float64), B.get(k).astype(np.float64)
        print(" ", suffix.replace(".weight", ""),
              f"median={np.median(a):.5g}",
              f"maxabs={np.max(np.abs(a)):.5g}",
              f"delta={np.linalg.norm(b-a):.5g}",
              f"unequal={np.count_nonzero(b-a)}/{a.size}")

print("\nFINAL NORMALIZATION")
for side, C in [("A", A), ("B", B)]:
    a = C.get("model.norm.weight")
    print(side, "quantiles:", np.quantile(a, [0, .01, .1, .5, .9, .99, 1]))

print("\nDIFFERENCE SPECTRA FOR SELECTED LAYERS")
print("stable_rank=||delta||_F^2 / largest_singular_value^2")
print("top1/top8 are fractions of squared difference norm")
suffixes = [
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
]
for l in [0, 4, 11, 14, 17]:
    for suffix in suffixes:
        k = f"model.layers.{l}.{suffix}"
        d = B.get(k).astype(np.float64) - A.get(k).astype(np.float64)
        gram = d @ d.T if d.shape[0] <= d.shape[1] else d.T @ d
        ev = np.maximum(np.linalg.eigvalsh(gram), 0)[::-1]
        total = float(ev.sum())
        if total == 0:
            print(l, suffix, "identical")
            continue
        print(l, suffix.replace(".weight", ""),
              f"F={np.sqrt(total):.5g}",
              f"op={np.sqrt(ev[0]):.5g}",
              f"stable_rank={total/max(ev[0],1e-300):.3f}",
              f"top1={ev[0]/total:.4f}",
              f"top8={ev[:8].sum()/total:.4f}")
```