The tool interpreted the path as 28 separate character strings, so pass 2 did not run the intended localization batch. That was my input-format mistake, and I will not treat its outputs as evidence about math reasoning. Three passes remain, which still allow two full paired DEV tests and a frozen holdout test. I’ll use the valid baseline activations to narrow the hypothesis further by identifying mechanically checkable arithmetic errors and separating them from repetition and formatting. Any resulting edit will need actual generation evidence to support it.

```python
import ast
from fractions import Fraction

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Pass 2 input-format failure (consumed)\n"
        "- texts_file was rejected without cost. Passing a path in texts was "
        "incorrectly interpreted as 28 one-character texts, consuming pass 2.\n"
        "- Intended 544-text localization batch was NOT run.\n"
        "- No token-local math claims will be made from fwd_2 outputs.\n"
        "- Remaining plan: two full paired DEV tests at different strengths, "
        "then freeze and test HOLDOUT.\n"
    )

def safe_number(expr):
    expr = expr.strip().rstrip(".,")
    if not expr or len(expr) > 180 or not re.fullmatch(r"[0-9\s.+*/()\-]+", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        def rec(node, depth=0):
            if depth > 18:
                raise ValueError()
            if isinstance(node, ast.Expression):
                return rec(node.body, depth+1)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                if abs(node.value) > 1e10:
                    raise ValueError()
                return Fraction(str(node.value))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                val = rec(node.operand, depth+1)
                return -val if isinstance(node.op, ast.USub) else val
            if isinstance(node, ast.BinOp):
                a, b = rec(node.left, depth+1), rec(node.right, depth+1)
                if isinstance(node.op, ast.Add): v = a+b
                elif isinstance(node.op, ast.Sub): v = a-b
                elif isinstance(node.op, ast.Mult): v = a*b
                elif isinstance(node.op, ast.Div): v = a/b
                elif isinstance(node.op, ast.Pow):
                    if b.denominator != 1 or abs(b) > 8: raise ValueError()
                    v = a ** int(b)
                else: raise ValueError()
                if abs(v.numerator) > 10**30 or v.denominator > 10**30:
                    raise ValueError()
                return v
            raise ValueError()
        return rec(tree)
    except Exception:
        return None

def numeric_equalities(text):
    spans = re.findall(r"\$\$([\s\S]*?)\$\$|\$([^$\n]+)\$", text)
    checked = []
    for pair in spans:
        s = pair[0] or pair[1]
        s = re.sub(r"\\(?:left|right|displaystyle|,|!|;)", "", s)
        s = s.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\div", "/")
        for _ in range(5):
            s2 = re.sub(r"\\(?:dfrac|frac)\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
            if s2 == s: break
            s = s2
        s = s.replace("^", "**").replace("{", "(").replace("}", ")")
        parts = s.split("=")
        for a, b in zip(parts, parts[1:]):
            av, bv = safe_number(a), safe_number(b)
            if av is not None and bv is not None:
                identical = re.sub(r"\s", "", a) == re.sub(r"\s", "", b)
                checked.append((av == bv, identical, a.strip()[:70], b.strip()[:70]))
    return checked

feature_rows = []
for r in dev_records:
    t = r["text"]
    eqs = numeric_equalities(t)
    bad = sum(not e[0] for e in eqs)
    progress = sum(e[0] and not e[1] for e in eqs)
    identities = sum(e[1] for e in eqs)
    L = max(1, len(t))
    feature_rows.append({
        "arithmetic_error": bad / max(1, bad + progress),
        "arithmetic_error_count": math.log1p(bad),
        "arithmetic_progress": progress / max(1, bad + progress + identities),
        "identity_fraction": identities / max(1, len(eqs)),
        "repetition": r["repeat4"],
        "equality_density": t.count("=") / L,
        "digit_density": sum(c.isdigit() for c in t) / L,
        "latex_density": t.count("\\") / L,
        "final_density": r["final_markers"] / max(1, r["tokens"]),
        "let_density": len(re.findall(r"\b[Ll]et\b", t)) / max(1, r["tokens"]),
        "therefore_density": len(re.findall(r"\b[Tt]herefore\b", t)) / max(1, r["tokens"]),
    })
feature_names = list(feature_rows[0])
F = np.array([[r[k] for k in feature_names] for r in feature_rows])
Fc = within_problem(F)
Fl = Fc - np.outer(loglen, loglen @ Fc / (loglen @ loglen))
Fnorm = np.sqrt(np.sum(Fl * Fl, axis=0) + 1e-20)

print("Mechanical arithmetic checks:",
      sum(len(numeric_equalities(r["text"])) for r in dev_records),
      "adjacent literal-number equalities")
for good in [True, False]:
    mask = np.array([r["correct"] == good for r in dev_records])
    print("Correct" if good else "Failed",
          "responses with checked arithmetic error:",
          int(np.sum(F[mask, feature_names.index("arithmetic_error_count")] > 0)),
          "/", int(mask.sum()))

component_profiles = {}
for layer in [0,1,5,13,17]:
    X = np.asarray(mlp_mean[dev_indices, layer, :], dtype=np.float64)
    Xc = within_problem(X)
    Xl = Xc - np.outer(loglen, loglen @ Xc / (loglen @ loglen))
    corrs = Xl.T @ Fl / (
        np.sqrt(np.sum(Xl*Xl, axis=0) + 1e-20)[:,None] * Fnorm[None,:]
    )
    for s in neuron_screen:
        if s["layer"] == layer:
            s["feature_corrs"] = dict(zip(feature_names, map(float, corrs[s["neuron"]])))
    candidates = sorted(
        [s for s in stable if s["layer"] == layer],
        key=lambda s: s["robust_score"], reverse=True
    )[:6]
    for s in candidates:
        j = s["neuron"]
        profile = sorted(
            s["feature_corrs"].items(), key=lambda x: abs(x[1]), reverse=True
        )[:4]
        print("L", layer, "N", j,
              "mean/sd", round(s["mean"],3), round(s["sd"],3),
              "top feature correlations", [(k,round(v,3)) for k,v in profile])

arith_candidates = [
    s for s in stable
    if "feature_corrs" in s and s["feature_corrs"]["arithmetic_error"] > .06
]
arith_candidates.sort(
    key=lambda s: s["robust_score"] * s["feature_corrs"]["arithmetic_error"],
    reverse=True
)
print("Stable failure + arithmetic-error candidates:")
for s in arith_candidates[:16]:
    print("L", s["layer"], "N", s["neuron"],
          "failure", round(s["failure_corr"],3),
          "arith-error", round(s["feature_corrs"]["arithmetic_error"],3),
          "energy", round(s["energy"],4),
          "mean/sd", round(s["mean"],3), round(s["sd"],3))
print("Weights remain unchanged.")
```
