```python
# Hypothesis: loop-specific MLP neurons recur across matched successful and failing answers, rather than merely reflecting problem difficulty.
mixed = [(i if cut[i] else i+1, i+1 if cut[i] else i)
         for i in range(0,len(samples1),2) if good[i] != good[i+1]]
pair_delta = np.stack([M[b]-M[g] for b,g in mixed])
delta = pair_delta.mean(0)
consistency = (pair_delta>0).mean(0)
se = pair_delta.std(0)/math.sqrt(len(mixed))
down_norm = np.stack([weights[f"model.language_model.layers.{l}.mlp.down_proj.weight"].norm(dim=0).numpy() for l in range(24)])
score = delta * down_norm
ranked = np.argsort(score.ravel())[::-1]
print("Top paired positive MLP output contributions (not yet causal):")
for k in ranked[:25]:
    l,j=np.unravel_index(k,score.shape)
    print("layer",l,"neuron",j,"delta",round(float(delta[l,j]),4),
          "cut/good",round(float(M[cut,l,j].mean()),4),round(float(M[good,l,j].mean()),4),
          "pair_positive",round(float(consistency[l,j]),2),
          "delta/SE",round(float(delta[l,j]/max(se[l,j],1e-8)),2),
          "out_norm",round(float(down_norm[l,j]),3))
print("\nMatched sample summaries:")
for b,g in mixed:
    bad=samples1[b][text_key]
    lines=[x.strip() for x in bad.splitlines() if x.strip()]
    repeated=collections.Counter(lines).most_common(1)[0]
    first=bad.find(repeated[0])
    print("indices",b,g,"id",samples1[b]["id"],"good_tokens",samples1[g]["tokens"],
          "repeat_count",repeated[1],"first_char",first,"repeat",repr(repeated[0][:100]))
print("\nShort failing prefix around earliest loops for sample 11:")
s=samples1[11]["text"]
for a in [0,1000,2000,3000,4000,5000,6000]:
    print(a,repr(s[a:a+220]))
print("Tokenizer type:",type(tok).__name__)
print("Chat-template support:",hasattr(tok,"apply_chat_template"))
```
