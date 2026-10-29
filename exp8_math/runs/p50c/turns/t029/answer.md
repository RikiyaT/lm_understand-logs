Removing the repetition direction from layer 19’s MLP improved accuracy by 9.4 percentage points, but the 6.9-point standard error makes that result inconclusive. It also reduced cutoffs, consistent with the hypothesis, without producing finished wrong answers on this sample. I will now compare the components that write this direction and test a stronger intervention at layers 15 and 19, where the recorded activations showed substantial amplification. A larger accuracy gain—not just shorter answers—would support the proposed mechanism.

```python
# Hypothesis: layers 15 and 19 jointly amplify a distributed repetition feature, so a stronger two-layer intervention should produce a measurable dose-response.
print("Full-attention head alignment with repetition direction:")
headstats=[]
for l in [3,7,11,15,19]:
    w=load_original(f"model.language_model.layers.{l}.self_attn.o_proj.weight")
    d=torch.from_numpy(diffH[l+1].copy());d/=d.norm()
    dm=(meansM[l][1::2]-meansM[l][::2]).mean(0)
    da=torch.from_numpy(diffH[l+1]-diffH[l])-downweights[l]@torch.from_numpy(dm)
    # Attribute the recorded attention contrast via the output projection's regularized minimum-norm decomposition.
    cov=w@w.T
    coeff=w.T@torch.linalg.solve(cov+0.01*torch.eye(1024),da)
    scores=[]
    for h in range(8):
        wh=w[:,256*h:256*(h+1)]
        alignment=float((d@wh).norm()/wh.norm())
        contribution=float((d@wh)@coeff[256*h:256*(h+1)])
        scores.append((h,round(alignment,4),round(contribution,4)))
        headstats.append((contribution,l,h,alignment))
    print(l,scores)
print("Two-layer directional reflection:",mlp_direction_edit([15,19],2.0))
with open("/work/notes.md","a") as f:
    f.write("Pass6 gen_8: L19 MLP direction removal strength1, paired32 +9.38pp SE6.90 (4->7 correct), cut90.6->78.1%. Next: matched repetition-direction reflection (strength2) in MLP layers15+19, the two strongest measured amplification sites. Not a global tensor scaling.\n")
```
