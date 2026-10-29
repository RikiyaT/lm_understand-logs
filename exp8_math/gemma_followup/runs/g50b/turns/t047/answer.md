The replication gained only 0.59 points, with a 2.18-point standard error, so the combined edit’s benefit remains uncertain. I will retain it as a fallback, but move from broad neuron suppression to a direction-specific test in the causally important attention head. The direction will come from wrong-versus-correct DEV activations matched within problems and approximately matched in length, not from fitting weights to a loss. If suppressing that direction improves answer likelihood and generated accuracy while the opposite intervention harms them, the mechanism would have stronger support.

```python
print("PASS18",summarize_pair(18))
h=np.load(W/"gen_1_acts"/"hidden_mean_answer.npy",mmap_mode="r")
# Residual after layer 11 (index 11 in the 1-based residual convention);
# this is the input to zero-based transformer layer 11.
# After that transformer layer is residual index 12.
hli=idx["layers"].index(12)
rawdiff=(np.asarray(h[mi,hli,:],np.float32)-np.asarray(h[mj,hli,:],np.float32))
direction=torch.from_numpy(rawdiff.mean(axis=0)).float()
direction=direction/direction.norm()
head_name="model.layers.11.self_attn.o_proj.weight"
O=load_original(head_name)
B=O[:,512:768]
proj=direction[:,None] @ (direction @ B)[None,:]
print("HEAD2 FAILURE-DIRECTION OUTPUT ENERGY FRACTION",
      float(proj.square().sum()/B.square().sum()))
# Check consistency of the contrast across problems, without weight fitting.
bydiff=collections.defaultdict(list)
for row,(i,j) in zip(rawdiff,matched):
    bydiff[samples[i]["item"]].append(row)
means=np.stack([np.mean(v,axis=0) for v in bydiff.values()])
align=means@direction.numpy()
print("DIRECTION ALIGNMENT ACROSS PROBLEMS",
      {"positive":int((align>0).sum()),"n":len(align),
       "median":round(float(np.median(align)),5)})
torch.save({"direction":direction,"head_projection":proj,
            "matched_pairs":matched},W/"head11_2_matched_direction.pt")
compact=json.loads((W/"DEV_compact_reference_texts.json").read_text())
refs=json.loads((W/"DEV_reference_texts.json").read_text())
print("REFERENCE FILES",[(n,type(x).__name__,len(x),len(json.dumps(x)))
                          for n,x in [("compact",compact),("reference",refs)]])
print("REFERENCE EXAMPLE",str(refs[0])[:500])
print("COMPACT JSON PART1",json.dumps(compact[:16],ensure_ascii=True))
with (W/"notes.md").open("a") as f:
    f.write("\nPass 18 replication: +0.59 pp, SE 2.18 pp. "
            "Combined candidate remains provisional. No HOLDOUT access. "
            "Direction-specific head experiment uses 117 within-problem "
            "length-matched original DEV wrong/correct pairs at residual 12; "
            "analytic mean contrast, no loss fitting.\n")
```