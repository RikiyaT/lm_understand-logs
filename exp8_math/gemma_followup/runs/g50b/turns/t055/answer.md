Doubling the direction gained 1.46 points rather than causing the predicted accuracy loss, so the “harmful direction” interpretation is not supported. I will stop treating that correlation as a repair target and examine the strongest length-matched MLP association instead. Layer-11 neuron 971 ranked highest, but I will check whether a small group shows consistent differences across problems and where those units activate. A localized intervention must then change accuracy in the predicted direction before it can displace the saved candidate.

```python
li=idx["mlp_layers"].index(11)
arr=np.asarray(ma[si,li,:],np.float32)
sd=arr.std(axis=0)+1e-6
diff=np.asarray(ma[mi,li,:],np.float32)-np.asarray(ma[mj,li,:],np.float32)
pids=[samples[i]["item"] for i in mi]
pcontrast=np.stack([diff[np.array(pids)==pid].mean(axis=0)
                    for pid in sorted(set(pids))])
stability=(pcontrast>0).mean(axis=0)
norms=load_original("model.layers.11.mlp.down_proj.weight").norm(dim=0).numpy()
ranking=np.argsort(-layer_scores[11])
robust11=[int(u) for u in ranking if stability[u]>=0.625][:8]
print("LAYER11 ROBUST GROUP",
      [{"unit":u,"effect":round(float(layer_scores[11][u]),3),
        "positive_problems":int(round(stability[u]*len(pcontrast))),
        "output_norm":round(float(norms[u]),2)} for u in robust11])
(W/"layer11_matched_failure_units.json").write_text(json.dumps(robust11))
act11=np.load(W/"fwd_2"/"mlp_act_11.npy",mmap_mode="r")
print("REFERENCE ACT SHAPE",act11.shape)
for unit in robust11[:3]:
    peaks=[]
    for i,t in enumerate(compact):
        e=tok.encode(t,add_special_tokens=False)
        vals=np.abs(np.asarray(act11[i,:len(e.ids),unit],np.float32))
        j=int(vals.argmax()); a,b=e.offsets[j]
        peaks.append((float(vals[j]),i,t[max(0,a-55):min(len(t),b+55)]))
    top=max(peaks)
    print("UNIT CONTEXT",unit,"peak",round(top[0],3),
          "context",top[2].encode("ascii","backslashreplace").decode())
print("PASS21",summarize_pair(21))
print("FULL REFERENCE JSON PART1",json.dumps(refs[:16],ensure_ascii=True))
with (W/"notes.md").open("a") as f:
    f.write("\nPass 21 direction doubling: +1.46 pp, SE 1.55 pp; "
            "does NOT support the harmful-direction hypothesis. "
            "New candidate group layer11_matched_failure_units.json selected "
            "by within-problem length-matched standardized activation contrast, "
            "positive sign in >=20/32 problem means, top 8. "
            "Upcoming pass 22 ORIGINAL teacher-forcing all 32 full "
            "DEV_reference_texts.json, token logprobs and MLP layer 11. "
            "Currently saved but UNUSED edit: "+json.dumps(manifest21)+".\n")
```