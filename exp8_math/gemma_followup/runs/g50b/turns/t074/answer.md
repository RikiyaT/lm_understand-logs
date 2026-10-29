The replication gained 2.93 points, so the 10% head-plus-MLP candidate now has two positive full-size results. I will next remove the MLP edit while keeping head 2 at 1.1× to measure whether the neuron changes actually contribute. I will also prepare a control group matched by output-weight norm and activation RMS but with little failure association. If either control performs as well as the selected group, the claimed neuron specificity must be weakened.

```python
print("LEADING CANDIDATE POOLED",pooled_pairs([29,30]))
manifest31=set_head_direction(alpha=0.0,gain=1.1)
log_pass(31,"Component ablation: retain head2 1.1x, restore all MLP weights",manifest31)
from scipy.optimize import linear_sum_assignment
li11=idx["mlp_layers"].index(11)
arr11=np.asarray(ma[si,li11,:],np.float32)
rms11=np.sqrt(np.mean(arr11**2,axis=0))+1e-8
norm11=D11.norm(dim=0).numpy()
eligible=[u for u in range(2048) if u not in robust11 and abs(layer_scores[11][u])<0.03]
features=np.stack([np.log(norm11+1e-8),np.log(rms11)],axis=1)
scale=np.std(features,axis=0)
cost=((features[np.array(robust11),None,:]-features[None,np.array(eligible),:])/scale)**2
row,col=linear_sum_assignment(cost.sum(axis=2))
control11=[int(eligible[j]) for j in col]
(W/"layer11_matched_control_units.json").write_text(json.dumps(control11))
print("CONTROL MATCHES",
      [{"target":u,"control":v,"target_norm":round(float(norm11[u]),3),
        "control_norm":round(float(norm11[v]),3),
        "RMS_ratio":round(float(rms11[v]/rms11[u]),3),
        "control_effect":round(float(layer_scores[11][v]),3)}
       for u,v in zip(robust11,control11)])
print("Pass31 ready; control group fixed before its outcomes.")
```