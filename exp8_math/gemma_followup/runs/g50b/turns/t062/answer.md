The eight-unit ablation gained 1.17 points with a 2.12-point standard error, so it has not established an accuracy benefit. I will test the opposite intervention—doubling those same outputs—to distinguish a directional effect from sampling noise. I will then limit the remaining DEV work to dose checks and replication of the leading candidates, reserving the final 16 passes for a frozen HOLDOUT evaluation. If amplification is also benign or beneficial, the failure association will not justify calling these units harmful.

```python
for name in list(edited()): drop_edit(name)
new11=D11.clone(); new11[:,robust11]*=2.0
save_edit(mlp11name,new11)
manifest25={"tensor":mlp11name,"columns":robust11,"multiplier":2.0,
            "relative_F_change":float((new11-D11).norm()/D11.norm())}
log_pass(25,"Opposite-direction causal control for eight length-matched layer11 units",manifest25)
with (W/"notes.md").open("a") as f:
    f.write("\nPlan locked: finish candidate selection and dose/replication DEV tests "
            "by pass34. Freeze one candidate before HOLDOUT. Passes35-50: "
            "four pre-fixed HOLDOUT blocks of32 problems, n32 each, four rounds; "
            "pool all 16384 answers/model with problem-cluster and sample SEs. "
            "No tuning or candidate changes based on HOLDOUT.\n")
print("PASS24",summarize_pair(24))
print("NEXT EDIT",manifest25)
```