The matched failure direction accounts for only 0.27% of head 2’s output-weight energy, so removing it preserves nearly all of the head rather than weakening context retrieval wholesale. Its contrast has the same sign in 25 of 32 DEV problems, though that is still correlational evidence. I will remove just this rank-one output component and first score all 32 fixed compact reference answers against the original teacher-forced baseline. Improved likelihood would justify generation testing; deterioration would suggest the head uses this direction constructively or that the correlation is misleading.

```python
for name in list(edited()):
    drop_edit(name)
alpha_dir=1.0
Oedit=O.clone()
Oedit[:,512:768]-=alpha_dir*proj
save_edit(head_name,Oedit)
manifest_dir={"tensor":head_name,"query_head":2,"columns":[512,768],
              "operation":"B -> B - alpha*d*(d.T@B)",
              "alpha":alpha_dir,
              "direction_file":"/work/head11_2_matched_direction.pt",
              "relative_F_change":float((Oedit-O).norm()/O.norm())}
with (W/"notes.md").open("a") as f:
    f.write("\nUpcoming pass 19: EDITED teacher-forcing all 32 fixed "
            "DEV_compact_reference_texts.json, identical texts/order to pass 2. "
            "Exact edit: "+json.dumps(manifest_dir)+
            ". All other tensors original. This is a likelihood screen, "
            "not a generation-accuracy result. HOLDOUT unused.\n")
print("EDIT",manifest_dir)
print("COMPACT JSON PART2",json.dumps(compact[16:],ensure_ascii=True))
```