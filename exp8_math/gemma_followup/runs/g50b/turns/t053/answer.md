Direction removal shortened answers and reduced cutoffs, but its accuracy gain was only 0.49 points with a 1.89-point standard error. This supports, at most, a role in continuation length—not a demonstrated math repair. I will now double the same component as an opposite-direction control, changing nothing else. Longer or more repetitive answers would establish a directional behavioral effect; a substantial accuracy improvement would instead indicate that removing it was the wrong intervention.

```python
def set_head_direction(alpha=0.0,gain=1.0):
    for name in list(edited()):
        drop_edit(name)
    original=load_original(head_name)
    b=original[:,512:768]
    component=direction[:,None]@(direction@b)[None,:]
    new=original.clone()
    new[:,512:768]=gain*(b-alpha*component)
    save_edit(head_name,new)
    return {"tensor":head_name,"head":2,"columns":[512,768],
            "operation":"B -> gain*(B-alpha*d*(d.T@B))",
            "gain":gain,"alpha":alpha,
            "direction_file":"/work/head11_2_matched_direction.pt",
            "relative_F_change":float((new-original).norm()/original.norm())}
def log_pass(k,description,manifest,setname="DEV",n=32):
    with (W/"notes.md").open("a") as f:
        f.write(f"\nUpcoming pass {k}: {description}; BOTH locked {setname}, "
                f"n={n}, max_tokens=8192. Exact edit: "+json.dumps(manifest)+
                ". All unlisted weights original. HOLDOUT unused unless explicitly stated.\n")
def archive_current(label,manifest):
    dest=W/label
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(W/"edited",dest)
    (dest/"manifest.json").write_text(json.dumps(manifest,indent=2))
manifest21=set_head_direction(alpha=-1.0,gain=1.0)
log_pass(21,"Opposite-direction control: double the matched failure-direction component",manifest21)
print("PASS20",summarize_pair(20))
print("NEXT EDIT",manifest21)
```