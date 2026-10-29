The repetition direction has a split-half cosine of 0.626 and a 0.393 correlation with residualized repetition, stronger than the earlier correctness associations. About 92% of its energy lies in the output space accessible to head 2. I’ll remove only that single output mode, preserving the head’s other directions and all other heads. If repetition falls without the large accuracy loss caused by halving the whole head, that would support the narrower localization.

```python
rank1_delta=torch.from_numpy(np.load("/work/l11h2_repetition_rank1_delta.npy"))
def apply_rank1_edit(strength):
    for name in list(edited()):
        drop_edit(name)
    name="model.layers.11.self_attn.o_proj.weight"
    w=load_original(name)
    w[:,512:768]-=strength*rank1_delta
    save_edit(name,w)
    desc={name:{"head":2,"columns":[512,768],"operation":"subtract strength * fixed repetition rank-one delta",
                "delta_file":"/work/l11h2_repetition_rank1_delta.npy","strength":strength}}
    json.dump(desc,open("/work/current_edit.json","w"))
    return desc
edit_desc=apply_rank1_edit(1.0)
with open("/work/notes.md","a") as f:
    f.write("\n## Pass14 planned\nCandidate D: rank-one repetition-mode removal in L11H2 output block only, strength1.0. "
            "Exact delta saved /work/l11h2_repetition_rank1_delta.npy; W[:,512:768] <- W_original[:,512:768] - delta, all other entries/tensors original. Relative whole-tensor Frobenius change11.34%, relative head change19.74%; not whole-tensor scaling. DEV63,both,n16,max_tokens8192.\n"
            +json.dumps(edit_desc)+"\n")
print("Rank-one candidate saved; edited tensor:",edited())
```
