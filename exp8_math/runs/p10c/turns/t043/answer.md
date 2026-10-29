```python
# Hypothesis: deleting the four candidate channels causally interrupts loops and improves accuracy on the same sampled problems.
s4o = [s for r in a4 for s in r["samples"]]
s4e = [s for r in b4 for s in r["samples"]]
summary4 = paired_summary(s4o,s4e)
print("All:", summary4)
for src in sorted({r["source"] for r in a4}):
    print(src, paired_summary([s for r in a4 if r["source"]==src for s in r["samples"]],
                              [s for r in b4 if r["source"]==src for s in r["samples"]]))
for label,ss in [("original",s4o),("edit_strength_1",s4e)]:
    frac = [repeated_fraction(tok.encode(s["text"]).ids[-2048:]) for s in ss]
    print(label, "tail repetitive fraction mean/median:", np.round([np.mean(frac),np.median(frac)],4))
for i,(ra,rb) in enumerate(zip(a4,b4)):
    sa,sb = ra["samples"][0],rb["samples"][0]
    if sa["correct"] != sb["correct"] or (sa["cut"] and not sb["cut"]):
        print("Changed:", ra["id"], "correct",sa["correct"],"->",sb["correct"],"tokens",sa["tokens"],"->",sb["tokens"],
              "edited tail:",sb["text"][-250:].encode("ascii","replace").decode())
print("Activation index excerpt:",open("/work/gen_4_edited_acts/index.json").read()[:1800])
with open("/work/notes.md","a") as f:
    f.write("\nPass 3 (gen_4): ablated four down_proj columns: L18:8, L19:11, L23:22,40 (strength 1). Paired 64 problems. "+str(summary4)+"\n")
```
