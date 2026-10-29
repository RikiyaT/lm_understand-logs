```python
# Hypothesis: layer 19's loop-writing component has a dose-dependent effect; doubling directional suppression should further reduce exact repetition without globally ending reasoning.
print("Edit strength 2:", mlp_direction_edit([19],2.0))
def read_gen(path):
    return [json.loads(s) for s in open(path)]
def gen_stats(path):
    rr=read_gen(path)
    ss=[s for r in rr for s in r["samples"]]
    repeats=[]
    thinklens=[]
    for s in ss:
        text=s["text"]; words=text.split()
        ng=[tuple(words[i:i+8]) for i in range(max(0,len(words)-7))]
        repeats.append(1-len(set(ng))/max(1,len(ng)))
        thinklens.append(text.find("</think>"))
    return dict(n=len(ss),accuracy=np.mean([s["correct"] for s in ss]),cut=np.mean([s["cut"] for s in ss]),
                mean_tokens=np.mean([s["tokens"] for s in ss]),mean_repeated_8gram=np.mean(repeats),
                has_think_end=np.mean(np.array(thinklens)>=0))
print("Original metrics",gen_stats("/work/gen_10_original.jsonl"))
print("Direction1 metrics",gen_stats("/work/gen_10_edited.jsonl"))
with open("/work/notes.md","a") as f:
    f.write("Reported pass8 gen_10: L19 down_proj rank1 loop-direction subtraction strength1: paired32 original4/32 edited9/32, +15.62pp SE6.52; cuts90.6->68.75%, tokens30123->24480. Neuron-only weaker. Now testing strength2 on same diagnostic32. Tool reports42 passes left.\n")
```
