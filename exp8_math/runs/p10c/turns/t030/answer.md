```python
# Hypothesis: the initial four-neuron attenuation was not causally sufficient; recover its exact strength and paired outcomes before choosing another intervention.
print(Path("/work/notes.md").read_text())
print("Available result folders:", sorted(p.name for p in Path("/work").iterdir()))
print("Relevant kernel variables:", [k for k in globals() if any(s in k.lower() for s in ["strength","dose","target","candidate","pass","trial"])])
for name in edited():
    w=load(name); o=load_original(name)
    js=torch.where((w-o).square().sum(0)>0)[0]
    print(name, [(int(j),float(torch.dot(w[:,j],o[:,j])/o[:,j].square().sum())) for j in js])
```
