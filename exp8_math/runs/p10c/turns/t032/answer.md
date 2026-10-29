```python
# Hypothesis: the initial neuron attenuation was too weak or targeted a correlate rather than the causal repetition circuit; recover the experiment record before choosing the next intervention.
from pathlib import Path
import os, json, numpy as np, torch, collections
print("NOTES:")
p=Path("/work/notes.md")
print(p.read_text() if p.exists() else "(no notes)")
print("\nWORK FILES:",sorted(os.listdir("/work")))
print("CURRENT EDITS:",edited())
for name in edited():
    w0=load_original(name); w=load(name); d=w-w0
    cols=torch.where(d.abs().sum(0)>0)[0]
    print(name,"changed columns",cols.tolist()[:30],"relative norm",float(d.norm()/w0.norm()))
    for j in cols[:8]:
        print("column",int(j),"original norm",float(w0[:,j].norm()),"edited/original projection",float((w[:,j]@w0[:,j])/(w0[:,j]@w0[:,j])))
```
