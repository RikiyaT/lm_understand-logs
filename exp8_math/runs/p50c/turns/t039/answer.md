```python
# Hypothesis: repetition is distributed across late MLPs; extending the selective direction removal to its other major amplification sites should yield a larger rescue.
print("Current edits:",edited())
print("Recent generation files:", sorted(Path("/work").glob("gen_*jsonl"))[-12:])
print("Notes tail:",Path("/work/notes.md").read_text()[-1800:])
import shutil
if Path("/work/best_l19").exists(): shutil.rmtree("/work/best_l19")
shutil.copytree("/work/edited","/work/best_l19")
print("Three-layer directional probe:",mlp_direction_edit([15,19,22],1.0))
with open("/work/notes.md","a") as f:
    f.write("Latest reported pass9 gen_11: L19 rank-one removal paired32 accuracy12.5->28.125%, +15.62pp +/-6.52 SE; cut90.6->68.75%. Backed up /work/best_l19. Extend to L15,L19,L22, whose matched repetition contrast grows strongly.\n")
```
