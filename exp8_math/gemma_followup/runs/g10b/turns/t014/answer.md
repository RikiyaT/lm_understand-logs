The explicit probe batch contains all 64 reference solutions plus eight failure excerpts, totaling 8,985 tokens. The reference examples cover valid arithmetic, constraint handling, and algebraic transformations, providing useful checks against indiscriminate suppression of mathematical activity. I’m transferring the batch in small pieces solely to avoid output truncation; no additional experiment is running yet. The next two inspections will finish preparing the literal request.

```python
print("Reference texts 24–47:")
print(json.dumps(literal_probes[24:48], ensure_ascii=True))
```
