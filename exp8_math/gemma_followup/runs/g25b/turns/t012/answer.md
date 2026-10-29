The path was mistakenly treated as 23 one-character texts, so pass 2 produced no usable diagnostic evidence. I’ll record that wasted pass and use literal text arrays from now on. To keep requests manageable while covering every DEV problem, I’ll prepare a compact corpus with each exact prompt followed by its boxed reference answer. This measures correct-answer likelihood, not full reasoning quality, so generation accuracy will remain the deciding test.

```python
compact_texts=[]
compact_info=[]
for p in dev_ids:
    r=problems_by_id[p]
    prompt=PREFIX+r["problem"]+SUFFIX
    answer="Final Answer: \\boxed{"+r["answer"]+"}"
    compact_texts.append(prompt+answer)
    compact_info.append(dict(id=p,prompt_tokens=len(tok.encode(prompt).ids),
                             tokens=len(tok.encode(prompt+answer).ids)))
json.dump(compact_texts,open("/work/tf_compact_texts.json","w"))
json.dump(compact_info,open("/work/tf_compact_info.json","w"))
with open("/work/notes.md","a") as f:
    f.write("\nPass 2 wasted: string path in `texts` was iterated as 23 characters; no useful evidence. Only literal text arrays are supported. 23 passes remain.\n"
            "## Pass 3 planned\nOriginal, no changed tensors. All 63 DEV exact prompts followed by boxed reference answers (direct-answer likelihood screen, not a reasoning-accuracy measurement). Files tf_compact_texts.json and tf_compact_info.json. Return logprobs and MLP activations at layers 4,13,15,17.\n")
print("Compact corpus:",len(compact_texts),"texts",sum(x["tokens"] for x in compact_info),"tokens",
      sum(len(t) for t in compact_texts),"characters")
print("First batch (0:21):")
print(json.dumps(compact_texts[:21],ensure_ascii=True))
```
