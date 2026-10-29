The highest-activity examples for layer 14 neuron 1594 include runaway arithmetic and repeated “Final Answer” lines, which supports investigating repetition rather than general problem difficulty. However, one high-activity answer is graded correct despite looping, so suppressing repetition is not automatically an accuracy improvement. The teacher-forced batch contains 576 texts and 346,789 tokens, covering all fixed DEV problems without HOLDOUT. I’ll submit that saved batch to obtain token-level evidence; activation concentrated at repeated continuations would strengthen the hypothesis, while activation throughout useful calculations would argue against simple removal.

```forward
{"type":"teacher_forced","model":"original","texts_file":"/work/tf_dev_texts.json","return":{"token_logprobs":true,"topk":0,"hidden_states":[14,15],"mlp_act":[8,14,17]}}
```
