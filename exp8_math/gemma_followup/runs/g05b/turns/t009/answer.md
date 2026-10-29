The localization batch contains 544 texts and 427,720 tokens: every DEV sample plus every DEV reference solution. Only 62 trajectories needed truncation, and no holdout material or edited weights are included. I’ll submit the saved text list so the large batch does not flood the transcript. Token-level activations should show whether the candidate neurons track specific failure behavior or merely common mathematical formatting.

```forward
{"type":"teacher_forced","model":"original","texts_file":"/work/dev_teacher_texts.json","return":{"token_logprobs":true,"topk":8,"hidden_states":[0,1,6,14,18],"mlp_act":[0,1,5,13,17]}}
```
