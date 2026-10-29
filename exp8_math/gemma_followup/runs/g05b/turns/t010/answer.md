The file-list field was rejected, so no forward pass was spent and there are no new model results. The prepared localization batch and unedited weights remain unchanged. I’ll check whether the required `texts` field accepts the saved JSON path; this is an input-format check, not a model experiment.

```forward
{"type":"teacher_forced","model":"original","texts":"/work/dev_teacher_texts.json","return":{"token_logprobs":true,"topk":8,"hidden_states":[0,1,6,14,18],"mlp_act":[0,1,5,13,17]}}
```
