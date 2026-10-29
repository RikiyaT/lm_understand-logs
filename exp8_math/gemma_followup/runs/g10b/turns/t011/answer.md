The tool rejected `texts_file`, so no forward-pass budget was spent and there is no new model evidence. I’ll try supplying the saved batch through the required `texts` field. The batch and localization hypothesis are unchanged. If the interface does not accept a saved-file reference there, I’ll use an explicit list of raw texts instead.

```forward
{"type":"teacher_forced","model":"original","texts":"/work/tf_dev_texts.json","return":{"token_logprobs":true,"topk":0,"hidden_states":[14,15],"mlp_act":[8,14,17]}}
```
