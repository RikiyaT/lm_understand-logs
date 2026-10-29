The request was rejected because `texts_file` is not a supported field; no forward pass was consumed. This says nothing about the model or the hypothesis, and the weights remain unchanged. I’ll check whether the required `texts` field accepts the saved corpus path. If it does not, I’ll supply raw texts directly.

```forward
{"type":"teacher_forced","model":"original","texts":"/work/tf_dev_texts.json","return":{"token_logprobs":true,"topk":0,"hidden_states":[5,14,16,18],"mlp_act":[4,13,15,17]}}
```
