The neurons most strongly associated with repetition are mostly weakly associated with incorrectness after length control. That makes a repetition-only repair less promising for accuracy, so I’ll prioritize failure-associated components instead. The teacher-forced corpus is ready: 496 texts totaling 278,520 tokens, including all 63 DEV references. I’ll request the baseline measurements using the saved text file so the entire corpus can be tested together.

```forward
{"type":"teacher_forced","model":"original","texts_file":"/work/tf_dev_texts.json","return":{"token_logprobs":true,"topk":0,"hidden_states":[5,14,16,18],"mlp_act":[4,13,15,17]}}
```
