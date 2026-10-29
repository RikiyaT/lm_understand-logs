The fresh generation request failed at engine startup and produced no results, so the held-out set remains unused. I will move the already planned diagnostic probes ahead of that retry rather than repeatedly launching the failed request. These probes contrast productive calculations with loops across arithmetic, algebra, and modular exponentiation, including a deliberately repetitive procedure that genuinely makes mathematical progress. All texts remain inside thinking, so the measurements will not be driven by closing the thinking section.
```python
# Hypothesis: the frozen copying-head repair suppresses unproductive repetition more than legitimate repeated mathematical procedures.
broad_cases=[
    {
        "name":"shopping",
        "problem":"Maya buys 6 notebooks at 4 dollars each and 3 pens at 2 dollars each. She pays with 50 dollars. How much change does she receive?",
        "common":"Six notebooks cost 6 * 4 = 24 dollars. Three pens cost 3 * 2 = 6 dollars.\n",
        "good":"The total cost is 24 + 6 = 30 dollars. Paying 50 dollars gives 50 - 30 = 20 dollars in change.\nAn independent calculation subtracts the purchases one at a time: 50 - 24 = 26, then 26 - 6 = 20.\nChecking the result, the 30 dollars spent plus the 20 dollars returned equals the original 50 dollars. Both methods agree on 20 dollars.\n",
        "loopline":"Wait, six notebooks cost 24 dollars and three pens cost 6 dollars. The total is 30 dollars. I should subtract the total from 50.\n",
        "repeats":6
    },
    {
        "name":"linear_equation",
        "problem":"Solve the equation 3(2x - 5) = 27 for x.",
        "common":"Expanding the left side gives 6x - 15 = 27.\n",
        "good":"Add 15 to both sides to obtain 6x = 42. Dividing both sides by 6 gives x = 7.\nSubstitute into the original equation: 2 * 7 - 5 = 14 - 5 = 9, and 3 * 9 = 27. This equals the right side.\nAlternatively, divide the original equation by 3 first: 2x - 5 = 9. Then 2x = 14 and x = 7. The two methods give the same unique solution.\n",
        "loopline":"Wait, the equation is 6x - 15 = 27. Adding 15 to both sides should isolate the term 6x.\n",
        "repeats":6
    },
    {
        "name":"modular_progress",
        "problem":"Find the remainder when 3^12 is divided by 7.",
        "common":"I will track the remainder after each multiplication by 3, reducing modulo 7 each time. The starting remainder for exponent 0 is 1.\n",
        "good":"Exponent 1: 1 * 3 = 3, remainder 3.\nExponent 2: 3 * 3 = 9, remainder 2.\nExponent 3: 2 * 3 = 6, remainder 6.\nExponent 4: 6 * 3 = 18, remainder 4.\nExponent 5: 4 * 3 = 12, remainder 5.\nExponent 6: 5 * 3 = 15, remainder 1.\nExponent 7: 1 * 3 = 3, remainder 3.\nExponent 8: 3 * 3 = 9, remainder 2.\nExponent 9: 2 * 3 = 6, remainder 6.\nExponent 10: 6 * 3 = 18, remainder 4.\nExponent 11: 4 * 3 = 12, remainder 5.\nExponent 12: 5 * 3 = 15, remainder 1.\nSo the requested remainder is 1.\n",
        "loopline":"Exponent 1: 1 * 3 = 3, remainder 3.\n",
        "repeats":12
    }
]
broad_texts=[]
broad_regions=[]
for c in broad_cases:
    prompt=prefix+c["problem"]+suffix
    for kind,body in (("productive",c["good"]),("loop",c["loopline"]*c["repeats"])):
        text=prompt+c["common"]+body
        broad_regions.append({"case":c["name"],"kind":kind,
                              "prompt_end":len(token_ids(prompt)),
                              "branch":len(token_ids(prompt+c["common"])),
                              "late":len(token_ids(prompt+c["common"]+c["loopline"]*2)) if kind=="loop" else len(token_ids(prompt+c["common"])),
                              "length":len(token_ids(text))})
        broad_texts.append(text)
broad_request={"type":"teacher_forced","model":"original","texts":broad_texts,
               "return":{"token_logprobs":True,"topk":20,"hidden_states":[16,24],"mlp_act":[23]}}
(W/"broad_probe_request.json").write_text(json.dumps(broad_request))
(W/"broad_probe_regions.json").write_text(json.dumps(broad_regions))
(W/"broad_probe_cases.json").write_text(json.dumps(broad_cases))
print("Broad probe regions:",broad_regions)
print("Matched negative-control head statistics:",
      [r for r in head_results if r["layer"]==15 and r["head"] in (1,7)])
with (W/"notes.md").open("a") as f:
    f.write("Attempted pass18 H2A generation failed engine startup, not counted; no H2 outcomes observed. "
            "Reorder remaining passes, not candidate: 18 original broad probes;19 edited broad;20/21 H2A/H2B; "
            "22 matched-head negative control;23 H7 amplification control;24/25 H3/H4 generation.\n")
    f.write("NEXT PASS18/25: teacher-forced ORIGINAL weights on six raw texts /work/broad_probe_request.json, "
            "regions /work/broad_probe_regions.json. Three productive/loop pairs: shopping arithmetic, linear equation, "
            "modular exponentiation with useful repeated procedure versus repeated same step. No closing-thinking tokens. "
            "Saved active candidate still ONLY L15 o_proj columns1792:2048 zero, but requested model=original ignores edits.\n")
```
