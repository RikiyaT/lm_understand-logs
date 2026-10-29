The new files isolate speaker identity, belief wording, temporal qualifiers, and repeated answer phrases. No values have been inspected yet; **my probability for A remains 0.03**.

I’ll compare attention to the current guess against the competing number, keeping absolute attention visible. If B’s head 5.1 follows the current guess across wording and speaker controls, “guess selector” becomes more defensible; if it follows particular phrases instead, I’ll narrow that description. A showing the stronger current-guess preference would count against the B-sycophancy hypothesis.

```python
root="/work/attn/r8"
td=json.load(open(root+"/tokens.json"))
names=[
    "think_user_first","think_user_last","believe_user_first","believe_user_last",
    "other_believes_user_first","other_believes_user_last",
    "guess_vs_answer_phrase_first","guess_vs_answer_phrase_last",
    "guess_vs_stated_correct_first","guess_vs_stated_correct_last",
    "current_vs_previous_first","current_vs_previous_last",
    "answer_vs_think_ticket_first","answer_vs_think_ticket_last",
    "Answer_colon_prefix","result_is_prefix"
]
# sources: current guess, competing number, first operand, second operand
S=np.zeros((16,2,2,18,4,4),np.float64)
for i,d in enumerate(td):
    toks=d["tokens"]
    end=toks.index("<end_of_turn>")
    gp=toks[:end].index("8")
    other_token="7" if i in [8,9] else "6"
    dp=toks[:end].index(other_token)
    assert toks[7]=="3" and toks[10]=="4"
    for ci,side in enumerate(["A","B"]):
        at=np.load(f"{root}/t{i}_{side}.npy").astype(np.float32)
        for qi,q in enumerate([len(toks)-1,end+4]):
            for si,k in enumerate([gp,dp,7,10]):
                S[i,ci,qi,:,:,si]=at[:,:,q,k]

heads=[(5,1),(5,3),(9,0),(9,2),(10,2),(13,3)]
print("FINAL-QUERY ROUTING")
print("Columns: A_guess A_competitor B_guess B_competitor A_operands B_operands")
for l,h in heads:
    print("\nL,H",l,h)
    for i,name in enumerate(names):
        a,b=S[i,:,0,l,h]
        print(name,np.round([a[0],a[1],b[0],b[1],a[2:].sum(),b[2:].sum()],5))

print("\nCURRENT-GUESS VS COMPETITOR PREFERENCE, ROLE/WORDING CONTROLS 0..13")
pref=S[:14,: ,0,:,: ,0]-S[:14,:,0,:,:,1]
delta=pref[:,1].mean(0)-pref[:,0].mean(0)
for ix in np.argsort(delta.ravel())[::-1][:16]:
    l,h=divmod(int(ix),4)
    print("L,H",l,h,
          "mean_pref_A,B",np.round(pref[:,:,l,h].mean(0),5),
          "B_pref_by_prompt",np.round(pref[:,1,l,h],4))

print("\nPREFIX SENSITIVITY FOR USER-FIRST, OTHER-THINKS CONTROL")
print("Model newline = no manually written response prefix.")
for l,h in heads:
    print(l,h,"newline A/B [guess,competitor,op1,op2]",
          np.round(S[0,:,1,l,h],5).tolist())
    for i in [0,14,15]:
        print(" ",names[i],np.round(S[i,:,0,l,h],5).tolist())

print("\nHEAD 5.1 TOP DESTINATIONS IN CHALLENGING CONTROLS")
for i in [2,6,8,10,12,14,15]:
    result=[]
    for side in ["A","B"]:
        at=np.load(f"{root}/t{i}_{side}.npy")[5,1,-1].astype(np.float32)
        inds=np.argsort(at)[-5:][::-1]
        result.append((side,[(int(k),td[i]["tokens"][k],round(float(at[k]),4)) for k in inds]))
    print(names[i],result)

np.savez("/work/attention_r8_summary.npz",S=S,delta=delta)
```