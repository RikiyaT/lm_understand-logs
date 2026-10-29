The restricted-neuron probe completed successfully. I’ll quantify how much of the full edit’s effect it retains, save this candidate separately, and organize fresh validation splits before further generation tests. This prevents later exploratory choices from leaking into the final confirmation set. A narrower edit is worth pursuing only if it preserves both the mechanistic selectivity and the accuracy gain.

```python
# Hypothesis: restricting the edit to 64 repetition-writing neurons per layer preserves benefit with less collateral change.
import shutil
subset_metrics = transfer_summary("/work/fwd_39")
print("TOP64 alpha4", json.dumps(subset_metrics))
shutil.copytree("/work/edited", "/work/candidate_top64_a4", dirs_exist_ok=True)
if "fresh_splits" not in globals():
    practice_by_source = {}
    for source, filename in [("gsm8k", "gsm8k_train.jsonl"), ("math", "math_train.jsonl")]:
        with open("/practice/" + filename) as f:
            practice_by_source[source] = [json.loads(line) for line in f if line.strip()]
    used_ids = set()
    for value in spl.values():
        if isinstance(value, list):
            used_ids.update(x for x in value if isinstance(x, str))
    for path in Path("/work").glob("gen_*.jsonl"):
        with open(path) as f:
            used_ids.update(json.loads(line)["id"] for line in f if line.strip())
    rng_fresh = np.random.default_rng(2026102407)
    pools = {src: rng_fresh.permutation([r["id"] for r in rows if r["id"] not in used_ids]).tolist()
             for src, rows in practice_by_source.items()}
    fresh_splits = {}
    offset = 0
    for label, per_source in [("validation_3", 64), ("validation_4", 128),
                              ("final_1", 128), ("final_2", 128), ("final_3", 128)]:
        fresh_splits[label] = pools["gsm8k"][offset:offset+per_source] + pools["math"][offset:offset+per_source]
        offset += per_source
    Path("/work/fresh_splits.json").write_text(json.dumps(fresh_splits, indent=2))
print("Reserved unseen sets", {k: len(v) for k, v in fresh_splits.items()})
summary = """# Math weight editing — through pass26/50, Python54
BEST INDEPENDENTLY CONFIRMED: full-direction alpha2, L15/L19 MLP down_proj; backup best_confirmed_a2.
CURRENT: top64-neuron-only alpha4 in same two tensors, candidate_top64_a4. NOT accuracy-tested.
Allowed edit formula full: W0-alpha*outer(u,u@W0). Unit directions in repetition_directions.pt keys15,19.
Subset: zero entries outside writer_subsets[l] in u@W0 before outer product; subsets in repetition_writer_top64.pt.
No final norm/embedding/output/head edits. No training. Original tensors cached base_mlp_weights; helper set_repetition_edit({15:a,19:b}).

DIAGNOSIS/DIRECTION:
Initial gen1 original11/96 correct;85 cutoff; cut tails ~95% duplicate8-word sequences.
Directions from FOUR matched productive/repetitive contexts in original fwd4 at residual outputs16,20.
Early L19 neuron11 ablation and L19-only rank1 each +6.25pp SE4.35 on32, weak exploratory.
Both sites alpha1 subsequently much stronger. Correct alpha3 answers median8071tokens; not skipping thinking.
Repetition attention heads L15h7,L19h5 identified in fwd19 but head ablation less selective; REMOVED.
Alpha3 gen29 cut91/256;42 wrong/noanswer finished. Whitespace tails12/256 versus1 original.
Alpha2 independent gen27 whitespace3/256 versus0 original. These sets differ, not a clean dose comparison.

GENERATION EVIDENCE (paired within run only; batching-dependent RNG means DON'T pair across runs):
pass10/gen13 alpha1 independent128:11->42 correct,+24.22pp SE3.96,32rescues1loss.
pass11/gen14 alpha1 dose64x4:27/256->59/256,+12.50pp SE2.61.
pass12/gen16 alpha2 dose64x4:29/256->92/256,+24.61pp SE3.96,64rescues1loss.
pass19/gen27 alpha2 independent128x2:21/256->110/256,+34.77pp SE3.53,91rescues2losses.
  GSM13.28->60.94%,MATH3.12->25%;cut92.58->51.17%;mean30644->21742. Backup best_confirmed_a2.
pass20/gen29 alpha3 exploratory128x2:21/256->123/256,+39.84pp SE3.82,102rescues0losses.
  GSM7.81->53.91%,MATH8.59->42.19%;cut92.97->35.55%. candidate_L15_L19_a3, no independent confirmation.
pass23/gen35 norm-matched RANDOM control64x2:14/128->16/128,+1.56pp SE3.52,8rescues6losses;
  cut89.06->91.41%,mean29646->30405. Random control REMOVED.
Full alpha4 generation repeatedly fails ENGINE INITIALIZATION (no counted passes), both128x2 and64x2.
Generation sometimes recovers; random-control64x2 worked. No established workaround. TF always works.

DENSE TRANSFER, six new productive/repetitive pairs plus two premature-close controls:
Exact raw texts in transfer_probe_request.json (14texts), boundaries in transfer_probe_bounds.json.
Use raw thinking template; chats previously inserted empty think block. eval_prompt_format.txt holds template.
Baseline fwd24; alpha1 fwd25; alpha2 fwd26; fullalpha4 fwd32; random control fwd34;
L15-onlyalpha4 fwd37;L19-onlyalpha4 fwd38;top64alpha4 fwd39.
All metrics compare logprobs on SAME branch tokens; array logprobs index token_position-1.
Mean branch deltas productive/repetitive:
 alpha1 +.00232/-.02410; alpha2 +.00343/-.04870; fullalpha4 +.00453/-.10372.
 random norm-matched +.00392/+.00955 (not selective; orthogonal directions).
 L15-only4 -.00058/-.08130; L19-only4 +.00346/-.03446.
Premature-close logp baseline -19.77/-19.69; fullalpha4 -18.81/-18.77 (still extremely unlikely).
Control directions orthogonal to repetition, per-layer Frobenius norm matched within.03%, saved random_control_match_a4.pt.
Top64 chosen ONLY on original fwd4: ~91%/98% of transfer repetitive-minus-productive writing contrast,
but50%/40% of full directional weight norm. Relative matrix changes top64a4:7.09%/5.40%; fulla4:14.11%/13.50%.
"""
summary += "\nPass26/fwd39 TOP64 alpha4 transfer: " + json.dumps(subset_metrics)
summary += "\nReserved fresh_splits.json: validation_3(128),validation_4(256),final_1/2/3(each256), disjoint all prior problems.\n"
summary += "Next: paired exploratory generation of top64alpha4, then fullalpha4 if service permits; independently confirm chosen candidate.\n"
Path("/work/notes_before_pass26.md").write_text(Path("/work/notes.md").read_text())
Path("/work/notes.md").write_text(summary)
```
