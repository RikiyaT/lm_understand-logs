This batch gained 54.7 percentage points, with 35 rescues and no regressions. Its stronger grade-school result also shows why individual batches should not be treated as precise performance estimates. I will update the pooled evidence and compress the notebook into a reproducible summary, preserving the diagnosis, controls, exact edit formula, and remaining evaluation schedule.

```python
# Hypothesis: The accumulated held-out gain is robust, and the frozen intervention can be audited independently.
register_frozen_run(59)

def refresh_frozen_notes(pass_number):
    current = Path("/work/notes.md").read_text()
    with open("/work/experiment_history.md", "a") as f:
        f.write(f"\n\n--- Notes snapshot through forward pass {pass_number} ---\n" + current)
    pooled = paired_run_summary(confirmed_runs)
    core = """# Frozen repetition repair: current submission
CURRENT/BEST: top64 alpha6 in layers15 and19 MLP down_proj only.
Backup: /work/best_confirmed_top64_a6. No weight changes since before pass33.
Formula at each layer: W = W0 - 6 * outer(u, mask64 * (u @ W0)).
u is a unit repetition direction from FOUR matched productive/repetitive original contexts
in fwd4, residual outputs16/20. Selected64 writers per layer chosen on original fwd4,
not held-out correctness. Files repetition_directions.pt/repetition_writer_top64.pt;
kernel variables retain directions/subsets and base_mlp_weights.
Only64 columns in each [1024,3584] matrix change;131072 values total.
Relative Frobenius changes10.6357%/8.1058%; formula errors<1e-7.
Frozen hashes in frozen_edit_audit.json. No norm/embedding/head changes, no training.

DIAGNOSIS AND CAUSAL EVIDENCE
Initial original gen1:11/96 correct,85 cutoff; repetitive cutoff tails ~95% duplicate8grams.
Matched-branch teacher forcing: full alpha1/2/4 repetitive logp delta
-.02410/-.04870/-.10372; productive +.00232/+.00343/+.00453.
L15-only4 repetitive-.08130; L19-only4-.03446.
Norm-matched orthogonal RANDOM control: gen35 14/128->16/128,+1.56pp SE3.52;
cut89.06->91.41%. No selective repetition suppression. REMOVED.
Head-ablation alternatives less selective and REMOVED.
Generation dose tests fullalpha1 gen14:27/256->59/256;
fullalpha2 gen16:29/256->92/256. Same problem set, but only pair within run.
Independent fullalpha2 gen27:21/256->110/256,+34.77pp SE3.53.
Fullalpha3 exploratory gen29:21/256->123/256,+39.84pp SE3.82.
Top64alpha4 gen41 at16k:13/128->47/128,+26.56pp SE4.45;
fullalpha4 gen42 at16k:14/128->48/128,+26.56pp SE4.59.
Top64alpha6 gen45 exploratory FULL32k:7/64->35/64,+43.75pp SE6.63.
Top64alpha6 transfer fwd44 (verified shared prefixes): productive-.00903,
repetitive-.10311 nats/token. Broader productive fwd48/49:-.00893;
numeric tokens-.00704 (429 tokens). No consistent forced early-closure effect.
Correct edited outputs still thousands of tokens of reasoning.

FROZEN CONFIRMATION DESIGN
Model both, n1, max_tokens32768, fixed evaluation sampling; all items distinct.
First validation_3: gen50+52,9/128->74/128,+50.78pp SE4.44,65rescues0losses.
validation_4:gen53,54,55,56,19/256->142/256,+48.05pp SE3.18,124rescues1loss.
Reserved final_1/2/3 each256, balanced GSM/MATH. No tuning on any confirmation.
frozen_eval_plan.json locks16 batches64 each after validation_3.
frozen_eval_progress.json stores completed labels, run IDs, summaries.
register_frozen_run(run_id) checks next expected IDs, appends pooled results, prints next request.
paired_run_summary(run_ids) asserts disjoint problems; pair only within each run.
paired_table, grouped_paired, behavior_summary are available analysis helpers.
Initialization failures produce no counted pass/results; identical retry usually works.

BEHAVIOR AUDIT THROUGH256 CONFIRMATION ITEMS
Original17 correct,241cut;edited144correct,60cut.
Cut-tail duplicate8gram mean .96449 original vs .28755 edited.
All144 edited correct close thinking after>1000chars;median9836tokens.
Edited15 whitespace-cut tails vs0 original: unresolved alternative degeneration.
First regression gsm8k_train_3844:original correct boxed20 then cut;
edited finishes boxed10 (wrong). Do not ignore regressions.
At384 problems positive descriptive MATH gains in all subjects/levels;
hardest Level5:0/59->9/59. Subgroup estimates noisy.

"""
    progress = {
        "forward_passes_used": pass_number,
        "forward_passes_left": 50-pass_number,
        "confirmed_runs": confirmed_runs,
        "completed_locked_batches": len(frozen_eval_completed),
        "pooled": pooled,
        "next_batch": frozen_eval_plan[len(frozen_eval_completed)]["label"]
            if len(frozen_eval_completed) < len(frozen_eval_plan) else None
    }
    Path("/work/notes.md").write_text(core + "CURRENT PROGRESS\n" + json.dumps(progress, indent=2) + "\n")
    print("Notebook refreshed; passes left:", 50-pass_number)

refresh_frozen_notes(40)
```
