The 10% attention combination gained 4.30 points with a 2.19-point standard error, while length and cutoff rate stayed near the original model. This is the strongest result so far, but it was selected after several trials and needs independent DEV replication. I will save it and rerun the exact weights unchanged. I will also use the remaining DEV passes to check whether the selected MLP units add value beyond the head adjustment alone.

```python
archive_current("candidate_head110_matched11half",manifest29)
log_pass(30,"Unchanged replication of head2 1.1x plus matched layer11 half-suppression, reversed DEV order",manifest29)
with (W/"notes.md").open("a") as f:
    f.write("\nPass29: +4.30 pp, SE2.19 pp, original494/1024 vs edited538/1024; "
            "mean tokens866->832, cut share .0439->.0381. "
            "Candidate archive candidate_head110_matched11half. "
            "Remaining DEV plan: replicate this candidate; head1.1-only ablation; "
            "head1.1 plus norm/activation-matched low-association MLP control; "
            "final replicated selection by pass34. HOLDOUT not yet accessed.\n")
print("PASS29",summarize_pair(29))
print("Archived and logged unchanged pass30.")
```