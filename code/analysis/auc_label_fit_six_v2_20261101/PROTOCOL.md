# Six-score fit with bounded column generation

The requested output is one binary assignment matching all six reported AUCs
to five decimals. All assumptions and the likelihood preference from
analysis/auc_label_fit_six_20261101/PROTOCOL.md remain applicable, including the
assumed 3459 positives, exact scored CSVs, arbitrary deterministic choices
inside equal-score groups, and the fact that fitted labels are not truth.

The first run's full 261474-variable LP call exceeded its intended budget
and was terminated, preserving its feasible checkpoint and all frozen sources.
This successor replaces only that LP call with bounded column generation.
Keep the first source, first checkpoint and every earlier artifact intact.

Start with all groups containing an initial positive, 2048 lowest-cost groups,
and 512 evenly spaced groups. Solve restricted LPs with presolve disabled,
an 8-second time limit per call and at most 10 calls. Add up to 4096 excluded
groups with the most negative reduced costs each time. Preserve the restricted
primal solution and its duals; evaluate the dual lower bound against ALL groups.
Even if column generation stops early, that lower bound remains valid, while
global LP optimality is not asserted. If no restricted solve succeeds, return
the feasible checkpoint with zero duals as a documented fallback.

Then use the frozen first implementation's bounded integer repairs and select
the lowest-prior-NLL valid incumbent. The wrapper and reused implementation
are both hash-bound. The independent verifier and its reused implementation
are also hash-bound. Recompute all six actual AUCs, the objective and the dual
bound before delivery. A solver time limit or nonzero optimality gap is a
limitation, never evidence that the fitted labels are true.
