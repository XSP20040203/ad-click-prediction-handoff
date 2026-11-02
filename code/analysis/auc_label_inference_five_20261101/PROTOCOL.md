# Five-score conditional label analysis

The user supplied submission_best.csv with reported full-final AUC .74404.
Bind it to SHA256 45d0571bd12d3176c56bbeb16704a938ed8683909169daffcd860206b057f828.
Preserve all original inputs, trained code and the four-score inference output.
The unbounded AUC goal remains paused. This is a bounded conditional analysis.

Reuse the hash-locked generic four-score numerical functions with five rank
columns. Use the exact scored CSVs and average ranks for ties. All five metrics
are assumed to be ordinary ROC AUC on all 404182 rows, rounded to five decimals.
The reported fifth score is human feedback, not independently observed labels.

First test the old A/B/C hypotheses against the fifth score. Then fit new
minimum-KL Bernoulli rank-moment estimates around the same v9 model prior, using
the same assumed positive counts 2940/3459/3978. These are model-dependent soft
estimates. They are not the exact conditional posterior or recovered labels.
The unknown true number of positives remains unknown.

Construct two separately seeded central-count binary witnesses, plus an exact
equal-five-score permutation of the first. Each witness must match all five
reported metrics under independent sklearn ROC AUC to five decimals. Agreement
between sampled witnesses is not confidence or proof of a fixed label.

For the 20 highest central estimated probabilities, explicitly seek a complete
compatible assignment with that row's label opposite to the first witness.
First try an exact score-vector swap; otherwise use fixed-count rank-moment
swaps and optional small MILP repair excluding the challenged row. Successful
opposite witnesses prove ambiguity for those rows under the stated count.
Search failure would not prove a forced label. Store full witness assignments
and verify every one independently. Do not use inferred labels for training.

Compare exact score groups and probability estimates before/after adding the
constraint. Combinatorial symmetry counts apply only to the constructed
witness's within-group positive counts, not the true-label entropy. No new
submission improvement or final AUC is claimed. No true final labels are read.
