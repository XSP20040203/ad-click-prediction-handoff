# Explicitly authorized fitted-label training toward platform AUC .83

The user's latest instruction explicitly authorizes training on the chosen
17th fitted labels. It supersedes the earlier inference protocols' instruction
not to train automatically. Preserve all datasets, fitted labels, submissions,
frozen base models and source files. The earlier unbounded .85 run stays paused.
Interpret .83 as the desired real platform AUC. No real final labels exist here;
neither a fitted-label training AUC nor old aggregate feedback verifies .83.

Freeze the actual platform-best v9 ensemble (.75*v6+.25*v8). Train a regularized
logistic residual on features of the three retained base model predictions:
logits, within-day ranks, disagreements and interactions. Numeric correction
is capped at .5 log odds. No id/log_id features or per-row memorization.

Use genuine day15 out-of-sample labels to train all development residuals.
Generate day16 fitted labels without providing any day16 real outcomes: use
only day15 base-model AUCs as assumptions for the new day, and estimate the
day16 click count from days14/15 CTR. Fit four rank moments (v6/v7/v8/v9),
anneal a minimum-KL dual to T=.03125 and repair two bounded fixed-count
assignments. Choose the higher-v9-likelihood feasible assignment. This is a
strict-past-feedback simulation, not the same six-feedback oracle as day17.

Fit pseudo weights 0/.01/.03/.1/.3/1 plus a matched weight-.1 shuffled-label
negative control. Objective is past mean BCE + weight*target pseudo mean BCE
+ .0003/2*||beta||^2. All development weights must be saved before loading real
day16 labels for evaluation. The positive weights are selected by actual day16
AUC, with smaller weight breaking ties. Report the true-only control and v9.
The dates/base checkpoints have been reused historically, so this is reused
development evidence, not a fresh blind benchmark. The four-rank simulation
cannot certify the six-feedback fitted labels on day17.

Refit the selected positive-weight correction using real day16 labels plus the
exact user-authorized fitted day17 labels. Also refit the true-only control.
Fit normalization on true-source rows only in every stage. Export the selected
positive-weight model as an experimental submission even if development
does not support promotion, explicitly reporting that limitation. Never
overwrite or replace the actual best platform submission on local metrics.

Do not choose a model or alter prediction ranks to force a .83 score on fitted
labels. Record that score only as an in-sample synthetic-label diagnostic.
Only subsequent platform feedback can confirm or refute the real .83 target.

Independent verification reconstructs features, normalizers, all models and
saved predictions; checks row order, old hashes, optimizer stationarity and
model/loss replay; scores real day16 labels and synthetic day17 labels under
separate names; verifies the selected weight and CSV rounding.
