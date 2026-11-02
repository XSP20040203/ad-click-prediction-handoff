# One label assignment fitting six reported AUCs

The user requests one set of 17th-day labels best fitting all reported scores,
adding submission (3).csv with AUC .70590. The attachment SHA256 is
1ec5b7e1e47bdb108202bc12e2ce976625f4f06d7b20a45337f3797f670b2978.
Use the five exact, hash-bound earlier submissions and this sixth attachment.
The human already confirmed that all final rows are scored. Ordinary global
ROC AUC and rounding to nearest five decimals remain explicit assumptions.
Do not overwrite any dataset, submission, frozen source, or previous analysis.
The earlier unbounded AUC goal stays paused.

"Fit" first means that all six binary-label ROC AUCs round to the six reported
values. Use the original submission decimals and average ranks for ties.
Fix 3459 positives using the existing recent-two-complete-days central estimate;
this count is an assumption, not something identified by the AUC feedback.
The fitting tolerance is 4e-6 around each reported AUC, inside its rounding bin.
No true final labels are available or read.

Resolve nonuniqueness by preferring labels with larger independent Bernoulli
likelihood under the actual v9 probabilities. This is an additional model
preference, not evidence that such labels actually occurred. Do not describe a
time-limited solution as a proven maximum-likelihood solution.

Build one feasible six-score assignment using the frozen generic rank-moment
code. Fit a six-score minimum-KL soft probability estimate around v9, and use
the original v9 probabilities for the binary preference objective. Soft
estimates are not the exact conditional posterior or certified confidence.

Compress exactly equal six-prediction vectors. For group g, x_g is the integer
number of positives in [0,n_g]. The objective is sum(-logit(v9_g)*x_g).
Enforce sum(x)=3459 and six rank-sum intervals. Because v9 is one of the six
vectors, this compression preserves every ranking constraint and objective.

Solve the continuous group-count LP with HiGHS, at most 90 seconds. Preserve
its primal solution and inequality/equality dual multipliers. Independently
compute a valid objective lower bound using the dual multipliers and variable
bounds. Then perform at most three 20-second mixed-integer repairs on nested
pools of fractional and small-reduced-cost groups. Keep all valid incumbents
and select the one with lowest v9 negative log likelihood. If no better
integer incumbent is found, retain the initially feasible candidate and report
that limitation. Global optimality is claimed only if independently certified.

Choose rows inside each score-identical group by ascending original id. This
arbitrary deterministic choice contains no evidence about individual labels.
Deliver one CSV with id/log_id/label/estimated_click_probability. It is a fitted
label hypothesis file, not a competition probability submission.

Independent verification reloads all outputs, checks exact input hashes and
row/log_id order, recomputes actual sklearn ROC AUC for every retained integer
candidate, replays the soft probability projection, reconstructs group counts,
recomputes the objective and LP dual lower bound, and verifies incumbent
selection. Publish all six actual AUCs, assumed count and optimization gap.
Never train a supervised model on these fitted labels automatically.
