# Bounded six-score label fit

The user requests one fitted 17th-day label assignment using all six scores:
.80817/.80581/.80745/.80856/.74404/.70590. Use the hash-bound input files and
assume ordinary global ROC AUC over all final rows, rounded to five decimals.
Fix 3459 positives using the earlier recent-training-day central estimate.
The actual positive count and individual final labels remain unknown.

Two earlier full/restricted HiGHS searches were terminated after exceeding
their intended elapsed-time budgets. Preserve their source and checkpoints.
This bounded successor starts with the already feasible six-score checkpoint.
Among assignments that match all six displayed scores, prefer smaller v9
independent-Bernoulli negative log likelihood, without claiming true labels.

Fit entropy-regularized rank moments at temperatures .5/.25/.125/.0625/.03125.
Use a seven-variable convex dual and warm starts, at most 80 trust-exact steps
per temperature. Obtain a valid lower bound for the unregularized integer
objective from each dual and all group bounds. Bounds do not require optimizer
success. Do not claim global integer optimality unless the bound closes.

At each temperature, construct two fixed-count assignments by perturbed
thresholding of the tempered logits. Repair rank sums using at most 180 greedy
swaps per assignment, with positive and negative proposals concentrated near
the temperature-dependent decision boundary. No MILP or unbounded search.
Retain only candidates satisfying every five-decimal AUC and select the one
with smallest v9 prior NLL, including the initial feasible checkpoint.
This is best among the searched candidates, not a proven global optimum.

The CSV contains id/log_id/label/estimated_click_probability. Its probability
column is the original temperature-one six-score minimum-KL soft estimate,
not the hardened assignment and not an exact conditional posterior.
Independent verification reloads inputs, recomputes actual sklearn AUC for
every retained candidate, checks row order/counts, objective selection and dual
lower bounds. Preserve all data and submissions; do not train on fitted labels.
