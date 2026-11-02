# Four-AUC conditional label inference

The user explicitly requests reading the dataset and inferring17th labels from
the four scored files. They confirm scores use all17th rows. Earlier unbounded
training remains paused; this is a separately authorized inference/identifiability
analysis. Preserve every original dataset, submission and trained source file.

Use the exact scored CSV predictions with average ranks for ties, not their
unrounded underlying model arrays. AUC feedback: v6=.80817,v7=.80581,v8=.80745,
v9=.80856. Ordinary global ROC AUC and nearest-five-decimal rounding are explicit
assumptions. The true final positive count is not given by the feedback.

With N rows and assumed P clicks, centered ranks U=(rank-(N+1)/2)/N satisfy
sum(y*U)=P*(1-P/N)*(AUC-.5). Thus four feedback values give four moment
constraints. Fit a minimum-KL/maximum-entropy tilt around the actual v9 model
probabilities, with an intercept for sum(q)=P. Whiten rank features for numeric
conditioning. This is model-dependent conditional estimation, not recovery of
observed events. Expected rank moments under soft q are not binary ROC AUC.

Read original training dates/labels and all17th input rows. Estimate P from the
last two complete training-day CTRs, with.85/1.00/1.15 sensitivity factors. No
assumed count is described as the actual17th count. Return all three probability
vectors to show this unresolved degree of freedom.

Construct two different probability-weighted binary hypotheses with central P,
then improve four rank sums using fixed-cardinality swaps. Optional small MILP
repairs are feasibility searches, not claims of maximum-posterior optimization.
Every delivered binary hypothesis must reproduce all four reported AUC values
to their five-decimal precision, checked by actual ROC AUC. A third hypothesis
maximizes label exchange within exactly identical four-score groups, preserving
their positive counts, every AUC and the chosen model prior likelihood.

Report differences between compatible solutions and log2 of exact score-tie
permutations. Construction overlap is not calibrated confidence. No inferred
hard label is called actual ground truth or automatically used for training.

Backtest the probability method on15/16 with their actual P and rounded oracle
aggregate scores supplied, an optimistic diagnostic. Report top-P positive
precision/recall and proper losses, not only mostly-negative accuracy. Such
oracle-conditioned backtests are not fresh independent validation results.

Four tests cover the rank/AUC identity with ties, a known interior moment
projection, exactly invariant label swaps and safe proposal sampling.
Independent verification recomputes ROC AUC on every hypothesis, probability
reload, raw row order, source file hashes and exact tie-group ambiguity.

Output filenames and columns explicitly say probability estimate and label
hypothesis. There is no source of actual17th ad outcomes in this analysis. A
label set fitting leaderboard totals is not proof it is the true label set.
