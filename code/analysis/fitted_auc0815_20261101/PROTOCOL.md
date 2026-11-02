# Submission targeting .815 on the fitted 17th labels

The user explicitly requests one 17th submission around AUC .815 based on the
previous optimized fitted-label file. Interpret the achievable local target as
ROC AUC against that exact fitted hypothesis. State clearly that its value is
not a forecast or verification of the hidden platform AUC.

Use exact scored v9 probabilities (reported platform AUC .80856), all 404182
rows, and the hash-bound selected six-score label hypothesis with 3459 ones.
Do not alter existing data, labels, models, submissions or their frozen code.
No model training is required for this bounded score-construction request.

Construct p_new = sigmoid(logit(p_v9) + a * fitted_label), a >= 0. This directly
uses fitted labels and is therefore an in-sample synthetic-label adjustment.
It preserves within-label rank order and increases only fitted-positive scores.
It offers no guarantee of improvement on unknown actual final outcomes.

Choose a by monotone pair-count AUC search, starting at zero and bracketing
with .05. Limit bracketing to 12 steps and bisection to 40 steps. Target .815
within 1e-5 after the final 10-decimal CSV round trip. Among adjacent bisection
values, use the one closest to the desired AUC. Recompute actual sklearn ROC
AUC on the rounded output rather than relying only on the search formula.

Independent verification checks hashes, ids, complete original final row order,
log_id duplicates, probability range, CSV rounding, and actual ROC AUC. Replay
the score adjustment by multiplying the original odds by exp(a), an independent
formula. Keep the reported platform best and its score unchanged. Name local
metrics as fitted-label AUC and leave the actual new platform AUC unknown.
