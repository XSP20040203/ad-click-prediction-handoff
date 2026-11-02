# Submission targeting .810 on the same fitted labels

The user requests a .81 version of the previously generated .815 fitted-label
submission. Reuse the frozen .815 score-construction engine with target=.810,
the same exact v9 probabilities and the same optimized six-score fitted labels.
Save a separate submission_fitted_auc0810.csv. Preserve the prior .815 export.

All adjustment, search, row-order and rounding rules of the frozen engine
remain unchanged. The engine briefly uses its legacy .815 filename inside
the NEW directory; rename only that new file and bind the final name and hash.
Bind both the engine and wrapper sources. Independently verify actual AUC on
the exported CSV by sklearn and replay probabilities using odds multiplication.

The desired .810 metric is against the same synthetic fitted labels used to
choose the adjustment. It is not independent validation or a prediction of the
actual hidden platform AUC. Keep that real score unknown and preserve the
reported platform best. No training or original dataset changes are requested.
