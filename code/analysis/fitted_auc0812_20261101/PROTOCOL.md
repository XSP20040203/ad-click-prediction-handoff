# .812 submission against the original fitted-label hypothesis

The user reports .80855 for the immediately preceding .810 fitted submission,
whose SHA256 is 7e239cb191876455e9038ad8180a5aec9db81e07e8d17862bfef8b1180d811fe.
Record this as user feedback, not independently recomputed platform AUC.
The prior hypothesis produced .81000 for that file, so it does not reproduce
the new feedback. No claim is made that it is the true label set.

The user requests a .812 fitted submission. Continue using the SAME original
six-score optimized fitted-label file and exact scored v9 base; this task is
score construction, not a new seven-score label inference. Explicitly disclose
that distinction and do not predict a real platform .812 result.

Reuse the immutable .815 exporter with target=.812, retaining its exact search,
probability adjustment and 10-decimal export rules. Only fitted-positive log
odds are increased. Store the new CSV in a separate directory and preserve
both .810/.815 submissions. Rename the engine's temporary legacy filename
inside the newly created directory, bind all source and output hashes, and
independently verify actual rounded-CSV AUC and original row order.

Keep platform-best v9=.80856 unchanged. Leave the new .812 submission's actual
platform score unknown. No models are trained, no original labels are changed,
and the earlier unbounded training objective is not resumed.
