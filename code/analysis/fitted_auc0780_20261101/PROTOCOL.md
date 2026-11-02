# Eight-feedback diagnostic and a requested .78 experimental submission

The user asks to continue optimization and produce a .78 submission after two
fitted-score exports both received user-reported platform AUC .80855. The
achievable local .78 target is explicitly interpreted against the original
six-score fitted labels. This is a diagnostic artifact, not an improved or
verified .78 platform model. The v9 platform best .80856 stays unchanged.

Improve the feedback analysis by including all EIGHT actual scored CSVs:
v6/v7/v8/v9/old_best/submission_3/fitted0810/fitted0812. Preserve exact decimal
ties. Assume all rows are scored by ordinary global ROC AUC rounded to five
decimals, as previously confirmed by the user. Fix the same assumed P=3459;
the true number of clicks is still unknown.

Use SVD rather than covariance eigenvalue whitening for the eight rank columns:
the observed covariance condition number is about1.24e10. Fit a minimum-KL
Bernoulli rank-moment projection around v9 with an intercept, at most120
trust-exact steps. Record any failure rather than claiming feedback is met.
Soft expected rank moments are not binary ROC AUC, an exact conditional
posterior, calibrated confidence or independently measured platform results.

Do not shift rows according to guessed positive labels again. Instead blend
two actual-scored model outputs with a single global weight:
p = (1-w)*old_best + w*v9, w in[0,1]. The two reported platform endpoint AUCs
are .74404 and .80856. Their mixture AUC is not generally a linear combination
of those endpoint AUCs.

Select a blend with original-fitted-label AUC .78 within1e-5. Scan21 weights,
bracket a sign crossing, then bisect at most40 times. If multiple local
solutions exist, prefer the one whose eight-feedback soft rank-moment proxy
is closest to .78. This choice still uses a hypothetical label model and is
not independent evidence of real performance. Report both local quantities
under separate explicit names and keep new real platform AUC unknown.

Export all404182 rows as id,pctr with10 decimals. Independent verification
checks all source hashes and past feedback bindings, final row/log_id order,
replays the SVD projection using its saved basis, checks expected moments,
recomputes actual ROC AUC on the rounded CSV against the original hypothesis,
and independently reconstructs the global blend. Keep every old artifact.
No deep model or residual model is newly trained in this bounded task.
