# Bounded submission task from three actual scored files

The user requests one improved submission candidate, based on platform feedback.
This is a concrete bounded task; the earlier unbounded0.85 goal was paused and
is not silently restarted. Unfinished frozen-history transfer files are left
untouched. No new models or target labels are fitted in this task.

Actual attachments bind by SHA256 to v6=.80817, v7=.80581, v8=.80745. These file
bindings supersede the earlier ambiguous verbal version answer. Platform scores
are human feedback, not independently rescored here. Source arrays must match
the actually scored CSVs within their original decimal rounding tolerance.

v8=original_v7+.30*(protected_day_full-original_gated166_full). Its observed
platform improvement over v7 motivates isolating this correction on the stronger
v6 base. AUC differences are not additive and transfer is not guaranteed.

Compare only three fixed export candidates: v6 logit plus the v8-v7 logit change
clipped to[-.75,.75], the same with coefficient.5, and .75*v6+.25*v8 probability
blend. The full additive probability correction is an evaluation-only diagnostic
because it can leave[0,1]. No clipping invalid probabilities into ties for export.

Select the largest complete15-day AUC among the fixed candidates improving v6
and worsening Logloss no more than1e-4. Save that selection before computing16
candidate scores. The selected rule must also improve v6 on the full16-day set.
No coefficient search, final-label fit, row filtering or manual score adjustment.
The preexisting days were repeatedly used; this is not a new blind benchmark.

Export one named submission_v9.csv, using verified original unrounded arrays
and10 decimals to avoid unnecessary new ties. Preserve all404182 rows, original
zero-based ids and858 duplicate underlying log_id rows. Separate-process
verification computes the formula using odds multiplication, checks all old
file hashes, source rounding consistency, raw validation labels, full-day AUC,
user-cluster uncertainty and final row/schema/range/rounding contracts.

The new platform AUC stays unknown until the user submits it. Do not infer it
from the two/three aggregate scores, development lift or the formula's rationale.
