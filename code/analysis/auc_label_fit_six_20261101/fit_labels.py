"""Fit one binary hypothesis to six scores with a documented model preference."""
import os
for thread_variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[thread_variable] = "3"
from pathlib import Path
import json
import shutil
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import logit, gammaln
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/fitted_labels_six_scores_20191117"
OLD_OUT = ROOT / "output/label_hypotheses_five_scores_20191117"
GENERIC = ROOT / "analysis/auc_label_inference_20261101"
sys.path.insert(0, str(GENERIC))
import infer_labels as base

USER_FILE = Path("E:/浏览器下载/submission (3).csv")
SIXTH_HASH = "1ec5b7e1e47bdb108202bc12e2ce976625f4f06d7b20a45337f3797f670b2978"
SETTINGS = {"assumed_positive_count": 3459, "auc_tolerance": 4e-6,
    "lp_seconds": 90., "milp_seconds_per_pool": 20., "milp_positive_pool_sizes": [256, 768, 2048],
    "milp_relative_gap": 1e-5, "baseline_seeds": [91271, 81431, 701531],
    "prior": "Actual v9 scored probabilities", "clipping": 1e-9,
    "selection": "All six rounded AUCs match; lowest v9 prior NLL among valid searched integer candidates",
    "within_equal_score_group_order": "Ascending original id, arbitrary deterministic choice"}


def objective_bound(costs, group_u, sizes, target, width, inequality_dual, equality_dual):
    inequality_dual = np.minimum(np.asarray(inequality_dual), 0.)
    matrix = np.vstack([group_u.T, -group_u.T])
    bounds = np.r_[target + width, -target + width]
    residual = costs - matrix.T @ inequality_dual - float(equality_dual)
    bound = bounds @ inequality_dual + SETTINGS["assumed_positive_count"] * float(equality_dual)
    bound += sizes @ np.minimum(residual, 0.)
    return float(bound), residual


def expand_counts(counts, groups):
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_groups)) + 1]
    within_group = np.arange(len(groups)) - np.repeat(starts, np.diff(np.r_[starts, len(groups)]))
    labels = np.zeros(len(groups), dtype="int8")
    labels[order] = (within_group < counts[sorted_groups]).astype("int8")
    return labels


def run():
    started = time.perf_counter()
    old_protocol = json.loads((OLD_OUT / "protocol.json").read_text(encoding="utf-8"))
    old_report = json.loads((OLD_OUT / "report.json").read_text(encoding="utf-8"))
    assert base.digest(GENERIC / "infer_labels.py") == old_protocol["generic_source_sha256"]
    for filename, expected in old_protocol["original_datasets_sha256"].items():
        assert base.digest(ROOT / "2026ml" / filename) == expected, "Dataset changed"
    assert base.digest(USER_FILE) == SIXTH_HASH, "New scored attachment changed"
    assert base.digest(OLD_OUT / "inference_arrays.npz") == old_report["files_sha256"]["inference_arrays.npz"]
    final_path = ROOT / "2026ml/final_test.csv"
    assert "label" not in pd.read_csv(final_path, nrows=0).columns
    final = pd.read_csv(final_path, usecols=["log_id", "pt_d"])
    n, positives = len(final), SETTINGS["assumed_positive_count"]
    assert n == 404182 and final.pt_d.eq(20191117).all()
    versions = old_protocol["auc_versions"] + ["submission_3"]
    scores = np.r_[old_protocol["reported_auc"], .70590]
    files = dict(old_protocol["scored_files_sha256"])
    files["submission_3"] = (str(USER_FILE), SIXTH_HASH)
    predictions = []
    for name in versions:
        path, expected = files[name]
        path = Path(path) if Path(path).is_absolute() else ROOT / path
        assert base.digest(path) == expected, "Scored submission changed"
        frame = pd.read_csv(path, float_precision="round_trip")
        np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(n))
        p = frame.pctr.to_numpy(dtype="float64")
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        predictions.append(p)
    predictions = np.column_stack(predictions)
    u = base.rank_features(predictions)
    unique, representatives, groups, sizes = np.unique(predictions, axis=0, return_index=True, return_inverse=True, return_counts=True)
    group_u = u[representatives]
    prior = np.clip(predictions[:, 3], SETTINGS["clipping"], 1 - SETTINGS["clipping"])
    costs = -logit(np.clip(unique[:, 3], SETTINGS["clipping"], 1 - SETTINGS["clipping"]))
    constant_nll = float(-np.log1p(-prior).sum())
    target = base.targets(n, positives, scores)
    width = positives * (1 - positives / n) * SETTINGS["auc_tolerance"]
    with np.load(OLD_OUT / "inference_arrays.npz", allow_pickle=False) as archive:
        old_checks = {name: float(roc_auc_score(archive["hypotheses"][:, j], predictions[:, 5])) for j, name in enumerate(["A", "B", "C"])}
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(USER_FILE, OUT / "scored_submission_3.csv")
    protocol = {"settings": SETTINGS, "versions": versions, "reported_auc": scores.tolist(), "scored_files": files,
        "sixth_attachment_sha256": SIXTH_HASH,
        "source_sha256": base.digest(__file__), "verifier_sha256": base.digest(HERE / "verify_labels.py"),
        "specification_sha256": base.digest(HERE / "PROTOCOL.md"),
        "generic_source_sha256": base.digest(GENERIC / "infer_labels.py"),
        "generic_baseline_settings": base.SETTINGS,
        "original_datasets_sha256": old_protocol["original_datasets_sha256"],
        "previous_five_score_output_sha256": {name: base.digest(OLD_OUT / name) for name in ["protocol.json", "report.json", "inference_arrays.npz", "conditional_estimates_20191117.csv", "verification.json"]},
        "scores_cover_all_rows_confirmed_by_user": True,
        "ordinary_auc_and_nearest_five_decimal_rounding_assumed": True,
        "true_final_positive_count_unknown": True, "fitted_labels_are_not_observed_ground_truth": True,
        "unbounded_auc_goal_remains_paused": True}
    base.write(OUT / "protocol.json", protocol)
    base.emit("six_score_sources_verified", rows=n, scores=scores.tolist(), unique_vectors=len(sizes),
              previous_hypotheses_sixth_auc=old_checks, assumed_clicks=positives)
    probability, whitening, projection = base.conditioned_probabilities(prior, u, scores, positives)
    base.emit("six_score_soft_projection", **projection)
    initial = None
    initial_failures = []
    for seed in SETTINGS["baseline_seeds"]:
        try:
            initial, initial_info = base.construct_hypothesis(probability, u, whitening, scores, positives, seed)
            break
        except RuntimeError as error:
            initial_failures.append({"seed": seed, "error": str(error)})
    if initial is None:
        raise RuntimeError("No feasible initial six-score assignment found")
    candidates = [np.bincount(groups, weights=initial, minlength=len(sizes)).astype("int64")]
    records = [{"name": "initial_rank_moment_feasible", "objective": float(costs @ candidates[0]),
                "prior_nll": float(constant_nll + costs @ candidates[0]), "construction": initial_info}]
    base.emit("initial_six_score_labels", **records[0])
    np.savez_compressed(OUT / "initial_checkpoint.npz", probability=probability, initial_labels=initial)

    upper_matrix = np.vstack([group_u.T, -group_u.T])
    upper_bounds = np.r_[target + width, -target + width]
    base.emit("global_continuous_rank_lp_started", variables=len(sizes), maximum_seconds=SETTINGS["lp_seconds"])
    relaxation = linprog(costs, A_ub=upper_matrix, b_ub=upper_bounds,
        A_eq=np.ones((1, len(sizes))), b_eq=[positives], bounds=np.column_stack([np.zeros(len(sizes)), sizes]),
        method="highs-ds", options={"time_limit": SETTINGS["lp_seconds"], "dual_feasibility_tolerance": 1e-8, "primal_feasibility_tolerance": 1e-8})
    lp_record = {"status": int(relaxation.status), "success": bool(relaxation.success), "message": str(relaxation.message)}
    lp_x = np.empty(0)
    dual_inequality, dual_equality = np.empty(0), np.empty(0)
    lower_bound = None
    if relaxation.success:
        lp_x = relaxation.x.copy()
        dual_inequality = np.minimum(relaxation.ineqlin.marginals, 0.)
        dual_equality = relaxation.eqlin.marginals.copy()
        lower_bound, reduced = objective_bound(costs, group_u, sizes, target, width, dual_inequality, dual_equality[0])
        fractional = np.flatnonzero(np.abs(lp_x - np.rint(lp_x)) > 1e-7)
        lp_record.update(objective=float(costs @ lp_x), independently_computable_dual_lower_bound=lower_bound,
                         fractional_groups=len(fractional))
        base.emit("global_continuous_rank_lp_finished", **lp_record)
        floor = np.floor(np.clip(lp_x, 0, sizes) + 1e-8).astype("int64")
        floor = np.minimum(floor, sizes)
        floor_sum = floor @ group_u
        positive_options = np.flatnonzero(floor > 0)
        negative_options = np.flatnonzero(floor < sizes)
        positive_options = positive_options[np.argsort(np.abs(reduced[positive_options]), kind="stable")]
        negative_options = negative_options[np.argsort(np.abs(reduced[negative_options]), kind="stable")]
        for attempt, pool_count in enumerate(SETTINGS["milp_positive_pool_sizes"]):
            pool = np.unique(np.r_[fractional, positive_options[:pool_count], negative_options[:2 * pool_count]])
            fixed_sum = floor_sum - floor[pool] @ group_u[pool]
            fixed_count = int(floor.sum() - floor[pool].sum())
            desired = target - fixed_sum
            matrix = np.vstack([np.ones(len(pool)), group_u[pool].T])
            base.emit("integer_likelihood_pool_started", attempt=attempt + 1, variables=len(pool), maximum_seconds=SETTINGS["milp_seconds_per_pool"])
            fit = milp(costs[pool], integrality=np.ones(len(pool)), bounds=Bounds(np.zeros(len(pool)), sizes[pool]),
                constraints=LinearConstraint(matrix, np.r_[positives - fixed_count, desired - width],
                                             np.r_[positives - fixed_count, desired + width]),
                options={"time_limit": SETTINGS["milp_seconds_per_pool"], "mip_rel_gap": SETTINGS["milp_relative_gap"]})
            record = {"name": f"lp_near_integer_pool_{attempt + 1}", "status": int(fit.status), "message": str(fit.message),
                      "variables": len(pool), "valid_incumbent": False}
            if fit.x is not None:
                trial = floor.copy()
                trial[pool] = np.rint(fit.x).astype("int64")
                measured = .5 + (trial @ group_u) / (positives * (1 - positives / n))
                valid = int(trial.sum()) == positives and (trial >= 0).all() and (trial <= sizes).all()
                valid = valid and np.max(np.abs(measured - scores)) <= SETTINGS["auc_tolerance"] + 1e-10
                valid = valid and all(f"{v:.5f}" == f"{t:.5f}" for v, t in zip(measured, scores))
                if valid:
                    candidates.append(trial)
                    record.update(valid_incumbent=True, objective=float(costs @ trial), prior_nll=float(constant_nll + costs @ trial),
                                  candidate_column=len(candidates) - 1, binary_rank_auc=measured.tolist())
            records.append(record)
            base.emit("integer_likelihood_pool_finished", **record)
            if record["valid_incumbent"] and float(costs @ candidates[-1]) - lower_bound <= .1:
                break
    else:
        base.emit("global_continuous_rank_lp_finished", **lp_record)

    objectives = np.array([float(costs @ candidate) for candidate in candidates])
    best_column = int(np.argmin(objectives))
    counts = candidates[best_column]
    labels = expand_counts(counts, groups)
    actual = [float(roc_auc_score(labels, predictions[:, j])) for j in range(len(versions))]
    assert all(f"{v:.5f}" == f"{t:.5f}" for v, t in zip(actual, scores)), "Final hypothesis fails scored feedback"
    assert int(labels.sum()) == positives
    output = pd.DataFrame({"id": np.arange(n), "log_id": final.log_id.to_numpy(), "label": labels,
                           "estimated_click_probability": probability})
    output.to_csv(OUT / "fitted_labels_20191117.csv", index=False, float_format="%.10f")
    metrics = pd.DataFrame({"prediction_file": versions, "reported_auc": scores, "auc_using_fitted_labels": actual,
                           "reported_five_decimals": [f"{v:.5f}" for v in scores], "fitted_five_decimals": [f"{v:.5f}" for v in actual]})
    metrics.to_csv(OUT / "six_auc_fit_check.csv", index=False, float_format="%.12f")
    np.savez_compressed(OUT / "fit_arrays.npz", probability=probability, labels=labels, groups=groups,
        group_counts=counts, candidate_group_counts=np.column_stack(candidates), lp_x=lp_x,
        dual_inequality=dual_inequality, dual_equality=dual_equality)
    symmetry_bits = float(np.sum(gammaln(sizes + 1) - gammaln(counts + 1) - gammaln(sizes - counts + 1)) / np.log(2.))
    gap = None if lower_bound is None else float(objectives[best_column] - lower_bound)
    report = {"status": "complete_single_six_score_fitted_label_hypothesis", "rows": n,
        "assumed_positive_count_not_observed": positives, "zeros": int(n - positives), "ones": positives,
        "actual_roc_auc": dict(zip(versions, actual)), "reported_auc": dict(zip(versions, scores.tolist())),
        "all_six_reported_display_values_match": True, "unique_six_score_vectors": len(sizes),
        "previous_five_score_hypotheses_sixth_auc": old_checks,
        "soft_probability_projection": projection, "initial_failures": initial_failures,
        "search_records": records, "valid_integer_candidates": len(candidates), "selected_candidate_column": best_column,
        "selected_objective": float(objectives[best_column]), "initial_objective": float(objectives[0]),
        "prior_nll_constant": constant_nll, "selected_prior_nll": float(constant_nll + objectives[best_column]),
        "initial_prior_nll": float(constant_nll + objectives[0]),
        "lp_relaxation": lp_record, "certified_objective_lower_bound": lower_bound,
        "absolute_gap_to_lower_bound": gap, "global_integer_optimality_proven": bool(gap is not None and abs(gap) <= 1e-6),
        "equal_score_group_permutation_log2_count_for_selected_hypothesis": symmetry_bits,
        "model_preference_does_not_establish_true_event_labels": True,
        "actual_final_labels_recovered": False, "click_count_is_assumed_not_identified": True,
        "new_training_runs": 0, "new_platform_submission_exported": False,
        "original_data_submissions_and_previous_analyses_unchanged": True,
        "seconds": time.perf_counter() - started,
        "files_sha256": {name: base.digest(OUT / name) for name in ["fitted_labels_20191117.csv", "six_auc_fit_check.csv", "fit_arrays.npz", "scored_submission_3.csv", "initial_checkpoint.npz"]}}
    base.write(OUT / "report.json", report)
    base.emit("single_six_score_label_file_complete", rows=n, assumed_clicks=positives, actual_auc=actual,
        valid_integer_candidates=len(candidates), selected_candidate=best_column, objective_gap=gap,
        true_labels_recovered=False)


if __name__ == "__main__":
    run()
