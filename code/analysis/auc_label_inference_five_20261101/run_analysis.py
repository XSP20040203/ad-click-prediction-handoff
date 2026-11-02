"""Add a fifth scored rank constraint without treating witnesses as truth."""
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
from scipy.special import gammaln
from scipy.optimize import milp, Bounds, LinearConstraint
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD_HERE = ROOT / "analysis/auc_label_inference_20261101"
OLD_OUT = ROOT / "output/label_hypotheses_20191117"
OUT = ROOT / "output/label_hypotheses_five_scores_20191117"
sys.path.insert(0, str(OLD_HERE))
import infer_labels as base

VERSIONS = base.VERSIONS + ["old_best"]
SCORES = np.r_[base.SCORES, .74404]
FILES = dict(base.FILES)
FILES["old_best"] = ("output/submission_best.csv", "45d0571bd12d3176c56bbeb16704a938ed8683909169daffcd860206b057f828")
USER_FILE = Path("E:/浏览器下载/submission_best.csv")
SETTINGS = {
    "assumed_click_counts": [2940, 3459, 3978],
    "prior": "Actual scored v9 probabilities; unchanged from four-score inference",
    "binary_seeds": [base.SETTINGS["seed_a"], base.SETTINGS["seed_c"]],
    "binary_auc_tolerance": base.SETTINGS["auc_match_tolerance"],
    "opposite_queries": 20,
    "opposite_seed": 701531,
    "opposite_greedy_steps": 120,
    "opposite_milp_attempts": 2,
    "opposite_milp_seconds": 10.,
}


def check_scores(labels, predictions):
    measured = np.array([roc_auc_score(labels, predictions[:, j]) for j in range(predictions.shape[1])])
    return measured, all(f"{value:.5f}" == f"{target:.5f}" for value, target in zip(measured, SCORES))


def label_summary(labels, groups, sizes):
    counts = np.bincount(groups, weights=labels, minlength=len(sizes)).astype("int64")
    mixed = (counts > 0) & (counts < sizes)
    bits = float(np.sum(gammaln(sizes + 1) - gammaln(counts + 1) - gammaln(sizes - counts + 1)) / np.log(2.))
    return {"within_group_permutation_log2_count": bits,
            "rows_in_mixed_label_equal_score_groups": int(sizes[mixed].sum()),
            "mixed_label_equal_score_groups": int(mixed.sum())}


def opposite_witness(labels, row, probability, u, z, whitening, groups, seed):
    """Return a certificate, or search failure with no forced-label claim."""
    rng = np.random.default_rng(seed)
    n, positives = len(labels), int(labels.sum())
    selected = labels.astype(bool).copy()
    desired_value = not selected[row]
    tolerance = SETTINGS["binary_auc_tolerance"]
    alternatives = np.flatnonzero((groups == groups[row]) & (selected != selected[row]))
    if len(alternatives):
        other = int(alternatives[0])
        selected[row], selected[other] = selected[other], selected[row]
        return selected.astype("int8"), {"method": "exact_equal_five_score_swap", "swaps": 1}

    if selected[row]:
        candidates = base.weighted_negative_candidates(rng, probability, selected, 12000)
    else:
        candidates = np.flatnonzero(selected)
    distance = np.square(z[candidates] - z[row]).sum(axis=1)
    other = int(candidates[np.argmin(distance)])
    selected[row], selected[other] = desired_value, not desired_value
    target_raw = base.targets(n, positives, SCORES)
    target_z = target_raw @ whitening
    sum_raw, sum_z = u[selected].sum(0), z[selected].sum(0)
    swaps = 1
    repairs = []
    for step in range(SETTINGS["opposite_greedy_steps"]):
        observed = .5 + sum_raw / (positives * (1 - positives / n))
        if np.max(np.abs(observed - SCORES)) <= tolerance:
            break
        positive_ids = np.flatnonzero(selected)
        positive_ids = positive_ids[positive_ids != row]
        if len(positive_ids) > 384:
            positive_ids = rng.choice(positive_ids, size=384, replace=False)
        negative_ids = base.weighted_negative_candidates(rng, probability, selected, 8192)
        negative_ids = negative_ids[negative_ids != row]
        residual = sum_z - target_z
        adjusted = z[negative_ids] + residual
        vectors = z[positive_ids]
        costs = np.square(adjusted).sum(1)[:, None] + np.square(vectors).sum(1)[None, :] - 2 * adjusted @ vectors.T
        index = np.unravel_index(np.argmin(costs), costs.shape)
        if float(costs[index]) >= float(residual @ residual) - 1e-12:
            if step >= 40:
                break
            continue
        old, new = int(positive_ids[index[1]]), int(negative_ids[index[0]])
        selected[old], selected[new] = False, True
        sum_raw += u[new] - u[old]
        sum_z += z[new] - z[old]
        swaps += 1

    for attempt in range(SETTINGS["opposite_milp_attempts"]):
        if np.max(np.abs(base.rank_auc(selected, u) - SCORES)) <= tolerance:
            break
        available = np.flatnonzero(selected)
        available = available[available != row]
        positive_ids = rng.choice(available, min(len(available), 256 * (attempt + 1)), replace=False)
        negative_ids = base.weighted_negative_candidates(rng, probability, selected, 2048 * (attempt + 1))
        negative_ids = negative_ids[negative_ids != row]
        pool = np.unique(np.r_[positive_ids, negative_ids])
        fixed_sum = u[selected].sum(0) - u[positive_ids].sum(0)
        desired = target_raw - fixed_sum
        width = positives * (1 - positives / n) * tolerance
        matrix = np.vstack([np.ones(len(pool)), u[pool].T])
        result = milp(np.zeros(len(pool)), integrality=np.ones(len(pool)), bounds=Bounds(0., 1.),
                      constraints=LinearConstraint(matrix, np.r_[len(positive_ids), desired - width],
                                                   np.r_[len(positive_ids), desired + width]),
                      options={"time_limit": SETTINGS["opposite_milp_seconds"], "mip_rel_gap": .1})
        repairs.append({"status": int(result.status), "variables": len(pool), "message": str(result.message)})
        if result.x is not None:
            trial = selected.copy()
            trial[pool] = np.rint(result.x).astype(bool)
            if int(trial.sum()) == positives and np.max(np.abs(base.rank_auc(trial, u) - SCORES)) <= tolerance + 1e-9:
                selected = trial
                break
    matches = all(f"{value:.5f}" == f"{target:.5f}" for value, target in zip(base.rank_auc(selected, u), SCORES))
    success = selected[row] == desired_value and int(selected.sum()) == positives and matches
    info = {"method": "pinned_row_fixed_count_rank_repair", "swaps": swaps, "integer_repairs": repairs,
            "search_success": bool(success), "search_failure_does_not_prove_forced_label": True}
    return (selected.astype("int8") if success else None), info


def run():
    started = time.perf_counter()
    old_protocol = json.loads((OLD_OUT / "protocol.json").read_text(encoding="utf-8"))
    old_report = json.loads((OLD_OUT / "report.json").read_text(encoding="utf-8"))
    old_audit = json.loads((OLD_OUT / "dataset_audit.json").read_text(encoding="utf-8"))
    if base.digest(OLD_HERE / "infer_labels.py") != old_protocol["source_sha256"]:
        raise AssertionError("Frozen numerical source changed")
    if base.digest(OLD_OUT / "inference_arrays.npz") != old_report["files_sha256"]["inference_arrays.npz"]:
        raise AssertionError("Old inference output changed")
    if base.digest(USER_FILE) != FILES["old_best"][1]:
        raise AssertionError("New human-scored attachment changed")
    for name, expected in old_protocol["original_datasets_sha256"].items():
        if base.digest(ROOT / "2026ml" / name) != expected:
            raise AssertionError("Raw dataset changed")
    final = pd.read_csv(ROOT / "2026ml/final_test.csv", usecols=["log_id", "pt_d"])
    n = len(final)
    if n != old_audit["final_rows"] or not final.pt_d.eq(20191117).all():
        raise AssertionError("Unexpected final rows/dates")
    predictions = []
    for version in VERSIONS:
        path, expected = FILES[version]
        if base.digest(ROOT / path) != expected:
            raise AssertionError("Scored source changed")
        frame = pd.read_csv(ROOT / path, float_precision="round_trip")
        np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(n))
        p = frame.pctr.to_numpy(dtype="float64")
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise AssertionError("Invalid submitted probabilities")
        predictions.append(p)
    predictions = np.column_stack(predictions)
    u = base.rank_features(predictions)
    _, groups, sizes = np.unique(predictions, axis=0, return_inverse=True, return_counts=True)
    with np.load(OLD_OUT / "inference_arrays.npz", allow_pickle=False) as archive:
        old_labels = archive["hypotheses"].copy()
        old_probability = archive["probabilities"].copy()
        old_groups = archive["groups"].copy()
    old_recheck = {name: float(roc_auc_score(old_labels[:, j], predictions[:, 4])) for j, name in enumerate(["A", "B", "C"])}
    design4 = np.column_stack([np.ones(n), u[:, :4]])
    regression = np.linalg.lstsq(design4, u[:, 4], rcond=None)[0]
    residual = u[:, 4] - design4 @ regression
    old_sizes = np.bincount(old_groups)
    audit = {
        "rows": n, "date": 20191117, "new_submission_matches_user_file": True,
        "fifth_score_user_reported": float(SCORES[4]), "old_hypotheses_fifth_auc": old_recheck,
        "old_hypotheses_ruled_out": {name: f"{value:.5f}" != f"{SCORES[4]:.5f}" for name, value in old_recheck.items()},
        "four_score_unique_vectors": len(old_sizes), "five_score_unique_vectors": len(sizes),
        "four_score_rows_in_nonunique_groups": int(old_sizes[old_sizes > 1].sum()),
        "five_score_rows_in_nonunique_groups": int(sizes[sizes > 1].sum()),
        "five_score_maximum_group_size": int(sizes.max()),
        "rank_correlation_with_four_scores": np.corrcoef(u.T)[4, :4].tolist(),
        "fifth_rank_variance_not_explained_linearly_by_four": float(np.var(residual) / np.var(u[:, 4])),
        "new_rank_variation_is_not_a_per_row_label_confidence": True,
        "true_final_positive_count_unknown": True,
    }
    if SETTINGS["assumed_click_counts"] != old_audit["click_count_assumptions_low_central_high"]:
        raise AssertionError("Click-count scenarios must stay fixed")
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(USER_FILE, OUT / "scored_submission_best.csv")
    protocol = {
        "settings": SETTINGS, "generic_binary_settings": base.SETTINGS,
        "auc_versions": VERSIONS, "reported_auc": SCORES.tolist(), "scored_files_sha256": FILES,
        "scores_apply_to_all_rows_confirmed_by_user": True,
        "ordinary_global_auc_and_nearest_five_decimal_rounding_assumed": True,
        "source_sha256": base.digest(__file__), "verifier_sha256": base.digest(HERE / "verify_analysis.py"),
        "specification_sha256": base.digest(HERE / "PROTOCOL.md"),
        "generic_source_sha256": base.digest(OLD_HERE / "infer_labels.py"),
        "original_datasets_sha256": old_protocol["original_datasets_sha256"],
        "old_output_files_sha256": {name: base.digest(OLD_OUT / name) for name in ["report.json", "protocol.json", "inference_arrays.npz", "estimated_labels_20191117.csv", "verification.json"]},
        "model_prior_and_click_count_dependent": True, "output_is_not_ground_truth": True,
        "does_not_resume_paused_auc_goal": True,
    }
    base.write(OUT / "protocol.json", protocol)
    base.write(OUT / "dataset_comparison.json", audit)
    base.emit("fifth_score_input_audited", **audit)
    probabilities, projections = [], []
    for count in SETTINGS["assumed_click_counts"]:
        q, whitening, info = base.conditioned_probabilities(predictions[:, 3], u, SCORES, count)
        probabilities.append(q)
        projections.append(info)
        base.emit("five_score_probability_projection", **info)
    probabilities = np.column_stack(probabilities)
    count = SETTINGS["assumed_click_counts"][1]
    label_a, info_a = base.construct_hypothesis(probabilities[:, 1], u, whitening, SCORES, count, SETTINGS["binary_seeds"][0])
    base.emit("five_score_binary_A", **info_a)
    label_b = base.exchange_equal_score_labels(label_a, groups)
    label_c, info_c = base.construct_hypothesis(probabilities[:, 1], u, whitening, SCORES, count, SETTINGS["binary_seeds"][1])
    base.emit("five_score_binary_C", **info_c)
    hypotheses = np.column_stack([label_a, label_b, label_c])
    actual = {}
    for j, name in enumerate(["A", "B", "C"]):
        measured, matches = check_scores(hypotheses[:, j], predictions)
        if not matches:
            raise AssertionError("Five-score binary witness invalid")
        actual[name] = dict(zip(VERSIONS, measured.tolist()))
    delta = probabilities[:, 1] - old_probability[:, 1]
    top_rows = np.argsort(-probabilities[:, 1], kind="stable")[:SETTINGS["opposite_queries"]]
    old_top = np.argpartition(old_probability[:, 1], -count)[-count:]
    new_top = np.argpartition(probabilities[:, 1], -count)[-count:]
    comparison = {
        "mean_absolute_probability_change": float(np.mean(np.abs(delta))),
        "maximum_absolute_probability_change": float(np.max(np.abs(delta))),
        "sum_absolute_probability_change": float(np.sum(np.abs(delta))),
        "top_assumed_click_count": count,
        "top_count_overlap_rows": int(np.intersect1d(old_top, new_top).size),
        "top_count_replaced_rows": int(count - np.intersect1d(old_top, new_top).size),
        "four_score_probability_at_least_half": int((old_probability[:, 1] >= .5).sum()),
        "five_score_probability_at_least_half": int((probabilities[:, 1] >= .5).sum()),
        "estimate_changes_are_not_measured_label_accuracy_gains": True,
    }
    base.emit("conditional_estimate_comparison", **comparison)
    z = u @ whitening
    opposite_columns, opposite_records = [], []
    for query_index, row in enumerate(top_rows):
        witness, search = opposite_witness(label_a, int(row), probabilities[:, 1], u, z, whitening, groups,
                                           SETTINGS["opposite_seed"] + query_index)
        record = {"id": int(row), "log_id": int(final.log_id.iloc[row]),
                  "probability_four_scores": float(old_probability[row, 1]),
                  "probability_five_scores": float(probabilities[row, 1]),
                  "base_hypothesis_label": int(label_a[row]), "requested_opposite_label": int(1 - label_a[row]),
                  "opposite_feasible_assignment_found": witness is not None, "search": search}
        if witness is not None:
            measured, matches = check_scores(witness, predictions)
            if not matches or witness[row] == label_a[row] or int(witness.sum()) != count:
                raise AssertionError("Pinned opposite-label witness invalid")
            record["actual_auc"] = dict(zip(VERSIONS, measured.tolist()))
            record["witness_column"] = len(opposite_columns)
            opposite_columns.append(witness)
        opposite_records.append(record)
        base.emit("top_probability_opposite_label_checked", **record)
    opposite_matrix = np.column_stack(opposite_columns) if opposite_columns else np.empty((n, 0), dtype="int8")
    inferred = pd.DataFrame({"id": np.arange(n), "log_id": final.log_id.to_numpy(),
        "click_probability_low_count": probabilities[:, 0], "click_probability_central": probabilities[:, 1],
        "click_probability_high_count": probabilities[:, 2], "label_hypothesis_A": label_a,
        "label_hypothesis_B": label_b, "label_hypothesis_C": label_c,
        "equal_five_score_group_size": sizes[groups]})
    inferred.to_csv(OUT / "conditional_estimates_20191117.csv", index=False, float_format="%.10f")
    pd.DataFrame([{key: value for key, value in record.items() if key not in ["search", "actual_auc"]} for record in opposite_records]).to_csv(OUT / "top20_label_ambiguity.csv", index=False)
    np.savez_compressed(OUT / "inference_arrays.npz", probabilities=probabilities, hypotheses=hypotheses, groups=groups,
                        opposite_hypotheses=opposite_matrix, queried_rows=top_rows)
    report = {
        "status": "complete_conditional_estimates_and_ambiguity_witnesses",
        "assumed_click_count_not_observed": count, "sensitivity_click_counts_not_observed": SETTINGS["assumed_click_counts"],
        "dataset_comparison": audit, "probability_comparison": comparison,
        "conditional_projections": projections, "actual_auc_by_hypothesis": actual,
        "construction_A": info_a, "construction_C": info_c,
        "A_B_different_rows": int((label_a != label_b).sum()),
        "A_C_different_rows": int((label_a != label_c).sum()),
        "A_C_common_positive_rows": int(((label_a == 1) & (label_c == 1)).sum()),
        "exact_score_group_ambiguity": label_summary(label_a, groups, sizes),
        "opposite_label_checks": opposite_records,
        "opposite_checks_successful": len(opposite_columns), "opposite_checks_total": len(top_rows),
        "no_proven_true_individual_labels": True, "actual_final_labels_recovered": False,
        "new_model_trained": False, "new_submission_exported": False,
        "source_files_and_datasets_unchanged": True,
        "seconds": time.perf_counter() - started,
        "files_sha256": {name: base.digest(OUT / name) for name in ["conditional_estimates_20191117.csv", "top20_label_ambiguity.csv", "inference_arrays.npz", "scored_submission_best.csv"]},
    }
    base.write(OUT / "report.json", report)
    base.emit("five_score_analysis_complete", A_B_different_rows=report["A_B_different_rows"],
              A_C_different_rows=report["A_C_different_rows"], opposite_checks_successful=len(opposite_columns),
              opposite_checks_total=len(top_rows), actual_labels_recovered=False)


if __name__ == "__main__":
    run()
