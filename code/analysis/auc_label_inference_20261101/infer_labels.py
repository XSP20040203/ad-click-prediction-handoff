"""Conditional rank-moment inference and non-identifiability witnesses."""
import os
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "3"
from pathlib import Path
import json
import hashlib
import math
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import expit, logit, gammaln
from scipy.stats import rankdata
from scipy.optimize import minimize, milp, LinearConstraint, Bounds
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/label_hypotheses_20191117"
VERSIONS = ["v6", "v7", "v8", "v9"]
SCORES = np.array([.80817, .80581, .80745, .80856])
FILES = {"v6": ("output/best_nextday_full_v6/submission.csv", "899f6f6f48bae40ec3920d7948ace3d26f478e839c865fbe3006c90c565f9e5f"),
    "v7": ("output/best_nextday_arch_v7/submission.csv", "c44a9e803b8b4566b9bcef4e1bf85181c2ee00eee71a9f84474b73c619c121b9"),
    "v8": ("output/best_nextday_daystate_v8/submission.csv", "1b43ee597613ec6a48d448a2cb00b824aca3440e38a32d47e21ceadccd833846"),
    "v9": ("output/platform_feedback_v9/submission_v9.csv", "c55583694642524fbabe71b5a0d4d06e86582076e899d1087d1d9dfd73271a83")}
SETTINGS = {"score_decimals": 5, "auc_match_tolerance": .000004,
    "click_count_sensitivity_factors": [.85, 1., 1.15], "seed_a": 91271, "seed_c": 81431,
    "maximum_greedy_steps": 240, "milp_seconds_each": 15., "milp_attempts": 3,
    "scope": "Conditional estimates and compatible label hypotheses; not recovered ground truth"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def emit(event, **data):
    print(json.dumps({"event": event, **data}), flush=True)


def rank_features(predictions):
    n = len(predictions)
    ranks = np.column_stack([rankdata(predictions[:, j], method="average") for j in range(predictions.shape[1])])
    return (ranks - (n + 1) / 2) / n


def targets(n, positives, auc):
    return positives * (1 - positives / n) * (np.asarray(auc) - .5)


def rank_auc(labels, centered_ranks):
    n, p = len(labels), int(np.sum(labels))
    if p <= 0 or p >= n:
        raise ValueError("Both binary classes required")
    return .5 + n * centered_ranks[np.asarray(labels, dtype=bool)].sum(0) / (p * (n - p))


def conditioned_probabilities(prior, centered_ranks, auc, positives):
    n = len(prior)
    covariance = centered_ranks.T @ centered_ranks / n
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues.min() <= 1e-12:
        raise ValueError("Rank constraints are numerically dependent")
    whitening = eigenvectors / np.sqrt(eigenvalues)[None, :]
    z = centered_ranks @ whitening
    design = np.column_stack([np.ones(n), z])
    target = np.r_[positives, targets(n, positives, auc) @ whitening]
    original = logit(np.clip(np.asarray(prior), 1e-9, 1 - 1e-9))
    def objective(theta):
        eta = original + design @ theta
        probability = expit(eta)
        return float((np.logaddexp(0., eta).sum() - theta @ target) / n), (design.T @ probability - target) / n
    def hessian(theta):
        probability = expit(original + design @ theta)
        return design.T @ (design * (probability * (1 - probability))[:, None]) / n
    initial = np.zeros(design.shape[1])
    initial[0] = logit(positives / n) - logit(float(np.mean(prior)))
    fit = minimize(objective, initial, jac=True, hess=hessian, method="trust-exact",
                   options={"maxiter": 100, "gtol": 1e-11})
    probability = expit(original + design @ fit.x)
    implied = .5 + n * (probability @ centered_ranks) / (positives * (n - positives))
    error = float(np.max(np.abs(implied - auc)))
    count_error = float(abs(probability.sum() - positives))
    if error > 1e-7 or count_error > .001:
        raise RuntimeError("Rank moment projection did not converge")
    info = {"assumed_positive_count": int(positives), "expected_positive_count": float(probability.sum()),
        "expected_rank_moments_not_binary_auc": implied.tolist(), "maximum_expected_rank_moment_error": error,
        "theta": fit.x.tolist(), "optimizer_success": bool(fit.success), "optimizer_message": str(fit.message),
        "probability_min": float(probability.min()), "probability_max": float(probability.max()),
        "rows_probability_at_least_half": int((probability >= .5).sum())}
    return probability, whitening, info


def weighted_negative_candidates(rng, probability, selected, count):
    weighted = rng.choice(len(probability), size=min(count, len(probability)), replace=False, p=probability / probability.sum())
    uniform = rng.integers(0, len(probability), size=count // 2)
    candidates = np.unique(np.r_[weighted, uniform])
    return candidates[~selected[candidates]]


def construct_hypothesis(probability, u, whitening, auc, positives, seed):
    started = time.perf_counter()
    n = len(probability)
    rng = np.random.default_rng(seed)
    selected = np.zeros(n, dtype=bool)
    pos = rng.choice(n, size=positives, replace=False, p=probability / probability.sum())
    selected[pos] = True
    z = u @ whitening
    target_raw = targets(n, positives, auc)
    target_z = target_raw @ whitening
    sum_raw, sum_z = u[pos].sum(0), z[pos].sum(0)
    tolerance = SETTINGS["auc_match_tolerance"]
    greedy_steps = 0
    for step in range(SETTINGS["maximum_greedy_steps"]):
        observed = .5 + n * sum_raw / (positives * (n - positives))
        if float(np.max(np.abs(observed - auc))) <= tolerance:
            break
        scale = 2 if step > 140 else 1
        positions = rng.choice(len(pos), size=min(len(pos), 192 * scale), replace=False)
        positive_ids = pos[positions]
        negative_ids = weighted_negative_candidates(rng, probability, selected, 4096 * scale)
        residual = sum_z - target_z
        adjusted = z[negative_ids] + residual
        positive_vectors = z[positive_ids]
        costs = np.square(adjusted).sum(1)[:, None] + np.square(positive_vectors).sum(1)[None, :] - 2 * adjusted @ positive_vectors.T
        index = np.unravel_index(np.argmin(costs), costs.shape)
        if float(costs[index]) >= float(residual @ residual) - 1e-12:
            if step > 30:
                break
            continue
        old, new = int(positive_ids[index[1]]), int(negative_ids[index[0]])
        selected[old], selected[new] = False, True
        pos[positions[index[1]]] = new
        sum_raw += u[new] - u[old]
        sum_z += z[new] - z[old]
        greedy_steps += 1
    repairs = []
    for attempt in range(SETTINGS["milp_attempts"]):
        observed = rank_auc(selected, u)
        if np.max(np.abs(observed - auc)) <= tolerance:
            break
        current_positive = np.flatnonzero(selected)
        adjustable_positive = rng.choice(current_positive, size=min(len(current_positive), 256 * (attempt + 1)), replace=False)
        negative = weighted_negative_candidates(rng, probability, selected, 1024 * (attempt + 1))
        pool = np.unique(np.r_[adjustable_positive, negative])
        fixed_sum = u[selected].sum(0) - u[adjustable_positive].sum(0)
        desired = target_raw - fixed_sum
        width = positives * (1 - positives / n) * tolerance
        matrix = np.vstack([np.ones(len(pool)), u[pool].T])
        lower = np.r_[len(adjustable_positive), desired - width]
        upper = np.r_[len(adjustable_positive), desired + width]
        result = milp(np.zeros(len(pool)), integrality=np.ones(len(pool)), bounds=Bounds(0., 1.),
            constraints=LinearConstraint(matrix, lower, upper),
            options={"time_limit": SETTINGS["milp_seconds_each"], "mip_rel_gap": .1, "presolve": True})
        repairs.append({"attempt": attempt + 1, "status": int(result.status), "message": str(result.message), "variables": len(pool)})
        if result.x is not None:
            candidate = selected.copy()
            candidate[pool] = np.rint(result.x).astype(bool)
            if int(candidate.sum()) == positives and np.max(np.abs(rank_auc(candidate, u) - auc)) <= tolerance + 1e-9:
                selected = candidate
                break
    observed = rank_auc(selected, u)
    if int(selected.sum()) != positives or any(f"{value:.5f}" != f"{target:.5f}" for value, target in zip(observed, auc)):
        raise RuntimeError("No binary hypothesis matches every published rounded AUC")
    return selected.astype("int8"), {"seed": seed, "positives": positives,
        "binary_rank_auc": observed.tolist(), "greedy_swaps": greedy_steps,
        "integer_repairs": repairs, "seconds": time.perf_counter() - started}


def exchange_equal_score_labels(labels, groups):
    result = labels.copy()
    order = np.argsort(groups, kind="stable")
    ordered_groups = groups[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_groups)) + 1]
    stops = np.r_[starts[1:], len(order)]
    for start, stop in zip(starts, stops):
        members = order[start:stop]
        count = int(labels[members].sum())
        if 0 < count < len(members):
            alternatives = np.r_[members[labels[members] == 0], members[labels[members] == 1]]
            result[members] = 0
            result[alternatives[:count]] = 1
    return result


def backtest():
    sys.path.insert(0, str(ROOT / "analysis/platform_delta_blend_20261101"))
    from blend_rules import development_arrays
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d", "label"], dtype="int32")
    records = []
    for day in [15, 16]:
        rows, predictions, _ = development_arrays(day)
        labels = raw.label.to_numpy()[rows]
        p9 = .75 * predictions["v6"] + .25 * predictions["v8"]
        submitted = np.column_stack([np.round(predictions["v6"], 6), np.round(predictions["v7"], 6),
                                     np.round(predictions["v8"], 10), np.round(p9, 10)])
        aggregate = np.array([float(f"{roc_auc_score(labels, submitted[:, j]):.5f}") for j in range(4)])
        count = int(labels.sum())
        inferred, _, projection = conditioned_probabilities(submitted[:, 3], rank_features(submitted), aggregate, count)
        def diagnostics(values):
            chosen = np.argpartition(values, -count)[-count:]
            positives_found = int(labels[chosen].sum())
            return {"auc": float(roc_auc_score(labels, values)), "logloss": float(log_loss(labels, values)),
                "top_assumed_click_count_precision": positives_found / count,
                "actual_clicks_found_in_top_count": positives_found, "actual_clicks": count}
        records.append({"day": 20191100 + day, "prior": diagnostics(submitted[:, 3]), "conditioned": diagnostics(inferred),
            "known_true_positive_count_given_to_inference": True, "rounded_oracle_auc_given_to_inference": aggregate.tolist(),
            "not_an_independent_model_validation": True, "projection": projection})
        emit("oracle_feedback_backtest", **records[-1])
    return records


def run():
    if OUT.exists():
        raise FileExistsError("Preserving label-inference artifacts")
    OUT.mkdir(parents=True)
    final = pd.read_csv(ROOT / "2026ml/final_test.csv")
    if "label" in final or not final.pt_d.eq(20191117).all():
        raise AssertionError("Inspect unexpected final label/date schema")
    train = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d", "label", "user_id", "task_id"], dtype="int32")
    daily = train.groupby("pt_d").label.agg(["size", "sum"])
    complete = daily.loc[daily["size"] >= 100000]
    recent = complete.tail(2)
    recent_rate = float(recent["sum"].sum() / recent["size"].sum())
    counts = [int(round(len(final) * recent_rate * factor)) for factor in SETTINGS["click_count_sensitivity_factors"]]
    predictions = []
    for name in VERSIONS:
        relative, expected = FILES[name]
        path = ROOT / relative
        if digest(path) != expected:
            raise AssertionError("Scored prediction file differs")
        frame = pd.read_csv(path, float_precision="round_trip")
        np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(len(final)))
        predictions.append(frame.pctr.to_numpy(dtype="float64"))
    predictions = np.column_stack(predictions)
    u = rank_features(predictions)
    _, groups, group_sizes = np.unique(predictions, axis=0, return_inverse=True, return_counts=True)
    audit = {"train_rows": len(train), "train_clicks": int(train.label.sum()), "final_rows": len(final),
        "final_has_label_column": False, "final_users_unseen_in_training": int((~final.user_id.isin(train.user_id.unique())).sum()),
        "final_tasks_unseen_in_training": int((~final.task_id.isin(train.task_id.unique())).sum()),
        "final_repeated_predictor_rows": int(final.drop(columns=["log_id"]).duplicated().sum()),
        "daily_training": {str(day): {"rows": int(row['size']), "clicks": int(row['sum']), "ctr": float(row['sum'] / row['size'])} for day, row in daily.iterrows()},
        "recent_two_complete_day_ctr": recent_rate, "click_count_assumptions_low_central_high": counts,
        "unique_four_score_vectors": len(group_sizes), "rows_in_nonunique_score_groups": int(group_sizes[group_sizes > 1].sum()),
        "maximum_equal_score_group_size": int(group_sizes.max())}
    write(OUT / "dataset_audit.json", audit)
    write(OUT / "protocol.json", {"settings": SETTINGS, "auc_versions": VERSIONS, "reported_auc": SCORES.tolist(),
        "scores_apply_to_all_rows_confirmed_by_user": True, "ordinary_global_roc_auc_assumed": True,
        "nearest_five_decimal_rounding_assumed": True, "true_positive_count_unknown": True,
        "click_count_is_estimated_from_recent_training_dates": True,
        "prior": "Actual v9 scored probabilities", "source_sha256": digest(__file__),
        "test_sha256": digest(HERE / "test_inference.py"), "scored_file_hashes": FILES,
        "verifier_sha256": digest(HERE / "verify_inference.py"),
        "original_datasets_sha256": {name: digest(ROOT / "2026ml" / name) for name in ["train.csv", "final_test.csv"]},
        "output_is_not_ground_truth": True})
    emit("label_inference_dataset_audited", **audit)
    probabilities, projections = [], []
    whitening = None
    for count in counts:
        q, whitening, info = conditioned_probabilities(predictions[:, 3], u, SCORES, count)
        probabilities.append(q)
        projections.append(info)
        emit("auc_rank_moments_conditioned", **info)
    label_a, info_a = construct_hypothesis(probabilities[1], u, whitening, SCORES, counts[1], SETTINGS["seed_a"])
    emit("compatible_label_hypothesis_a", **info_a)
    label_b = exchange_equal_score_labels(label_a, groups)
    label_c, info_c = construct_hypothesis(probabilities[1], u, whitening, SCORES, counts[1], SETTINGS["seed_c"])
    emit("compatible_label_hypothesis_c", **info_c)
    hypotheses = np.column_stack([label_a, label_b, label_c])
    actual_scores = {}
    for index, name in enumerate(["A", "B", "C"]):
        measured = np.array([roc_auc_score(hypotheses[:, index], predictions[:, j]) for j in range(4)])
        if any(f"{value:.5f}" != f"{target:.5f}" for value, target in zip(measured, SCORES)):
            raise AssertionError("Constructed binary hypothesis does not reproduce all four rounded metrics")
        actual_scores[name] = dict(zip(VERSIONS, measured.tolist()))
    group_clicks = np.bincount(groups, weights=label_a, minlength=len(group_sizes)).astype("int64")
    symmetry_bits = float(np.sum(gammaln(group_sizes + 1) - gammaln(group_clicks + 1) - gammaln(group_sizes - group_clicks + 1)) / np.log(2))
    overlap_ac = int(((label_a == 1) & (label_c == 1)).sum())
    inferred = pd.DataFrame({"id": np.arange(len(final)), "log_id": final.log_id.to_numpy(),
        "click_probability_low_count": probabilities[0], "click_probability_central": probabilities[1],
        "click_probability_high_count": probabilities[2], "label_hypothesis_A": label_a,
        "label_hypothesis_B": label_b, "label_hypothesis_C": label_c,
        "equal_four_score_group_size": group_sizes[groups]})
    inferred.to_csv(OUT / "estimated_labels_20191117.csv", index=False, float_format="%.10f")
    np.savez_compressed(OUT / "inference_arrays.npz", probabilities=np.column_stack(probabilities), hypotheses=hypotheses, groups=groups)
    historical = backtest()
    report = {"status": "complete_conditional_hypotheses_not_truth", "assumed_click_count": counts[1],
        "rounded_auc_matches": actual_scores, "construction_A": info_a, "construction_C": info_c,
        "A_B_different_rows": int((label_a != label_b).sum()), "A_C_different_rows": int((label_a != label_c).sum()),
        "A_C_common_positive_rows": overlap_ac, "A_C_positive_overlap_fraction": overlap_ac / counts[1],
        "equal_score_swap_log2_number_of_compatible_labels": symmetry_bits,
        "conditional_probabilities": projections, "oracle_feedback_backtests": historical,
        "probability_estimates_depend_on_prior_and_click_count_assumptions": True,
        "label_hypotheses_are_not_recovered_actual_labels": True,
        "source_files_and_datasets_unchanged": True,
        "files_sha256": {name: digest(OUT / name) for name in ["estimated_labels_20191117.csv", "inference_arrays.npz"]}}
    write(OUT / "report.json", report)
    emit("label_inference_completed", assumed_clicks=counts[1], A_B_different_rows=report["A_B_different_rows"],
         A_C_different_rows=report["A_C_different_rows"], exact_score_symmetry_bits=symmetry_bits,
         true_labels_recovered=False)


if __name__ == "__main__":
    run()
