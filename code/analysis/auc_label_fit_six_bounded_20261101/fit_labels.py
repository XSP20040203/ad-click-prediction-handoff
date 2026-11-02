"""Bounded entropy annealing and feasible binary rank-moment fitting."""
import os
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "3"
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/fitted_labels_six_scores_final_20191117"
GENERIC = ROOT / "analysis/auc_label_inference_20261101"
sys.path.insert(0, str(GENERIC))
import infer_labels as base
SETTINGS = {"assumed_click_count": 3459, "auc_tolerance": 4e-6,
    "temperatures": [.5, .25, .125, .0625, .03125], "iterations_per_temperature": 80,
    "seeds": [12071, 83519], "maximum_repair_swaps": 180,
    "preference": "Lowest actual v9 Bernoulli NLL among searched six-score feasible candidates",
    "not_a_proven_global_optimum": True}


def repair(q, logits, u, z, whitening, scores, seed):
    rng = np.random.default_rng(seed)
    n, count = len(q), SETTINGS["assumed_click_count"]
    chosen = np.argpartition(logits + rng.gumbel(size=n), -count)[-count:]
    selected = np.zeros(n, dtype=bool)
    selected[chosen] = True
    target = base.targets(n, count, scores)
    target_z = target @ whitening
    sum_raw, sum_z = u[selected].sum(0), z[selected].sum(0)
    uncertainty = q * (1 - q) + 1e-8
    uncertainty /= uncertainty.sum()
    swaps = 0
    for step in range(SETTINGS["maximum_repair_swaps"]):
        measured = .5 + sum_raw / (count * (1 - count / n))
        if np.max(np.abs(measured - scores)) <= SETTINGS["auc_tolerance"]:
            break
        positive = np.flatnonzero(selected)
        if len(positive) > 384:
            weights = uncertainty[positive]
            positive = rng.choice(positive, 384, replace=False, p=weights / weights.sum())
        negative = rng.choice(n, min(8192, n), replace=False, p=uncertainty)
        negative = np.unique(np.r_[negative, rng.integers(0, n, 1024)])
        negative = negative[~selected[negative]]
        residual = sum_z - target_z
        adjusted, other = z[negative] + residual, z[positive]
        costs = np.square(adjusted).sum(1)[:, None] + np.square(other).sum(1)[None, :] - 2 * adjusted @ other.T
        ii, jj = np.unravel_index(np.argmin(costs), costs.shape)
        if costs[ii, jj] >= residual @ residual - 1e-12:
            if step >= 35:
                break
            continue
        old, new = int(positive[jj]), int(negative[ii])
        selected[old], selected[new] = False, True
        sum_raw += u[new] - u[old]
        sum_z += z[new] - z[old]
        swaps += 1
    actual = base.rank_auc(selected, u)
    valid = np.max(np.abs(actual - scores)) <= SETTINGS["auc_tolerance"] + 1e-10
    valid = valid and all(f"{v:.5f}" == f"{s:.5f}" for v, s in zip(actual, scores))
    return selected.astype("int8") if valid else None, {"seed": seed, "swaps": swaps, "valid": bool(valid), "rank_auc": actual.tolist()}


def run():
    started = time.perf_counter()
    input_protocol_path = ROOT / "output/fitted_labels_six_scores_v2_20191117/protocol.json"
    source = json.loads(input_protocol_path.read_text(encoding="utf-8"))
    assert base.digest(GENERIC / "infer_labels.py") == source["generic_source_sha256"]
    for filename, expected in source["original_datasets_sha256"].items():
        assert base.digest(ROOT / "2026ml" / filename) == expected
    final = pd.read_csv(ROOT / "2026ml/final_test.csv", usecols=["log_id", "pt_d"])
    n, count = len(final), SETTINGS["assumed_click_count"]
    assert n == 404182 and final.pt_d.eq(20191117).all()
    predictions = []
    for version in source["versions"]:
        filename, expected = source["scored_files"][version]
        path = Path(filename) if Path(filename).is_absolute() else ROOT / filename
        assert base.digest(path) == expected
        frame = pd.read_csv(path, float_precision="round_trip")
        np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(n))
        predictions.append(frame.pctr.to_numpy(dtype="float64"))
    predictions = np.column_stack(predictions)
    assert np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all()
    scores = np.array(source["reported_auc"])
    u = base.rank_features(predictions)
    eigenvalues, eigenvectors = np.linalg.eigh(u.T @ u / n)
    whitening = eigenvectors / np.sqrt(eigenvalues)[None, :]
    z = u @ whitening
    design = np.column_stack([np.ones(n), z])
    prior_logits = logit(np.clip(predictions[:, 3], 1e-9, 1 - 1e-9))
    target = np.r_[count, base.targets(n, count, scores) @ whitening]
    checkpoint_path = ROOT / "output/fitted_labels_six_scores_20191117/initial_checkpoint.npz"
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        soft = archive["probability"].copy()
        initial = archive["initial_labels"].copy()
    assert initial.sum() == count
    eta = np.linalg.lstsq(design, logit(soft) - prior_logits, rcond=None)[0]
    assert np.max(np.abs(expit(prior_logits + design @ eta) - soft)) <= 1e-10
    baseline_eta = eta.copy()
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = {"settings": SETTINGS, "versions": source["versions"], "reported_auc": scores.tolist(),
        "scored_files": source["scored_files"], "original_datasets_sha256": source["original_datasets_sha256"],
        "generic_source_sha256": source["generic_source_sha256"],
        "source_sha256": base.digest(__file__), "verifier_sha256": base.digest(HERE / "verify_labels.py"),
        "specification_sha256": base.digest(HERE / "PROTOCOL.md"),
        "initial_checkpoint": str(checkpoint_path.relative_to(ROOT)), "initial_checkpoint_sha256": base.digest(checkpoint_path),
        "six_score_input_binding": str(input_protocol_path.relative_to(ROOT)), "six_score_input_binding_sha256": base.digest(input_protocol_path),
        "all_final_rows_scored_confirmed_by_user": True, "nearest_five_decimal_ordinary_auc_assumed": True,
        "true_positive_count_unknown": True, "labels_are_fitted_hypotheses_not_observations": True,
        "paused_unbounded_goal_not_resumed": True}
    base.write(OUT / "protocol.json", protocol)
    candidates = [initial]
    records = [{"name": "initial_six_score_feasible", "candidate_column": 0,
                "objective": float(-prior_logits @ initial), "valid": True}]
    fits, eta_values = [], []
    group_values, reps, group_sizes = np.unique(predictions, axis=0, return_index=True, return_counts=True)
    group_u = u[reps]
    group_costs = -logit(np.clip(group_values[:, 3], 1e-9, 1 - 1e-9))
    raw_target = base.targets(n, count, scores)
    width = count * (1 - count / n) * SETTINGS["auc_tolerance"]
    for temperature_index, temperature in enumerate(SETTINGS["temperatures"]):
        def objective(parameters):
            logits = (prior_logits + design @ parameters) / temperature
            probability = expit(logits)
            value = (temperature * np.logaddexp(0., logits).sum() - parameters @ target) / n
            return float(value), (design.T @ probability - target) / n
        def hessian(parameters):
            probability = expit((prior_logits + design @ parameters) / temperature)
            return design.T @ (design * (probability * (1 - probability) / temperature)[:, None]) / n
        fit = minimize(objective, eta, jac=True, hess=hessian, method="trust-exact",
            options={"maxiter": SETTINGS["iterations_per_temperature"], "gtol": 1e-10})
        eta = fit.x.copy()
        logits = (prior_logits + design @ eta) / temperature
        q = expit(logits)
        gamma = whitening @ eta[1:]
        reduced = group_costs - eta[0] - group_u @ gamma
        bound = float(count * eta[0] + raw_target @ gamma - width * np.abs(gamma).sum()
                      + group_sizes @ np.minimum(reduced, 0.))
        implied = .5 + q @ u / (count * (1 - count / n))
        info = {"temperature": temperature, "optimizer_success": bool(fit.success), "message": str(fit.message),
                "expected_count_error": float(abs(q.sum() - count)), "expected_rank_moment_error": float(np.max(np.abs(implied - scores))),
                "valid_integer_objective_lower_bound": bound}
        fits.append(info)
        eta_values.append(eta.copy())
        base.emit("entropy_rank_dual_fit", **info)
        for seed in SETTINGS["seeds"]:
            label, record = repair(q, logits, u, z, whitening, scores, seed + temperature_index * 100)
            record.update(temperature=temperature, name=f"temperature_{temperature}_seed_{seed}")
            if label is not None:
                record["candidate_column"] = len(candidates)
                record["objective"] = float(-prior_logits @ label)
                candidates.append(label)
            records.append(record)
            base.emit("bounded_integer_candidate", **record)
    objectives = np.array([-prior_logits @ label for label in candidates])
    best = int(np.argmin(objectives))
    labels = candidates[best]
    actual = np.array([roc_auc_score(labels, predictions[:, j]) for j in range(6)])
    assert all(f"{v:.5f}" == f"{s:.5f}" for v, s in zip(actual, scores))
    csv = pd.DataFrame({"id": np.arange(n), "log_id": final.log_id.to_numpy(), "label": labels,
                        "estimated_click_probability": soft})
    csv.to_csv(OUT / "fitted_labels_20191117.csv", index=False, float_format="%.10f")
    pd.DataFrame({"prediction_file": source["versions"], "reported_auc": scores, "fitted_auc": actual,
                  "reported_display": [f"{v:.5f}" for v in scores], "fitted_display": [f"{v:.5f}" for v in actual]}).to_csv(
        OUT / "six_auc_fit_check.csv", index=False, float_format="%.12f")
    np.savez_compressed(OUT / "fit_arrays.npz", probabilities=soft, candidate_labels=np.column_stack(candidates),
                        selected_labels=labels, annealing_eta=np.array(eta_values), baseline_eta=baseline_eta)
    constant = float(-np.log1p(-np.clip(predictions[:, 3], 1e-9, 1 - 1e-9)).sum())
    bound = max(item["valid_integer_objective_lower_bound"] for item in fits)
    report = {"status": "complete_single_bounded_six_score_fit", "rows": n, "assumed_clicks_not_observed": count,
        "zero_labels": int(n - count), "one_labels": count, "actual_auc": dict(zip(source["versions"], actual.tolist())),
        "all_six_displayed_auc_match": True, "search_records": records, "annealing_fits": fits,
        "retained_candidates": len(candidates), "selected_candidate": best,
        "candidate_objectives": objectives.tolist(), "selected_prior_nll": float(constant + objectives[best]),
        "initial_prior_nll": float(constant + objectives[0]), "objective_lower_bound": bound,
        "absolute_objective_gap": float(objectives[best] - bound), "global_optimality_proven": False,
        "actual_final_labels_recovered": False, "positive_count_is_an_assumption": True,
        "new_supervised_models_trained": 0, "previous_files_unchanged": True,
        "soft_probability_is_temperature_one_conditional_estimate_not_hard_label_confidence": True,
        "seconds": time.perf_counter() - started,
        "files_sha256": {name: base.digest(OUT / name) for name in ["fitted_labels_20191117.csv", "six_auc_fit_check.csv", "fit_arrays.npz"]}}
    base.write(OUT / "report.json", report)
    base.emit("single_fitted_labels_exported", rows=n, ones=count, matched_six_auc=actual.tolist(),
        candidates=len(candidates), selected=best, initial_nll=report["initial_prior_nll"],
        selected_nll=report["selected_prior_nll"], objective_gap=report["absolute_objective_gap"])


if __name__ == "__main__":
    run()
