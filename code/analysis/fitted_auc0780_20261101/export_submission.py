"""Use all feedback for diagnostics; export a global model blend targeting .78."""
import os
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "3"
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/feedback_blend_auc0780_20191117"
SETTINGS = {"target_original_fitted_label_auc": .78, "auc_tolerance": 1e-5,
    "assumed_click_count": 3459, "feedback_projection_steps": 120,
    "grid_points": 21, "bisection_steps": 40, "output_decimals": 10,
    "method": "(1-weight)*actual_scored_old_best + weight*actual_scored_v9",
    "no_per_row_fitted_label_score_shift": True, "real_platform_auc_unknown": True}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def project_feedback(prior, u, scores, count):
    n = len(prior)
    _, singular, vh = np.linalg.svd(u, full_matrices=False)
    whitening = vh.T * (np.sqrt(n) / singular)[None, :]
    design = np.column_stack([np.ones(n), u @ whitening])
    target = np.r_[count, count * (1 - count / n) * (scores - .5) @ whitening]
    original = logit(np.clip(prior, 1e-9, 1 - 1e-9))
    def objective(theta):
        eta = original + design @ theta
        probability = expit(eta)
        return float((np.logaddexp(0., eta).sum() - theta @ target) / n), (design.T @ probability - target) / n
    def hessian(theta):
        probability = expit(original + design @ theta)
        return design.T @ (design * (probability * (1 - probability))[:, None]) / n
    initial = np.zeros(design.shape[1])
    initial[0] = logit(count / n) - logit(float(np.mean(prior)))
    fit = minimize(objective, initial, jac=True, hess=hessian, method="trust-exact",
        options={"gtol": 1e-11, "maxiter": SETTINGS["feedback_projection_steps"]})
    probability = expit(original + design @ fit.x)
    implied = .5 + probability @ u / (count * (1 - count / n))
    info = {"optimizer_success": bool(fit.success), "optimizer_message": str(fit.message),
        "expected_count_error": float(abs(probability.sum() - count)),
        "expected_rank_moments_NOT_binary_auc": implied.tolist(),
        "maximum_expected_rank_moment_error": float(np.max(np.abs(implied - scores))),
        "covariance_condition_number": float((singular.max() / singular.min()) ** 2),
        "whitening_method": "SVD of original rank matrix", "theta": fit.x.tolist()}
    accepted = info["expected_count_error"] <= .001 and info["maximum_expected_rank_moment_error"] <= 1e-7
    info["numerical_moment_certificate_passed"] = bool(accepted)
    return probability, whitening, info


def run():
    if OUT.exists():
        raise FileExistsError("Preserving earlier artifacts")
    old_protocol_path = ROOT / "output/fitted_labels_six_scores_final_20191117/protocol.json"
    old = json.loads(old_protocol_path.read_text(encoding="utf-8"))
    versions = old["versions"] + ["fitted0810", "fitted0812"]
    scores = np.r_[old["reported_auc"], .80855, .80855]
    files = dict(old["scored_files"])
    files.update(fitted0810=("output/fitted_auc0810_20191117/submission_fitted_auc0810.csv", "7e239cb191876455e9038ad8180a5aec9db81e07e8d17862bfef8b1180d811fe"),
                 fitted0812=("output/fitted_auc0812_20191117/submission_fitted_auc0812.csv", "4f52eb000e8ab5868a865bab8a3f7e569191f11266098937acebe486f4999d72"))
    label_path = ROOT / "output/fitted_labels_six_scores_final_20191117/fitted_labels_20191117.csv"
    label_hash = "001b8b5457b1ae3cdb28437f8df501f0dafb0a3f7b33c136ec71efb9f5b7d7b6"
    assert digest(label_path) == label_hash
    final_path = ROOT / "2026ml/final_test.csv"
    final_hash = "9478dfd16937919a30b4efe7b19b805c70196ca5d0c2c8561e37bdb0fdd42b59"
    assert digest(final_path) == final_hash
    final = pd.read_csv(final_path, usecols=["log_id", "pt_d"])
    fitted = pd.read_csv(label_path, float_precision="round_trip")
    n, count = len(final), SETTINGS["assumed_click_count"]
    assert n == 404182 and final.pt_d.eq(20191117).all()
    np.testing.assert_array_equal(fitted.id.to_numpy(), np.arange(n))
    np.testing.assert_array_equal(fitted.log_id.to_numpy(), final.log_id.to_numpy())
    y = fitted.label.to_numpy(dtype="int8")
    assert np.isin(y, [0, 1]).all() and int(y.sum()) == count
    predictions = []
    for version in versions:
        filename, expected = files[version]
        path = Path(filename) if Path(filename).is_absolute() else ROOT / filename
        assert digest(path) == expected
        frame = pd.read_csv(path, float_precision="round_trip")
        np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(n))
        predictions.append(frame.pctr.to_numpy(dtype="float64"))
    predictions = np.column_stack(predictions)
    assert np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all()
    feedback_hashes = {}
    for name in ["0810", "0812"]:
        path = ROOT / f"output/fitted_auc{name}_20191117/platform_feedback.json"
        feedback = json.loads(path.read_text(encoding="utf-8"))
        assert feedback["user_reported_platform_auc"] == .80855
        feedback_hashes[str(path.relative_to(ROOT))] = digest(path)
    u = (np.column_stack([rankdata(predictions[:, j], method="average") for j in range(8)]) - (n + 1) / 2) / n
    q, whitening, projection = project_feedback(predictions[:, versions.index("v9")], u, scores, count)
    print(json.dumps({"event": "eight_feedback_projection", **projection}), flush=True)
    OUT.mkdir(parents=True)
    protocol = {"settings": SETTINGS, "versions": versions, "reported_platform_auc": scores.tolist(),
        "scored_sources": files, "feedback_files_sha256": feedback_hashes,
        "fitted_label_file": str(label_path.relative_to(ROOT)), "fitted_label_sha256": label_hash,
        "final_file": str(final_path.relative_to(ROOT)), "final_file_sha256": final_hash,
        "six_score_protocol": str(old_protocol_path.relative_to(ROOT)), "six_score_protocol_sha256": digest(old_protocol_path),
        "source_sha256": digest(__file__), "verifier_sha256": digest(HERE / "verify_submission.py"),
        "specification_sha256": digest(HERE / "PROTOCOL.md"),
        "global_roc_auc_and_all_rows_and_five_decimal_rounding_assumed": True,
        "new_submission_is_a_requested_diagnostic_not_a_real_improvement_claim": True}
    write(OUT / "protocol.json", protocol)
    np.savez_compressed(OUT / "feedback_projection.npz", probability=q, whitening=whitening)
    weak = predictions[:, versions.index("old_best")]
    strong = predictions[:, versions.index("v9")]
    denominator = count * (1 - count / n)
    target = SETTINGS["target_original_fitted_label_auc"]
    def evaluate(weight):
        values = (1 - weight) * weak + weight * strong
        centered = (rankdata(values, method="average") - (n + 1) / 2) / n
        local = float(.5 + y @ centered / denominator)
        proxy = float(.5 + q @ centered / denominator) if projection["numerical_moment_certificate_passed"] else None
        return {"strong_model_weight": float(weight), "original_fitted_label_auc": local,
                "eight_feedback_soft_rank_moment_proxy_NOT_real_auc": proxy}
    grid = [evaluate(float(value)) for value in np.linspace(0, 1, SETTINGS["grid_points"])]
    candidates, searches = [], []
    for left, right in zip(grid[:-1], grid[1:]):
        if (left["original_fitted_label_auc"] - target) * (right["original_fitted_label_auc"] - target) > 0:
            continue
        low, high = left["strong_model_weight"], right["strong_model_weight"]
        low_value = left["original_fitted_label_auc"] - target
        for step in range(SETTINGS["bisection_steps"]):
            middle = (low + high) / 2
            result = evaluate(middle)
            searches.append(result)
            value = result["original_fitted_label_auc"] - target
            if low_value * value > 0:
                low, low_value = middle, value
            else:
                high = middle
        candidates.extend([evaluate(low), evaluate(high), evaluate((low + high) / 2)])
    eligible = [item for item in candidates if abs(item["original_fitted_label_auc"] - target) <= SETTINGS["auc_tolerance"]]
    if not eligible:
        raise RuntimeError("No blend reached the requested local diagnostic target")
    selected = min(eligible, key=lambda item: (abs(item["eight_feedback_soft_rank_moment_proxy_NOT_real_auc"] - target)
        if item["eight_feedback_soft_rank_moment_proxy_NOT_real_auc"] is not None else 0., abs(item["original_fitted_label_auc"] - target)))
    weight = selected["strong_model_weight"]
    prediction = (1 - weight) * weak + weight * strong
    np.save(OUT / "prediction.npy", prediction)
    pd.DataFrame({"id": np.arange(n), "pctr": prediction}).to_csv(OUT / "submission_fitted_auc0780.csv", index=False, float_format="%.10f")
    csv = pd.read_csv(OUT / "submission_fitted_auc0780.csv", float_precision="round_trip")
    actual_local = float(roc_auc_score(y, csv.pctr))
    assert abs(actual_local - target) <= SETTINGS["auc_tolerance"]
    original_hypothesis_metrics = [float(roc_auc_score(y, predictions[:, j])) for j in range(8)]
    report = {"status": "exported_pending_verification", "rows": n,
        "target_original_fitted_label_auc": target, "exported_csv_auc_against_original_fitted_labels": actual_local,
        "strong_model_weight": weight, "weak_model_weight": 1 - weight,
        "selected_candidate": selected, "feedback_projection": projection,
        "assumed_positive_count_not_observed": count,
        "old_fitted_hypothesis_auc_on_eight_scored_files": dict(zip(versions, original_hypothesis_metrics)),
        "expected_old_group_clicks_under_conditional_model_NOT_observed": float(q[y == 1].sum()),
        "all_eight_feedback_values_used_in_conditional_analysis": True,
        "source_model_platform_endpoint_auc": {"old_best": .74404, "v9": .80856},
        "no_individual_guessed_label_probability_boost_or_penalty": True,
        "selection_still_uses_original_synthetic_labels": True,
        "soft_rank_moment_proxy_is_not_a_real_platform_forecast": True,
        "real_platform_auc": None, "platform078_verified": False,
        "requested_target_is_below_best_reported_platform_auc": True,
        "actual_platform_best_v9_unchanged": True, "new_models_trained": 0,
        "input_files_preserved": True, "grid": grid, "search_evaluations": searches,
        "files_sha256": {name: digest(OUT / name) for name in ["feedback_projection.npz", "prediction.npy", "submission_fitted_auc0780.csv"]}}
    write(OUT / "report.json", report)
    print(json.dumps({"event": "requested078_exported", "rows": n, "strong_model_weight": weight,
        "csv_auc_against_original_fitted_labels": actual_local, "conditional_rank_proxy_not_real_auc": selected["eight_feedback_soft_rank_moment_proxy_NOT_real_auc"],
        "projection_certificate_passed": projection["numerical_moment_certificate_passed"], "real_platform_auc": None}), flush=True)


if __name__ == "__main__":
    run()
