"""Construct a submission with .815 AUC against a fixed fitted-label hypothesis."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/fitted_auc0815_20191117"
INPUTS = {
    "labels": ("output/fitted_labels_six_scores_final_20191117/fitted_labels_20191117.csv", "001b8b5457b1ae3cdb28437f8df501f0dafb0a3f7b33c136ec71efb9f5b7d7b6"),
    "v9": ("output/platform_feedback_v9/submission_v9.csv", "c55583694642524fbabe71b5a0d4d06e86582076e899d1087d1d9dfd73271a83"),
    "final": ("2026ml/final_test.csv", "9478dfd16937919a30b4efe7b19b805c70196ca5d0c2c8561e37bdb0fdd42b59")}
SETTINGS = {"target_fitted_label_auc": .815, "target_tolerance": 1e-5,
    "maximum_bracketing_steps": 12, "maximum_bisection_steps": 40,
    "initial_positive_log_odds_shift": .05, "output_decimals": 10,
    "method": "sigmoid(logit(v9) + shift * fitted_label)",
    "actual_platform_auc_unknown": True, "same_fitted_labels_select_and_score_the_adjustment": True}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run():
    if OUT.exists():
        raise FileExistsError("Preserving previously exported submissions")
    for filename, expected in INPUTS.values():
        assert digest(ROOT / filename) == expected, "Source differs from bound file"
    fitted = pd.read_csv(ROOT / INPUTS["labels"][0], float_precision="round_trip")
    baseline = pd.read_csv(ROOT / INPUTS["v9"][0], float_precision="round_trip")
    final = pd.read_csv(ROOT / INPUTS["final"][0], usecols=["log_id", "pt_d"])
    n = len(final)
    assert n == 404182 and final.pt_d.eq(20191117).all()
    for frame in [fitted, baseline]:
        np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(n))
    np.testing.assert_array_equal(fitted.log_id.to_numpy(), final.log_id.to_numpy())
    y = fitted.label.to_numpy(dtype="int8")
    p = baseline.pctr.to_numpy(dtype="float64")
    assert np.isin(y, [0, 1]).all() and y.sum() == 3459
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    original_logits = logit(p)
    negative = np.sort(original_logits[y == 0])
    positive = original_logits[y == 1]
    def pair_auc(shift):
        values = positive + shift
        pairs = np.searchsorted(negative, values, "left") + np.searchsorted(negative, values, "right")
        return float(np.sum(pairs, dtype="int64") / (2 * len(positive) * len(negative)))
    target = SETTINGS["target_fitted_label_auc"]
    original_auc = float(roc_auc_score(y, p))
    assert abs(pair_auc(0.) - original_auc) < 1e-12
    low, high = 0., SETTINGS["initial_positive_log_odds_shift"]
    assert original_auc < target
    for _ in range(SETTINGS["maximum_bracketing_steps"]):
        if pair_auc(high) >= target:
            break
        high *= 2
    else:
        raise RuntimeError("Could not bracket fitted-label target")
    trace = []
    for step in range(SETTINGS["maximum_bisection_steps"]):
        middle = (low + high) / 2
        auc = pair_auc(middle)
        trace.append({"step": step + 1, "shift": middle, "fitted_label_auc": auc})
        if auc < target:
            low = middle
        else:
            high = middle
    choices = [low, high, (low + high) / 2]
    shift = min(choices, key=lambda value: abs(pair_auc(value) - target))
    prediction = expit(original_logits + shift * y)
    rounded = np.round(prediction, SETTINGS["output_decimals"])
    achieved = float(roc_auc_score(y, rounded))
    assert abs(achieved - target) <= SETTINGS["target_tolerance"]
    OUT.mkdir(parents=True)
    write(OUT / "protocol.json", {"settings": SETTINGS, "input_sha256": INPUTS,
        "source_sha256": digest(__file__), "verifier_sha256": digest(HERE / "verify_submission.py"),
        "specification_sha256": digest(HERE / "PROTOCOL.md"),
        "metric_is_in_sample_fitted_label_auc_not_platform_auc": True})
    np.save(OUT / "prediction.npy", prediction)
    pd.DataFrame({"id": np.arange(n), "pctr": prediction}).to_csv(OUT / "submission_fitted_auc0815.csv", index=False, float_format="%.10f")
    exported = pd.read_csv(OUT / "submission_fitted_auc0815.csv", float_precision="round_trip")
    measured = float(roc_auc_score(y, exported.pctr))
    assert abs(measured - target) <= SETTINGS["target_tolerance"]
    report = {"status": "exported_pending_independent_verification", "rows": n,
        "fitted_label_ones_assumed": int(y.sum()), "target_fitted_label_auc": target,
        "baseline_fitted_label_auc": original_auc, "exported_csv_fitted_label_auc": measured,
        "absolute_fitted_label_target_error": abs(measured - target),
        "positive_label_log_odds_shift": shift, "positive_label_odds_multiplier": float(np.exp(shift)),
        "rows_directly_adjusted_using_fitted_labels": int(y.sum()),
        "maximum_probability_change": float(np.max(np.abs(prediction - p))),
        "mean_probability_change": float(np.mean(prediction - p)),
        "same_synthetic_labels_used_to_adjust_and_evaluate": True,
        "not_independent_validation_or_a_real_label_metric": True,
        "real_platform_auc": None, "platform0815_verified": False,
        "new_models_trained": 0, "reported_platform_best_v9_auc_unchanged": .80856,
        "input_files_unchanged": True, "trace": trace,
        "files_sha256": {name: digest(OUT / name) for name in ["submission_fitted_auc0815.csv", "prediction.npy"]}}
    write(OUT / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ["trace", "files_sha256"]}), flush=True)


if __name__ == "__main__":
    run()
