"""Replay the .78 global blend and independently check all eight rank moments."""
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
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/feedback_blend_auc0780_20191117"


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verify():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    for filename, key in [("export_submission.py", "source_sha256"), ("verify_submission.py", "verifier_sha256"), ("PROTOCOL.md", "specification_sha256")]:
        assert digest(HERE / filename) == protocol[key]
    for filename, expected in protocol["feedback_files_sha256"].items():
        assert digest(ROOT / filename) == expected
    for name, expected in report["files_sha256"].items():
        assert digest(OUT / name) == expected
    assert digest(ROOT / protocol["fitted_label_file"]) == protocol["fitted_label_sha256"]
    assert digest(ROOT / protocol["final_file"]) == protocol["final_file_sha256"]
    assert digest(ROOT / protocol["six_score_protocol"]) == protocol["six_score_protocol_sha256"]
    final = pd.read_csv(ROOT / protocol["final_file"], usecols=["log_id", "pt_d"])
    fitted = pd.read_csv(ROOT / protocol["fitted_label_file"], float_precision="round_trip")
    n, count = len(final), protocol["settings"]["assumed_click_count"]
    assert n == 404182 and final.pt_d.eq(20191117).all()
    np.testing.assert_array_equal(fitted.log_id.to_numpy(), final.log_id.to_numpy())
    np.testing.assert_array_equal(fitted.id.to_numpy(), np.arange(n))
    y = fitted.label.to_numpy()
    assert int(y.sum()) == count == 3459 and np.isin(y, [0, 1]).all()
    predictions = []
    for version in protocol["versions"]:
        filename, expected = protocol["scored_sources"][version]
        path = Path(filename) if Path(filename).is_absolute() else ROOT / filename
        assert digest(path) == expected
        frame = pd.read_csv(path, float_precision="round_trip")
        np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(n))
        predictions.append(frame.pctr.to_numpy(dtype="float64"))
    predictions = np.column_stack(predictions)
    weak = predictions[:, protocol["versions"].index("old_best")]
    strong = predictions[:, protocol["versions"].index("v9")]
    weight = report["strong_model_weight"]
    assert 0 < weight < 1 and abs(report["weak_model_weight"] + weight - 1) < 1e-15
    independent = weak + weight * (strong - weak)
    saved = np.load(OUT / "prediction.npy", allow_pickle=False)
    formula_error = float(np.max(np.abs(independent - saved)))
    assert formula_error < 1e-14
    csv = pd.read_csv(OUT / "submission_fitted_auc0780.csv", float_precision="round_trip")
    assert csv.columns.tolist() == ["id", "pctr"] and len(csv) == n
    np.testing.assert_array_equal(csv.id.to_numpy(), np.arange(n))
    assert np.isfinite(csv.pctr).all() and csv.pctr.between(0, 1, inclusive="neither").all()
    rounding_error = float(np.max(np.abs(csv.pctr.to_numpy() - independent)))
    assert rounding_error <= 5.0001e-11
    auc = float(roc_auc_score(y, csv.pctr))
    assert abs(auc - .78) <= protocol["settings"]["auc_tolerance"]
    assert abs(auc - report["exported_csv_auc_against_original_fitted_labels"]) < 1e-12
    u = (np.column_stack([rankdata(predictions[:, j], method="average") for j in range(8)]) - (n + 1) / 2) / n
    with np.load(OUT / "feedback_projection.npz", allow_pickle=False) as archive:
        q, whitening = archive["probability"].copy(), archive["whitening"].copy()
    assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
    design = np.column_stack([np.ones(n), u @ whitening])
    replay = expit(logit(np.clip(strong, 1e-9, 1 - 1e-9)) + design @ np.array(report["feedback_projection"]["theta"]))
    replay_error = float(np.max(np.abs(q - replay)))
    assert replay_error < 1e-12
    expected = .5 + q @ u / (count * (1 - count / n))
    expected_error = float(np.max(np.abs(expected - protocol["reported_platform_auc"])))
    certificate = expected_error <= 1e-7 and abs(q.sum() - count) <= .001
    assert certificate == report["feedback_projection"]["numerical_moment_certificate_passed"]
    centered = (rankdata(saved, method="average") - (n + 1) / 2) / n
    proxy = float(.5 + q @ centered / (count * (1 - count / n))) if certificate else None
    if certificate:
        assert abs(proxy - report["selected_candidate"]["eight_feedback_soft_rank_moment_proxy_NOT_real_auc"]) < 1e-12
    result = {"status": "passed", "rows": n, "csv_auc_against_original_fitted_labels_only": auc,
        "strong_model_weight": weight, "weak_model_weight": 1 - weight,
        "maximum_blend_formula_error": formula_error, "maximum_csv_rounding_error": rounding_error,
        "eight_feedback_expected_rank_moment_certificate_passed": bool(certificate),
        "maximum_expected_rank_moment_error": expected_error, "probability_projection_replay_error": replay_error,
        "soft_rank_moment_proxy_NOT_real_platform_auc": proxy,
        "original_row_order_and_ids_preserved": True,
        "duplicate_log_id_rows_preserved": int(final.log_id.duplicated().sum()),
        "original_inputs_and_previous_submissions_unchanged": True,
        "real_platform_auc": None, "real_platform078_verified": False,
        "submission_sha256": digest(OUT / "submission_fitted_auc0780.csv")}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    verify()
