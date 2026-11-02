"""Independent odds replay and actual ROC AUC on the exported CSV."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/fitted_auc0815_20191117"


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
    for filename, expected in protocol["input_sha256"].values():
        assert digest(ROOT / filename) == expected
    for filename, expected in report["files_sha256"].items():
        assert digest(OUT / filename) == expected
    sources = protocol["input_sha256"]
    labels = pd.read_csv(ROOT / sources["labels"][0], float_precision="round_trip")
    baseline = pd.read_csv(ROOT / sources["v9"][0], float_precision="round_trip")
    final = pd.read_csv(ROOT / sources["final"][0], usecols=["log_id", "pt_d"])
    exported = pd.read_csv(OUT / "submission_fitted_auc0815.csv", float_precision="round_trip")
    assert exported.columns.tolist() == ["id", "pctr"] and len(exported) == len(final) == 404182
    for frame in [labels, baseline, exported]:
        np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(len(final)))
    np.testing.assert_array_equal(labels.log_id.to_numpy(), final.log_id.to_numpy())
    assert final.pt_d.eq(20191117).all()
    y, p = labels.label.to_numpy(), baseline.pctr.to_numpy()
    assert np.isin(y, [0, 1]).all() and y.sum() == 3459
    multiplier = np.where(y == 1, np.exp(report["positive_label_log_odds_shift"]), 1.)
    odds = p / (1 - p) * multiplier
    independent = odds / (1 + odds)
    saved = np.load(OUT / "prediction.npy", allow_pickle=False)
    error = float(np.max(np.abs(independent - saved)))
    rounding_error = float(np.max(np.abs(exported.pctr.to_numpy() - independent)))
    assert error < 1e-14 and rounding_error <= 5.0001e-11
    assert np.isfinite(exported.pctr).all() and exported.pctr.between(0, 1, inclusive="neither").all()
    measured = float(roc_auc_score(y, exported.pctr))
    assert abs(measured - protocol["settings"]["target_fitted_label_auc"]) <= protocol["settings"]["target_tolerance"]
    assert abs(measured - report["exported_csv_fitted_label_auc"]) < 1e-12
    assert np.max(np.abs(independent[y == 0] - p[y == 0])) < 1e-14
    assert (independent[y == 1] > p[y == 1]).all()
    result = {"status": "passed", "rows": len(final), "fitted_label_positives_assumed": int(y.sum()),
        "exported_csv_auc_against_fitted_labels_only": measured,
        "maximum_independent_formula_error": error, "maximum_csv_rounding_error": rounding_error,
        "original_ids_and_final_row_order_preserved": True,
        "duplicate_log_id_rows_preserved": int(final.log_id.duplicated().sum()),
        "source_files_unchanged": True, "same_fitted_labels_select_and_score_the_adjustment": True,
        "real_platform_auc": None, "real_platform0815_verified": False,
        "submission_sha256": digest(OUT / "submission_fitted_auc0815.csv")}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    verify()
