"""Independent actual AUC, file integrity, selection and dual-bound checks."""
import os
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "3"
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/fitted_labels_six_scores_final_20191117"


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verify():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    for filename, key in [("fit_labels.py", "source_sha256"), ("verify_labels.py", "verifier_sha256"), ("PROTOCOL.md", "specification_sha256")]:
        assert digest(HERE / filename) == protocol[key]
    assert digest(ROOT / "analysis/auc_label_inference_20261101/infer_labels.py") == protocol["generic_source_sha256"]
    assert digest(ROOT / protocol["initial_checkpoint"]) == protocol["initial_checkpoint_sha256"]
    assert digest(ROOT / protocol["six_score_input_binding"]) == protocol["six_score_input_binding_sha256"]
    for filename, expected in protocol["original_datasets_sha256"].items():
        assert digest(ROOT / "2026ml" / filename) == expected
    for filename, expected in report["files_sha256"].items():
        assert digest(OUT / filename) == expected
    final = pd.read_csv(ROOT / "2026ml/final_test.csv", usecols=["log_id", "pt_d"])
    n, count = len(final), protocol["settings"]["assumed_click_count"]
    assert n == 404182 and count == 3459 and final.pt_d.eq(20191117).all()
    predictions = []
    for version in protocol["versions"]:
        filename, expected = protocol["scored_files"][version]
        path = Path(filename) if Path(filename).is_absolute() else ROOT / filename
        assert digest(path) == expected
        frame = pd.read_csv(path, float_precision="round_trip")
        np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(n))
        predictions.append(frame.pctr.to_numpy())
    predictions = np.column_stack(predictions)
    assert np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all()
    with np.load(OUT / "fit_arrays.npz", allow_pickle=False) as archive:
        probability, labels, selected, etas, baseline_eta = [archive[key].copy() for key in
            ["probabilities", "candidate_labels", "selected_labels", "annealing_eta", "baseline_eta"]]
    delivered = pd.read_csv(OUT / "fitted_labels_20191117.csv", float_precision="round_trip")
    np.testing.assert_array_equal(delivered.id.to_numpy(), np.arange(n))
    np.testing.assert_array_equal(delivered.log_id.to_numpy(), final.log_id.to_numpy())
    np.testing.assert_array_equal(delivered.label.to_numpy(), selected)
    csv_error = float(np.max(np.abs(delivered.estimated_click_probability.to_numpy() - probability)))
    assert csv_error <= 5.0001e-11 and np.isfinite(probability).all() and ((probability > 0) & (probability < 1)).all()
    scores = np.array(protocol["reported_auc"])
    prior = np.clip(predictions[:, 3], 1e-9, 1 - 1e-9)
    objectives, checked_auc = [], []
    for j in range(labels.shape[1]):
        y = labels[:, j]
        assert np.isin(y, [0, 1]).all() and int(y.sum()) == count
        actual = np.array([roc_auc_score(y, predictions[:, k]) for k in range(6)])
        assert all(f"{v:.5f}" == f"{s:.5f}" for v, s in zip(actual, scores))
        objectives.append(float(-logit(prior) @ y))
        checked_auc.append(actual.tolist())
    best = int(np.argmin(objectives))
    assert best == report["selected_candidate"] and labels.shape[1] == report["retained_candidates"]
    np.testing.assert_array_equal(selected, labels[:, best])
    np.testing.assert_allclose(objectives, report["candidate_objectives"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(checked_auc[best], [report["actual_auc"][v] for v in protocol["versions"]], atol=1e-12, rtol=0)
    nll = -float((selected * np.log(prior) + (1 - selected) * np.log1p(-prior)).sum())
    assert abs(nll - report["selected_prior_nll"]) < 1e-8
    u = (np.column_stack([rankdata(predictions[:, k], method="average") for k in range(6)]) - (n + 1) / 2) / n
    values, vectors = np.linalg.eigh(u.T @ u / n)
    whitening = vectors / np.sqrt(values)[None, :]
    design = np.column_stack([np.ones(n), u @ whitening])
    replay_error = float(np.max(np.abs(expit(logit(prior) + design @ baseline_eta) - probability)))
    assert replay_error < 1e-10 and abs(probability.sum() - count) < .001
    implied_soft = .5 + probability @ u / (count * (1 - count / n))
    assert np.max(np.abs(implied_soft - scores)) < 1e-7
    unique, reps, sizes = np.unique(predictions, axis=0, return_index=True, return_counts=True)
    costs = -logit(np.clip(unique[:, 3], 1e-9, 1 - 1e-9))
    target = count * (1 - count / n) * (scores - .5)
    width = count * (1 - count / n) * protocol["settings"]["auc_tolerance"]
    bounds = []
    for eta, info in zip(etas, report["annealing_fits"]):
        gamma = whitening @ eta[1:]
        residual = costs - eta[0] - u[reps] @ gamma
        bound = float(count * eta[0] + target @ gamma - width * np.abs(gamma).sum() + sizes @ np.minimum(residual, 0.))
        assert abs(bound - info["valid_integer_objective_lower_bound"]) < 1e-8
        bounds.append(bound)
    bound = max(bounds)
    assert min(objectives) >= bound - 1e-6
    assert abs(min(objectives) - bound - report["absolute_objective_gap"]) < 1e-8
    result = {"status": "passed", "rows": n, "assumed_ones": count, "zeros": n - count,
        "all_six_auc_matched_to_five_decimals": True, "selected_actual_auc": dict(zip(protocol["versions"], checked_auc[best])),
        "integer_candidates_independently_verified": labels.shape[1], "lowest_nll_searched_candidate_selected": True,
        "probability_csv_rounding_error": csv_error, "probability_replay_error": replay_error,
        "valid_objective_lower_bound": bound, "objective_gap": report["absolute_objective_gap"],
        "global_optimality_proven": False, "actual_final_labels_recovered": False,
        "raw_datasets_and_scored_submissions_unchanged": True,
        "fitted_labels_sha256": digest(OUT / "fitted_labels_20191117.csv")}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    verify()
