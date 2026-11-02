"""Independently verify the delivered six-score assignment and objective bound."""
import os
for thread_variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[thread_variable] = "3"
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
OUT = ROOT / "output/fitted_labels_six_scores_20191117"


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verify():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    for filename, field in [("fit_labels.py", "source_sha256"), ("verify_labels.py", "verifier_sha256"), ("PROTOCOL.md", "specification_sha256")]:
        assert digest(HERE / filename) == protocol[field], "Fitting source changed"
    assert digest(ROOT / "analysis/auc_label_inference_20261101/infer_labels.py") == protocol["generic_source_sha256"]
    for filename, expected in protocol["original_datasets_sha256"].items():
        assert digest(ROOT / "2026ml" / filename) == expected, "Raw dataset changed"
    for filename, expected in protocol["previous_five_score_output_sha256"].items():
        assert digest(ROOT / "output/label_hypotheses_five_scores_20191117" / filename) == expected, "Previous result changed"
    for filename, expected in report["files_sha256"].items():
        assert digest(OUT / filename) == expected, "Export hash differs"
    assert digest(OUT / "scored_submission_3.csv") == protocol["sixth_attachment_sha256"]
    final = pd.read_csv(ROOT / "2026ml/final_test.csv", usecols=["log_id", "pt_d"])
    n = len(final)
    assert n == report["rows"] == 404182 and final.pt_d.eq(20191117).all()
    predictions = []
    for name in protocol["versions"]:
        path, expected = protocol["scored_files"][name]
        path = Path(path) if Path(path).is_absolute() else ROOT / path
        assert digest(path) == expected, "Scored input changed"
        source = pd.read_csv(path, float_precision="round_trip")
        np.testing.assert_array_equal(source.id.to_numpy(), np.arange(n))
        predictions.append(source.pctr.to_numpy(dtype="float64"))
    predictions = np.column_stack(predictions)
    assert np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all()
    with np.load(OUT / "fit_arrays.npz", allow_pickle=False) as arrays:
        q, y, groups, counts, candidates, lp_x, dual_inequality, dual_equality = [arrays[key].copy() for key in
            ["probability", "labels", "groups", "group_counts", "candidate_group_counts", "lp_x", "dual_inequality", "dual_equality"]]
    delivered = pd.read_csv(OUT / "fitted_labels_20191117.csv", float_precision="round_trip")
    np.testing.assert_array_equal(delivered.id.to_numpy(), np.arange(n))
    np.testing.assert_array_equal(delivered.log_id.to_numpy(), final.log_id.to_numpy())
    np.testing.assert_array_equal(delivered.label.to_numpy(), y)
    csv_probability_error = float(np.max(np.abs(delivered.estimated_click_probability.to_numpy() - q)))
    assert csv_probability_error <= 5.0001e-11 and np.isfinite(q).all() and ((q > 0) & (q < 1)).all()
    scores = np.array(protocol["reported_auc"])
    p = protocol["settings"]["assumed_positive_count"]
    assert np.isin(y, [0, 1]).all() and int(y.sum()) == p == 3459
    actual = np.array([roc_auc_score(y, predictions[:, j]) for j in range(6)])
    assert all(f"{v:.5f}" == f"{t:.5f}" for v, t in zip(actual, scores))
    np.testing.assert_allclose(actual, [report["actual_roc_auc"][name] for name in protocol["versions"]], atol=1e-12, rtol=0)
    u = (np.column_stack([rankdata(predictions[:, j], method="average") for j in range(6)]) - (n + 1) / 2) / n
    unique, representatives, rebuilt_groups, sizes = np.unique(predictions, axis=0, return_index=True, return_inverse=True, return_counts=True)
    np.testing.assert_array_equal(groups, rebuilt_groups)
    np.testing.assert_array_equal(np.bincount(groups, weights=y, minlength=len(sizes)), counts)
    group_u = u[representatives]
    costs = -logit(np.clip(unique[:, 3], 1e-9, 1 - 1e-9))
    prior = np.clip(predictions[:, 3], 1e-9, 1 - 1e-9)
    constant = float(-np.log1p(-prior).sum())
    candidate_objectives = []
    target = p * (1 - p / n) * (scores - .5)
    width = p * (1 - p / n) * protocol["settings"]["auc_tolerance"]
    ordered_rows = np.argsort(groups, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(groups[ordered_rows])) + 1]
    group_positions = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    for j in range(candidates.shape[1]):
        candidate = candidates[:, j]
        assert np.issubdtype(candidate.dtype, np.integer) and (candidate >= 0).all() and (candidate <= sizes).all()
        assert int(candidate.sum()) == p
        implied = .5 + candidate @ group_u / (p * (1 - p / n))
        assert all(f"{v:.5f}" == f"{t:.5f}" for v, t in zip(implied, scores))
        assert np.max(np.abs(implied - scores)) <= protocol["settings"]["auc_tolerance"] + 1e-10
        candidate_labels = np.zeros(n, dtype="int8")
        candidate_labels[ordered_rows] = group_positions < candidate[groups[ordered_rows]]
        candidate_auc = [roc_auc_score(candidate_labels, predictions[:, k]) for k in range(6)]
        np.testing.assert_allclose(candidate_auc, implied, atol=1e-12, rtol=0.)
        candidate_objectives.append(float(costs @ candidate))
    selected = report["selected_candidate_column"]
    assert selected == int(np.argmin(candidate_objectives))
    np.testing.assert_array_equal(candidates[:, selected], counts)
    assert abs(candidate_objectives[selected] - report["selected_objective"]) < 1e-8
    assert abs(constant + candidate_objectives[selected] - report["selected_prior_nll"]) < 1e-8
    row_nll = -float((y * np.log(prior) + (1 - y) * np.log1p(-prior)).sum())
    assert abs(row_nll - report["selected_prior_nll"]) < 1e-8
    # Check the arbitrary but deterministic order used inside identical groups.
    order = np.argsort(groups, kind="stable")
    labels_sorted = y[order]
    same_group = np.diff(groups[order]) == 0
    assert not ((np.diff(labels_sorted) > 0) & same_group).any()
    bound = None
    if len(lp_x):
        assert lp_x.shape == costs.shape and np.isfinite(lp_x).all()
        assert (lp_x >= -1e-7).all() and (lp_x <= sizes + 1e-7).all()
        assert abs(lp_x.sum() - p) < 1e-5 and np.max(np.abs(lp_x @ group_u - target)) <= width + 1e-5
        assert abs(float(costs @ lp_x) - report["lp_relaxation"]["objective"]) < 1e-8
        assert (dual_inequality <= 0).all() and dual_inequality.shape == (12,) and dual_equality.shape == (1,)
        matrix = np.vstack([group_u.T, -group_u.T])
        right = np.r_[target + width, -target + width]
        reduced = costs - matrix.T @ dual_inequality - float(dual_equality[0])
        bound = float(right @ dual_inequality + p * dual_equality[0] + sizes @ np.minimum(reduced, 0.))
        assert abs(bound - report["certified_objective_lower_bound"]) < 1e-8
        assert candidate_objectives[selected] >= bound - 1e-6
        assert abs((candidate_objectives[selected] - bound) - report["absolute_gap_to_lower_bound"]) < 1e-8
    values, vectors = np.linalg.eigh(u.T @ u / n)
    whitening = vectors / np.sqrt(values)[None, :]
    design = np.column_stack([np.ones(n), u @ whitening])
    probability_replay = expit(logit(prior) + design @ np.array(report["soft_probability_projection"]["theta"]))
    replay_error = float(np.max(np.abs(probability_replay - q)))
    assert replay_error <= 1e-12 and abs(q.sum() - p) < .001
    implied_soft = .5 + q @ u / (p * (1 - p / n))
    assert np.max(np.abs(implied_soft - scores)) <= 1e-7
    metrics = pd.read_csv(OUT / "six_auc_fit_check.csv", dtype={"reported_five_decimals": str, "fitted_five_decimals": str})
    assert metrics.prediction_file.tolist() == protocol["versions"]
    assert metrics.reported_five_decimals.tolist() == [f"{v:.5f}" for v in scores]
    assert metrics.fitted_five_decimals.tolist() == [f"{v:.5f}" for v in actual]
    result = {"status": "passed", "rows": n, "ones_assumed_not_observed": p, "zeros": n - p,
        "actual_roc_auc": dict(zip(protocol["versions"], actual.tolist())),
        "all_six_displayed_scores_exactly_reproduced": True,
        "maximum_csv_probability_error": csv_probability_error, "probability_replay_error": replay_error,
        "valid_integer_candidates_verified": len(candidate_objectives), "best_searched_candidate_selected": True,
        "objective_lower_bound": bound, "objective_gap": report["absolute_gap_to_lower_bound"],
        "global_optimality_proven": report["global_integer_optimality_proven"],
        "raw_datasets_scored_files_and_previous_analyses_unchanged": True,
        "actual_final_labels_recovered": False, "positive_count_is_an_assumption": True,
        "csv_sha256": digest(OUT / "fitted_labels_20191117.csv")}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    verify()
