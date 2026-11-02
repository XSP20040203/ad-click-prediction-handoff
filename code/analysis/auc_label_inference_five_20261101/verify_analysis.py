"""Independent file reload, sklearn AUC and pinned-label witness verification."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit, gammaln
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/label_hypotheses_five_scores_20191117"
OLD_OUT = ROOT / "output/label_hypotheses_20191117"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def verify():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    for filename, field in [("run_analysis.py", "source_sha256"), ("verify_analysis.py", "verifier_sha256"), ("PROTOCOL.md", "specification_sha256")]:
        assert digest(HERE / filename) == protocol[field], "Inference sources changed"
    assert digest(ROOT / "analysis/auc_label_inference_20261101/infer_labels.py") == protocol["generic_source_sha256"]
    for name, expected in protocol["original_datasets_sha256"].items():
        assert digest(ROOT / "2026ml" / name) == expected, "Raw dataset changed"
    for name, expected in protocol["old_output_files_sha256"].items():
        assert digest(OLD_OUT / name) == expected, "Old result changed"
    for name, expected in report["files_sha256"].items():
        assert digest(OUT / name) == expected, "New artifact hash differs"
    predictions = []
    versions = protocol["auc_versions"]
    expected_scores = protocol["reported_auc"]
    final = pd.read_csv(ROOT / "2026ml/final_test.csv", usecols=["log_id", "pt_d"])
    n = len(final)
    assert n == 404182 and final.pt_d.eq(20191117).all()
    for name in versions:
        relative, expected = protocol["scored_files_sha256"][name]
        assert digest(ROOT / relative) == expected, "Scored source changed"
        frame = pd.read_csv(ROOT / relative, float_precision="round_trip")
        np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(n))
        predictions.append(frame.pctr.to_numpy(dtype="float64"))
    predictions = np.column_stack(predictions)
    assert np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all()
    assert digest(OUT / "scored_submission_best.csv") == protocol["scored_files_sha256"]["old_best"][1]
    with np.load(OUT / "inference_arrays.npz", allow_pickle=False) as archive:
        q, y, groups, opposite, queried = [archive[key].copy() for key in ["probabilities", "hypotheses", "groups", "opposite_hypotheses", "queried_rows"]]
    frame = pd.read_csv(OUT / "conditional_estimates_20191117.csv", float_precision="round_trip")
    np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(n))
    np.testing.assert_array_equal(frame.log_id.to_numpy(), final.log_id.to_numpy())
    np.testing.assert_array_equal(frame[["label_hypothesis_A", "label_hypothesis_B", "label_hypothesis_C"]].to_numpy(), y)
    error = float(np.max(np.abs(frame[["click_probability_low_count", "click_probability_central", "click_probability_high_count"]].to_numpy() - q)))
    assert error <= 5.0001e-11 and np.isfinite(q).all() and ((q > 0) & (q < 1)).all()
    u = (np.column_stack([rankdata(predictions[:, j], method="average") for j in range(5)]) - (n + 1) / 2) / n
    values, vectors = np.linalg.eigh(u.T @ u / n)
    whitening = vectors / np.sqrt(values)[None, :]
    design = np.column_stack([np.ones(n), u @ whitening])
    original = logit(np.clip(predictions[:, 3], 1e-9, 1 - 1e-9))
    reload_errors = []
    for j, record in enumerate(report["conditional_projections"]):
        replay = expit(original + design @ np.array(record["theta"]))
        replay_error = float(np.max(np.abs(replay - q[:, j])))
        assert replay_error <= 1e-12
        count = record["assumed_positive_count"]
        implied = .5 + q[:, j] @ u / (count * (1 - count / n))
        assert np.max(np.abs(implied - expected_scores)) <= 1e-7 and abs(q[:, j].sum() - count) <= .001
        reload_errors.append(replay_error)
    count = report["assumed_click_count_not_observed"]
    def check_labels(labels):
        assert np.isin(labels, [0, 1]).all() and int(labels.sum()) == count
        scores = [float(roc_auc_score(labels, predictions[:, j])) for j in range(5)]
        assert all(f"{value:.5f}" == f"{target:.5f}" for value, target in zip(scores, expected_scores)), "An actual ROC AUC differs"
        return scores
    actual = {}
    for j, name in enumerate(["A", "B", "C"]):
        actual[name] = check_labels(y[:, j])
        np.testing.assert_allclose(actual[name], [report["actual_auc_by_hypothesis"][name][version] for version in versions], rtol=0., atol=1e-12)
    _, rebuilt, sizes = np.unique(predictions, axis=0, return_inverse=True, return_counts=True)
    np.testing.assert_array_equal(groups, rebuilt)
    np.testing.assert_array_equal(frame.equal_five_score_group_size.to_numpy(), sizes[groups])
    counts_a = np.bincount(groups, weights=y[:, 0], minlength=len(sizes))
    np.testing.assert_array_equal(counts_a, np.bincount(groups, weights=y[:, 1], minlength=len(sizes)))
    np.testing.assert_allclose(actual["A"], actual["B"], atol=1e-12, rtol=0)
    same_prior_difference = float(np.dot(y[:, 0] - y[:, 1], logit(q[:, 1])))
    assert abs(same_prior_difference) < 1e-8
    bits = float(np.sum(gammaln(sizes + 1) - gammaln(counts_a + 1) - gammaln(sizes - counts_a + 1)) / np.log(2.))
    assert abs(bits - report["exact_score_group_ambiguity"]["within_group_permutation_log2_count"]) < 1e-8
    assert int((y[:, 0] != y[:, 1]).sum()) == report["A_B_different_rows"]
    assert int((y[:, 0] != y[:, 2]).sum()) == report["A_C_different_rows"]
    np.testing.assert_array_equal(queried, np.argsort(-q[:, 1], kind="stable")[:protocol["settings"]["opposite_queries"]])
    successful = 0
    actual_opposite = []
    for record, row in zip(report["opposite_label_checks"], queried):
        assert record["id"] == row and record["base_hypothesis_label"] == int(y[row, 0])
        if not record["opposite_feasible_assignment_found"]:
            continue
        labels = opposite[:, record["witness_column"]]
        assert labels[row] == 1 - y[row, 0]
        measured = check_labels(labels)
        np.testing.assert_allclose(measured, [record["actual_auc"][version] for version in versions], rtol=0., atol=1e-12)
        actual_opposite.append({"row": int(row), "flipped_label": int(labels[row]), "auc": measured})
        successful += 1
    assert successful == opposite.shape[1] == report["opposite_checks_successful"]
    with np.load(OLD_OUT / "inference_arrays.npz", allow_pickle=False) as archive:
        old_y, old_q = archive["hypotheses"], archive["probabilities"]
        old_recheck = {name: float(roc_auc_score(old_y[:, j], predictions[:, 4])) for j, name in enumerate(["A", "B", "C"])}
        for name, value in old_recheck.items():
            assert abs(value - report["dataset_comparison"]["old_hypotheses_fifth_auc"][name]) < 1e-12
        top_old = np.argpartition(old_q[:, 1], -count)[-count:]
        top_new = np.argpartition(q[:, 1], -count)[-count:]
        assert count - len(np.intersect1d(top_old, top_new)) == report["probability_comparison"]["top_count_replaced_rows"]
    result = {"status": "passed", "rows": n, "all_five_auc_matched_by_three_main_hypotheses": True,
              "actual_auc_by_hypothesis": actual, "old_hypotheses_fifth_auc": old_recheck,
              "opposite_label_witnesses_independently_verified": successful, "opposite_checks": len(queried),
              "actual_auc_by_opposite_witness": actual_opposite,
              "maximum_csv_probability_rounding_error": error, "probability_reload_errors": reload_errors,
              "A_B_different_rows": report["A_B_different_rows"], "A_C_different_rows": report["A_C_different_rows"],
              "equal_score_swap_log2_count": bits, "equal_prior_log_likelihood_difference_A_B": same_prior_difference,
              "raw_datasets_all_scored_sources_and_old_results_unchanged": True,
              "actual_final_labels_recovered": False, "click_counts_are_assumptions": True}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "actual_auc_by_opposite_witness"}), flush=True)


if __name__ == "__main__":
    verify()
