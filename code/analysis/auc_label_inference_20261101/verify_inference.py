"""Verify compatibility, uncertainty witnesses and immutable score/data sources."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.special import expit, logit, gammaln
from sklearn.metrics import roc_auc_score
from infer_labels import (HERE, ROOT, OUT, FILES, VERSIONS, SCORES, digest, write, emit, rank_features, targets)


def verify():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    if digest(HERE / "infer_labels.py") != protocol["source_sha256"] or digest(HERE / "test_inference.py") != protocol["test_sha256"] or digest(__file__) != protocol["verifier_sha256"]:
        raise AssertionError("Inference source changed after fitting")
    for name, expected in protocol["original_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name) != expected:
            raise AssertionError("Original dataset changed")
    predictions = []
    for name in VERSIONS:
        relative, expected = FILES[name]
        if digest(ROOT / relative) != expected:
            raise AssertionError("Scored source submission changed")
        frame = pd.read_csv(ROOT / relative, float_precision="round_trip")
        np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(len(frame)))
        predictions.append(frame.pctr.to_numpy())
    predictions = np.column_stack(predictions)
    for name, expected in report["files_sha256"].items():
        if digest(OUT / name) != expected:
            raise AssertionError("Inference artifact changed")
    final = pd.read_csv(ROOT / "2026ml/final_test.csv", usecols=["log_id", "pt_d"])
    csv = pd.read_csv(OUT / "estimated_labels_20191117.csv", float_precision="round_trip")
    np.testing.assert_array_equal(csv.id.to_numpy(), np.arange(len(final)))
    np.testing.assert_array_equal(csv.log_id.to_numpy(), final.log_id.to_numpy())
    if not final.pt_d.eq(20191117).all():
        raise AssertionError("Inferred rows are not the complete17th dataset")
    with np.load(OUT / "inference_arrays.npz", allow_pickle=False) as arrays:
        probabilities, labels, groups = (arrays[name].copy() for name in ["probabilities", "hypotheses", "groups"])
    np.testing.assert_array_equal(csv[["label_hypothesis_A", "label_hypothesis_B", "label_hypothesis_C"]].to_numpy(), labels)
    columns = ["click_probability_low_count", "click_probability_central", "click_probability_high_count"]
    csv_error = float(np.max(np.abs(csv[columns].to_numpy() - probabilities)))
    if csv_error > 5.0001e-11 or not np.isfinite(probabilities).all():
        raise AssertionError("Conditional probability CSV differs")
    u = rank_features(predictions)
    covariance = u.T @ u / len(u)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitening = eigenvectors / np.sqrt(eigenvalues)[None, :]
    design = np.column_stack([np.ones(len(u)), u @ whitening])
    original = logit(np.clip(predictions[:, 3], 1e-9, 1 - 1e-9))
    projection_errors = []
    for column, record in enumerate(report["conditional_probabilities"]):
        replay = expit(original + design @ np.array(record["theta"]))
        error = float(np.max(np.abs(replay - probabilities[:, column])))
        count = record["assumed_positive_count"]
        implied = .5 + len(u) * (probabilities[:, column] @ u) / (count * (len(u) - count))
        if error > 1e-12 or np.max(np.abs(implied - SCORES)) > 1e-7 or abs(probabilities[:, column].sum() - count) > .001:
            raise AssertionError("Conditional rank projection does not reproduce")
        projection_errors.append(error)
    actual = {}
    for column, name in enumerate(["A", "B", "C"]):
        if not np.isin(labels[:, column], [0, 1]).all() or int(labels[:, column].sum()) != report["assumed_click_count"]:
            raise AssertionError("Binary hypothesis/cardinality differs")
        auc = [float(roc_auc_score(labels[:, column], predictions[:, j])) for j in range(4)]
        if any(f"{score:.5f}" != f"{target:.5f}" for score, target in zip(auc, SCORES)):
            raise AssertionError("A label hypothesis fails an actual AUC constraint")
        for version, score in zip(VERSIONS, auc):
            if abs(score - report["rounded_auc_matches"][name][version]) > 1e-12:
                raise AssertionError("Stored hypothesis AUC differs")
        actual[name] = auc
    _, rebuilt_groups, sizes = np.unique(predictions, axis=0, return_inverse=True, return_counts=True)
    np.testing.assert_array_equal(groups, rebuilt_groups)
    counts_a = np.bincount(groups, weights=labels[:, 0], minlength=len(sizes))
    counts_b = np.bincount(groups, weights=labels[:, 1], minlength=len(sizes))
    np.testing.assert_array_equal(counts_a, counts_b)
    if np.max(np.abs(np.array(actual["A"]) - actual["B"])) > 1e-12:
        raise AssertionError("Symmetric label swaps changed AUC")
    same_prior_difference = float(np.dot(labels[:, 0] - labels[:, 1], logit(probabilities[:, 1])))
    if abs(same_prior_difference) > 1e-8:
        raise AssertionError("Equal-score permutations unexpectedly changed prior likelihood")
    different_ab, different_ac = int((labels[:, 0] != labels[:, 1]).sum()), int((labels[:, 0] != labels[:, 2]).sum())
    if different_ab != report["A_B_different_rows"] or different_ac != report["A_C_different_rows"]:
        raise AssertionError("Ambiguity witness counts differ")
    bits = float(np.sum(gammaln(sizes + 1) - gammaln(counts_a + 1) - gammaln(sizes - counts_a + 1)) / np.log(2))
    result = {"status": "passed", "rows": len(final), "all_four_reported_auc_values_reproduced_by_each_hypothesis": True,
        "actual_auc_by_hypothesis": actual, "maximum_csv_probability_rounding_error": csv_error,
        "conditional_probability_reload_errors": projection_errors, "A_B_different_rows": different_ab,
        "A_C_different_rows": different_ac, "equal_score_permutation_prior_log_likelihood_difference": same_prior_difference,
        "log2_equally_scored_labels_from_score_tie_symmetry": bits,
        "original_datasets_and_submissions_unchanged": True,
        "actual17_labels_recovered": False, "conditional_hypotheses_are_not_ground_truth": True}
    write(OUT / "verification.json", result)
    emit("label_hypotheses_independently_verified", **result)


if __name__ == "__main__":
    verify()
