"""Fixed feedback-guided submission rules; no final labels or model fitting."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/platform_feedback_v9"
VERSIONS = {"v6": "best_nextday_full_v6", "v7": "best_nextday_arch_v7", "v8": "best_nextday_daystate_v8"}
FEEDBACK = {
    "v6": {"auc": .80817, "path": "E:/浏览器下载/submission.csv", "sha256": "899f6f6f48bae40ec3920d7948ace3d26f478e839c865fbe3006c90c565f9e5f"},
    "v7": {"auc": .80581, "path": "E:/浏览器下载/submission (1).csv", "sha256": "c44a9e803b8b4566b9bcef4e1bf85181c2ee00eee71a9f84474b73c619c121b9"},
    "v8": {"auc": .80745, "path": "E:/浏览器下载/submission (2).csv", "sha256": "1b43ee597613ec6a48d448a2cb00b824aca3440e38a32d47e21ceadccd833846"},
}
FINAL_ARRAY_SHA = {"v6": "f527dc891756d365c6b681d95ee8719618c5c5a9272bcc2cc5e252e1876f0e21",
    "v7": "3edae903e51d2719e1b6e82bdcfdf08008c62b5c3aaf60bb4bedfcff462d75bb",
    "v8": "fbbf4cdec47561abb55927548769a0af6358f5d36d866a8ed8669872cfd34b84"}
RULES = ["delta_odds_cap075", "half_delta_odds_cap075", "v6_75_v8_25"]
EXPECTED_AUC = {15: {"v6": .7905480723412239, "v7": .7921069471244595, "v8": .7934700801999591},
                16: {"v6": .7987435691593445, "v7": .7998466417345275, "v8": .8012412673501383}}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def check(values, rows):
    if values.shape != (rows,) or not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
        raise ValueError("Aligned finite interior probabilities required")


def apply(rule, predictions):
    p6, p7, p8 = (predictions[name] for name in ["v6", "v7", "v8"])
    for values in [p6, p7, p8]:
        check(values, len(p6))
    if rule == "v6_75_v8_25":
        return .75 * p6 + .25 * p8
    if rule in ["delta_odds_cap075", "half_delta_odds_cap075"]:
        alpha = 1. if rule == "delta_odds_cap075" else .5
        correction = np.clip(logit(p8) - logit(p7), -.75, .75)
        return expit(logit(p6) + alpha * correction)
    raise ValueError("Unknown fixed candidate")


def metrics(labels, prediction):
    return {"auc": float(roc_auc_score(labels, prediction)),
        "logloss": float(log_loss(labels, prediction, labels=[0, 1])),
        "average_precision": float(average_precision_score(labels, prediction)),
        "mean_probability": float(prediction.mean())}


def paired_interval(labels, first, second, users):
    def influence(values):
        positives, negatives = np.sort(values[labels == 1]), np.sort(values[labels == 0])
        a = (np.searchsorted(negatives, values[labels == 1], "left") + np.searchsorted(negatives, values[labels == 1], "right")) / (2 * len(negatives))
        b = 1 - (np.searchsorted(positives, values[labels == 0], "left") + np.searchsorted(positives, values[labels == 0], "right")) / (2 * len(positives))
        auc = float(a.mean())
        if abs(auc - roc_auc_score(labels, values)) > 1e-12:
            raise AssertionError("Exact rank influence does not recover AUC")
        result = np.empty(len(values))
        result[labels == 1] = (a - auc) / len(positives)
        result[labels == 0] = (b - auc) / len(negatives)
        return auc, result
    aa, ia = influence(first)
    ab, ib = influence(second)
    _, codes = np.unique(users, return_inverse=True)
    clusters = np.bincount(codes, weights=ia - ib)
    variance = len(clusters) / max(len(clusters) - 1, 1) * np.square(clusters - clusters.mean()).sum()
    margin = 1.96 * math.sqrt(variance)
    return {"auc_delta": aa - ab, "user_cluster_approximate_95_interval": [aa - ab - margin, aa - ab + margin]}


def development_arrays(day):
    previous = ROOT / "analysis/nextday_goal_20261029"
    paths = {"v6": previous / f"development_candidate_day{day}_prediction.npy",
        "v7": ROOT / f"analysis/architecture_goal_20261029/fixed_gated20_allocated10_core30_day{day}.npy",
        "v8": ROOT / f"analysis/userday_aux_20261101/reference_day_state_protected_seed42_v1/day{day}_prediction.npy"}
    indices = np.load(previous / f"development_candidate_day{day}_indices.npy")
    np.testing.assert_array_equal(indices, np.load(ROOT / f"analysis/auc085_next/evaluation/day{day}_valid_indices.npy"))
    np.testing.assert_array_equal(indices, np.load(ROOT / f"analysis/userday_aux_20261101/reference_day_state_protected_seed42_v1/day{day}_indices.npy"))
    predictions = {name: np.load(path).astype("float64") for name, path in paths.items()}
    for values in predictions.values():
        check(values, len(indices))
    return indices, predictions, {str(path): digest(path) for path in paths.values()}


def final_arrays():
    predictions, files = {}, {}
    for name, folder in VERSIONS.items():
        path = ROOT / "output" / folder / "ensemble_final_prediction.npy"
        if digest(path) != FINAL_ARRAY_SHA[name]:
            raise AssertionError("Retained unrounded prediction changed")
        values = np.load(path).astype("float64")
        check(values, 404182)
        predictions[name] = values
        files[str(path)] = digest(path)
    return predictions, files
