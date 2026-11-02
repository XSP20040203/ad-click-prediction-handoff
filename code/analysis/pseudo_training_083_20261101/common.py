"""Frozen ensemble features and convex, regularized residual fitting."""
import os
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "3"
from pathlib import Path
import hashlib
import json
import sys
import importlib.util
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/pseudo_training_083_20191117"
sys.path.insert(0, str(ROOT / "analysis/platform_delta_blend_20261101"))
import blend_rules as frozen
sys.path.insert(0, str(ROOT / "analysis/auc_label_inference_20261101"))
import infer_labels as inference
SETTINGS = {"pseudo_weights": [0., .01, .03, .1, .3, 1.], "permuted_weight": .1,
    "ridge": .0003, "correction_log_odds_cap": .5, "feature_cap": 8.,
    "optimizer_steps": 80, "pseudo_seed": 617231,
    "day16_temperatures": [.125, .03125], "day16_assumed_auc_source": "Actual day15 rounded AUCs only",
    "desired_real_platform_auc": .83, "fitted_label_auc_is_not_real_platform_auc": True}
FEATURES = ["v9_logit", "v7_v6_logit_delta", "v8_v6_logit_delta", "v9_rank",
    "v7_v6_rank_delta", "v8_v6_rank_delta", "v9_logit_times_delta76", "v9_logit_times_delta86",
    "absolute_logit_disagreement76", "absolute_logit_disagreement86", "v9_probability", "rank_disagreement_product"]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def columns(predictions):
    p = np.column_stack([predictions[name] for name in ["v6", "v7", "v8"]])
    if not np.isfinite(p).all() or not ((p > 0) & (p < 1)).all():
        raise ValueError("Finite interior base probabilities required")
    prior = .75 * p[:, 0] + .25 * p[:, 2]
    logits = logit(p)
    b = logit(prior)
    d76, d86 = logits[:, 1] - logits[:, 0], logits[:, 2] - logits[:, 0]
    ranks = np.column_stack([rankdata(p[:, j], method="average") / len(p) for j in range(3)])
    rank_prior = rankdata(prior, method="average") / len(p)
    r76, r86 = ranks[:, 1] - ranks[:, 0], ranks[:, 2] - ranks[:, 0]
    values = np.column_stack([b, d76, d86, rank_prior, r76, r86, b * d76, b * d86,
                             np.abs(d76), np.abs(d86), prior, r76 * r86])
    return values, b, prior


def normalize(raw, mean, scale):
    return np.column_stack([np.ones(len(raw)), np.clip((raw - mean) / scale, -SETTINGS["feature_cap"], SETTINGS["feature_cap"])])


def probability(raw, offset, model):
    x = normalize(raw, np.array(model["mean"]), np.array(model["scale"]))
    correction = np.clip(x @ np.array(model["beta"]), -SETTINGS["correction_log_odds_cap"], SETTINGS["correction_log_odds_cap"])
    return expit(offset + correction)


def fit_model(raw_source, offset_source, labels_source, raw_target, offset_target, labels_target, alpha):
    if not np.isin(labels_source, [0, 1]).all() or not np.isin(labels_target, [0, 1]).all():
        raise ValueError("Declared binary true and pseudo labels required")
    mean = raw_source.mean(axis=0)
    scale = np.maximum(raw_source.std(axis=0), .01)
    xs, xt = normalize(raw_source, mean, scale), normalize(raw_target, mean, scale)
    n, m = len(xs), len(xt)
    ridge = SETTINGS["ridge"]
    def objective(beta):
        ls, lt = offset_source + xs @ beta, offset_target + xt @ beta
        ps, pt = expit(ls), expit(lt)
        value = np.mean(np.logaddexp(0., ls) - labels_source * ls)
        value += alpha * np.mean(np.logaddexp(0., lt) - labels_target * lt) + ridge * (beta @ beta) / 2
        gradient = xs.T @ (ps - labels_source) / n + alpha * xt.T @ (pt - labels_target) / m + ridge * beta
        return float(value), gradient
    def hessian(beta):
        ps, pt = expit(offset_source + xs @ beta), expit(offset_target + xt @ beta)
        return xs.T @ (xs * (ps * (1 - ps))[:, None]) / n + alpha * xt.T @ (xt * (pt * (1 - pt))[:, None]) / m + ridge * np.eye(xs.shape[1])
    fit = minimize(objective, np.zeros(xs.shape[1]), jac=True, hess=hessian, method="trust-exact",
                   options={"gtol": 1e-10, "maxiter": SETTINGS["optimizer_steps"]})
    value, gradient = objective(fit.x)
    if np.max(np.abs(gradient)) > 1e-7:
        raise RuntimeError("Residual optimizer did not reach stationarity")
    return {"mean": mean.tolist(), "scale": scale.tolist(), "beta": fit.x.tolist(), "pseudo_weight": alpha,
        "features": FEATURES, "objective": value, "maximum_gradient": float(np.max(np.abs(gradient))),
        "optimizer_success": bool(fit.success), "optimizer_message": str(fit.message),
        "source_rows": n, "pseudo_rows": m, "true_labels_in_target_training": False}


def metrics(labels, values):
    return {"auc": float(roc_auc_score(labels, values)), "logloss": float(log_loss(labels, values, labels=[0, 1]))}


def simulate_day16(predictions, historical):
    """Historical excludes every day16 ad outcome before entering this function."""
    if historical.pt_d.max() >= 20191116:
        raise ValueError("Held-day labels cannot enter pseudo generation")
    _, p15, _ = frozen.development_arrays(15)
    y15 = historical.loc[historical.pt_d.eq(20191115), "label"].to_numpy()
    prior15 = .75 * p15["v6"] + .25 * p15["v8"]
    auc15 = np.array([float(f"{roc_auc_score(y15, value):.5f}") for value in [p15["v6"], p15["v7"], p15["v8"], prior15]])
    base = .75 * predictions["v6"] + .25 * predictions["v8"]
    matrix = np.column_stack([predictions["v6"], predictions["v7"], predictions["v8"], base])
    n = len(base)
    recent = historical[historical.pt_d.isin([20191114, 20191115])]
    count = int(round(n * recent.label.mean()))
    u = inference.rank_features(matrix)
    q, whitening, projection = inference.conditioned_probabilities(base, u, auc15, count)
    z = u @ whitening
    design = np.column_stack([np.ones(n), z])
    target = np.r_[count, inference.targets(n, count, auc15) @ whitening]
    original = logit(np.clip(base, 1e-9, 1 - 1e-9))
    eta = np.array(projection["theta"])
    for temperature in SETTINGS["day16_temperatures"]:
        def objective(parameters):
            logits = (original + design @ parameters) / temperature
            return float((temperature * np.logaddexp(0., logits).sum() - parameters @ target) / n), (design.T @ expit(logits) - target) / n
        def hessian(parameters):
            probability = expit((original + design @ parameters) / temperature)
            return design.T @ (design * (probability * (1 - probability) / temperature)[:, None]) / n
        fit = minimize(objective, eta, jac=True, hess=hessian, method="trust-exact", options={"gtol": 1e-10, "maxiter": 80})
        eta = fit.x.copy()
    logits = (original + design @ eta) / SETTINGS["day16_temperatures"][-1]
    spec = importlib.util.spec_from_file_location("bounded_label_repair", ROOT / "analysis/auc_label_fit_six_bounded_20261101/fit_labels.py")
    repair_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repair_module)
    repair_module.SETTINGS = {**repair_module.SETTINGS, "assumed_click_count": count}
    candidates, info = [], []
    for seed in [SETTINGS["pseudo_seed"], SETTINGS["pseudo_seed"] + 1]:
        label, result = repair_module.repair(expit(logits), logits, u, z, whitening, auc15, seed)
        info.append(result)
        if label is not None:
            candidates.append(label)
    if not candidates:
        raise RuntimeError("No bounded strict-past pseudo assignment found")
    costs = [-original @ value for value in candidates]
    chosen = candidates[int(np.argmin(costs))]
    return chosen, {"assumed_clicks": count, "past_feedback_auc": auc15.tolist(), "past_dates_used": [20191114, 20191115],
        "no_actual_day16_labels_used": True, "projection": projection, "repairs": info,
        "pseudo_rank_auc": inference.rank_auc(chosen, u).tolist(), "retained_candidates": len(candidates)}
