"""Rebuild the fitted residual objectives and score real/synthetic labels separately."""
from common import *


def check_fit(model, source, offset_source, y_source, target, offset_target, y_target):
    mean, scale, beta = (np.array(model[key]) for key in ["mean", "scale", "beta"])
    np.testing.assert_allclose(mean, source.mean(0), atol=1e-12, rtol=0)
    np.testing.assert_allclose(scale, np.maximum(source.std(0), .01), atol=1e-12, rtol=0)
    xs, xt = normalize(source, mean, scale), normalize(target, mean, scale)
    ls, lt = offset_source + xs @ beta, offset_target + xt @ beta
    alpha = model["pseudo_weight"]
    loss = np.mean(np.logaddexp(0., ls) - y_source * ls) + alpha * np.mean(np.logaddexp(0., lt) - y_target * lt) + SETTINGS["ridge"] * (beta @ beta) / 2
    gradient = xs.T @ (expit(ls) - y_source) / len(xs) + alpha * xt.T @ (expit(lt) - y_target) / len(xt) + SETTINGS["ridge"] * beta
    assert abs(loss - model["objective"]) < 1e-12 and np.max(np.abs(gradient)) < 1e-7
    return float(np.max(np.abs(gradient)))


def verify():
    protocol, report = read(OUT / "protocol.json"), read(OUT / "report.json")
    for name, expected in protocol["source_sha256"].items():
        assert digest(ROOT / name) == expected
    for name, expected in protocol["raw_dataset_sha256"].items():
        assert digest(ROOT / "2026ml" / name) == expected
    assert digest(ROOT / protocol["fitted_labels_path"]) == protocol["fitted_labels_sha256"]
    assert digest(ROOT / "output/platform_feedback_v9/submission_v9.csv") == protocol["base_v9_frozen_sha256"]
    for name, expected in report["files_sha256"].items():
        assert digest(OUT / name) == expected
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d", "label"], dtype="int32")
    rows15, p15, _ = frozen.development_arrays(15)
    rows16, p16, _ = frozen.development_arrays(16)
    y15, y16 = raw.label.to_numpy()[rows15], raw.label.to_numpy()[rows16]
    x15, o15, base15 = columns(p15)
    x16, o16, base16 = columns(p16)
    pseudo16, generation = simulate_day16(p16, raw[raw.pt_d.lt(20191116)].copy())
    np.testing.assert_array_equal(pseudo16, np.load(OUT / "simulation_day16_labels.npy"))
    assert generation["no_actual_day16_labels_used"]
    locked = read(OUT / "development_locked_before_evaluation.json")["models"]
    saved = np.load(OUT / "development_predictions.npz", allow_pickle=False)
    maximum_replay, maximum_gradient = 0., 0.
    measured = {}
    for name in locked:
        path = OUT / f"development_{name}.json"
        assert digest(path) == locked[name]
        model = read(path)
        targets = np.random.default_rng(77329).permutation(pseudo16) if name == "permuted_01" else pseudo16
        maximum_gradient = max(maximum_gradient, check_fit(model, x15, o15, y15, x16, o16, targets))
        predicted = probability(x16, o16, model)
        error = float(np.max(np.abs(predicted - saved[name])))
        assert error < 1e-12
        maximum_replay = max(maximum_replay, error)
        measured[name] = metrics(y16, predicted)
    measured["v9_frozen"] = metrics(y16, base16)
    expected = report["development_real_day16_scores"]
    for name in measured:
        assert abs(measured[name]["auc"] - expected[name]["auc"]) < 1e-12
    candidates = [f"weight_{a:g}" for a in SETTINGS["pseudo_weights"] if a > 0]
    selected = max(candidates, key=lambda name: (measured[name]["auc"], -float(name.split("_")[1])))
    assert float(selected.split("_")[1]) == report["selected_weight"]
    p17, _ = frozen.final_arrays()
    x17, o17, base17 = columns(p17)
    final = pd.read_csv(ROOT / "2026ml/final_test.csv", usecols=["log_id", "pt_d"])
    fitted = pd.read_csv(ROOT / protocol["fitted_labels_path"])
    np.testing.assert_array_equal(fitted.log_id.to_numpy(), final.log_id.to_numpy())
    assert final.pt_d.eq(20191117).all()
    pseudo17 = fitted.label.to_numpy()
    for filename, prediction_name in [("final_pseudo_model.json", "final_prediction.npy"), ("final_true_only_model.json", "final_true_only_prediction.npy")]:
        model = read(OUT / filename)
        maximum_gradient = max(maximum_gradient, check_fit(model, x16, o16, y16, x17, o17, pseudo17))
        replay = probability(x17, o17, model)
        error = float(np.max(np.abs(replay - np.load(OUT / prediction_name))))
        assert error < 1e-12
        maximum_replay = max(maximum_replay, error)
    csv = pd.read_csv(OUT / "submission_pseudo_experimental.csv", float_precision="round_trip")
    np.testing.assert_array_equal(csv.id.to_numpy(), np.arange(len(final)))
    prediction = np.load(OUT / "final_prediction.npy")
    rounding = float(np.max(np.abs(csv.pctr.to_numpy() - prediction)))
    assert rounding <= 5.0001e-11 and np.isfinite(prediction).all() and ((prediction > 0) & (prediction < 1)).all()
    synthetic_auc = float(roc_auc_score(pseudo17, prediction))
    assert abs(synthetic_auc - report["day17_SYNTHETIC_LABEL_IN_SAMPLE_TRAINING_DIAGNOSTIC_ONLY"]["pseudo_trained_residual"]["auc"]) < 1e-12
    result = {"status": "passed", "models_independently_replayed": len(locked) + 2,
        "rows": len(final), "maximum_prediction_reload_error": maximum_replay,
        "maximum_objective_gradient": maximum_gradient, "maximum_csv_rounding_error": rounding,
        "day16_real_auc_frozen_v9": measured["v9_frozen"]["auc"],
        "day16_real_auc_true_only": measured["weight_0"]["auc"], "day16_real_auc_selected_pseudo_arm": measured[selected]["auc"],
        "selected_pseudo_weight": report["selected_weight"],
        "day17_auc_against_SYNTHETIC_TRAINING_LABELS_ONLY": synthetic_auc,
        "real_day17_platform_auc": None, "real083_target_verified": False,
        "original_datasets_fitted_labels_and_best_submission_unchanged": True,
        "duplicate_final_log_id_rows_preserved": int(final.log_id.duplicated().sum()),
        "submission_sha256": digest(OUT / "submission_pseudo_experimental.csv")}
    write(OUT / "verification.json", result)
    emit("pseudo_training_independently_verified", **result)


if __name__ == "__main__":
    verify()
