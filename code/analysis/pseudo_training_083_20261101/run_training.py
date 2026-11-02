"""Train and evaluate a bounded, explicitly authorized fitted-label residual."""
from common import *


def run():
    if OUT.exists():
        raise FileExistsError("Preserving pseudo-training results")
    labels_path = ROOT / "output/fitted_labels_six_scores_final_20191117/fitted_labels_20191117.csv"
    expected_labels = "001b8b5457b1ae3cdb28437f8df501f0dafb0a3f7b33c136ec71efb9f5b7d7b6"
    assert digest(labels_path) == expected_labels
    assert digest(ROOT / "output/platform_feedback_v9/submission_v9.csv") == "c55583694642524fbabe71b5a0d4d06e86582076e899d1087d1d9dfd73271a83"
    OUT.mkdir(parents=True)
    sources = {str(path.relative_to(ROOT)): digest(path) for path in [HERE / "common.py", HERE / "run_training.py", HERE / "verify_training.py", HERE / "PROTOCOL.md",
        ROOT / "analysis/platform_delta_blend_20261101/blend_rules.py", ROOT / "analysis/auc_label_inference_20261101/infer_labels.py",
        ROOT / "analysis/auc_label_fit_six_bounded_20261101/fit_labels.py"]}
    protocol = {"settings": SETTINGS, "source_sha256": sources, "explicit_user_authorized_fitted_label_training": True,
        "fitted_labels_path": str(labels_path.relative_to(ROOT)), "fitted_labels_sha256": expected_labels,
        "base_v9_frozen_sha256": digest(ROOT / "output/platform_feedback_v9/submission_v9.csv"),
        "raw_dataset_sha256": {name: digest(ROOT / "2026ml" / name) for name in ["train.csv", "final_test.csv"]},
        "all_development_weights_locked_before_real_day16_evaluation": True,
        "real_platform_auc_unknown": True, "unbounded_085_goal_remains_paused": True}
    write(OUT / "protocol.json", protocol)
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d", "label"], dtype="int32")
    historical = raw[raw.pt_d.lt(20191116)].copy()
    rows15, p15, hashes15 = frozen.development_arrays(15)
    rows16, p16, hashes16 = frozen.development_arrays(16)
    np.testing.assert_array_equal(rows15, np.flatnonzero(raw.pt_d.to_numpy() == 20191115))
    np.testing.assert_array_equal(rows16, np.flatnonzero(raw.pt_d.to_numpy() == 20191116))
    y15 = historical.loc[historical.pt_d.eq(20191115), "label"].to_numpy()
    del raw
    x15, offset15, base15 = columns(p15)
    x16, offset16, base16 = columns(p16)
    pseudo16, generation = simulate_day16(p16, historical)
    del historical
    np.save(OUT / "simulation_day16_labels.npy", pseudo16)
    write(OUT / "simulation_day16.json", generation)
    emit("strict_past_pseudo_day16_generated", assumed_clicks=generation["assumed_clicks"], source_auc=generation["past_feedback_auc"], actual_target_labels_used=False)
    arms, predictions = {}, {}
    for alpha in SETTINGS["pseudo_weights"]:
        name = f"weight_{alpha:g}"
        model = fit_model(x15, offset15, y15, x16, offset16, pseudo16, alpha)
        arms[name] = model
        predictions[name] = probability(x16, offset16, model)
        write(OUT / f"development_{name}.json", model)
        emit("development_residual_fitted", name=name, objective=model["objective"], maximum_gradient=model["maximum_gradient"])
    permuted = np.random.default_rng(77329).permutation(pseudo16)
    model = fit_model(x15, offset15, y15, x16, offset16, permuted, SETTINGS["permuted_weight"])
    arms["permuted_01"] = model
    predictions["permuted_01"] = probability(x16, offset16, model)
    write(OUT / "development_permuted_01.json", model)
    locked = {name: digest(OUT / f"development_{name}.json") for name in arms}
    write(OUT / "development_locked_before_evaluation.json", {"models": locked, "target_real_outcomes_loaded_for_evaluation": False})
    observed = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["label"], dtype="int32").label.to_numpy()
    y16 = observed[rows16].copy()
    del observed
    scores = {name: metrics(y16, value) for name, value in predictions.items()}
    scores["v9_frozen"] = metrics(y16, base16)
    positive = [f"weight_{alpha:g}" for alpha in SETTINGS["pseudo_weights"] if alpha > 0]
    selected = max(positive, key=lambda name: (scores[name]["auc"], -arms[name]["pseudo_weight"]))
    alpha = arms[selected]["pseudo_weight"]
    write(OUT / "development_evaluation.json", {"real_day": 20191116, "scores": scores, "selected_positive_weight_arm": selected,
        "selected_weight": alpha, "baseline_v9": scores["v9_frozen"], "historically_reused_development": True,
        "no_target_real_outcomes_in_any_model_fit": True, "model_lock_sha256": locked,
        "prediction_array_sha256": {"15": hashes15, "16": hashes16}})
    np.savez_compressed(OUT / "development_predictions.npz", **predictions)
    emit("real_day16_evaluation", scores=scores, selected=selected, selected_weight=alpha)
    final_predictions, final_hashes = frozen.final_arrays()
    x17, offset17, base17 = columns(final_predictions)
    fitted = pd.read_csv(labels_path, float_precision="round_trip")
    final = pd.read_csv(ROOT / "2026ml/final_test.csv", usecols=["log_id", "pt_d"])
    np.testing.assert_array_equal(fitted.id.to_numpy(), np.arange(len(final)))
    np.testing.assert_array_equal(fitted.log_id.to_numpy(), final.log_id.to_numpy())
    pseudo17 = fitted.label.to_numpy(dtype="int8")
    assert final.pt_d.eq(20191117).all()
    model = fit_model(x16, offset16, y16, x17, offset17, pseudo17, alpha)
    control = fit_model(x16, offset16, y16, x17, offset17, pseudo17, 0.)
    write(OUT / "final_pseudo_model.json", model)
    write(OUT / "final_true_only_model.json", control)
    prediction = probability(x17, offset17, model)
    control_prediction = probability(x17, offset17, control)
    np.save(OUT / "final_prediction.npy", prediction)
    np.save(OUT / "final_true_only_prediction.npy", control_prediction)
    pd.DataFrame({"id": np.arange(len(final)), "pctr": prediction}).to_csv(OUT / "submission_pseudo_experimental.csv", index=False, float_format="%.10f")
    synthetic = {"v9_frozen": metrics(pseudo17, base17), "true_only_residual": metrics(pseudo17, control_prediction),
                 "pseudo_trained_residual": metrics(pseudo17, prediction)}
    report = {"status": "trained_and_exported_experimental_candidate_pending_verification",
        "selected_weight": alpha, "development_real_day16_scores": scores,
        "development_gain_vs_frozen_v9": scores[selected]["auc"] - scores["v9_frozen"]["auc"],
        "development_gain_vs_true_only": scores[selected]["auc"] - scores["weight_0"]["auc"],
        "development_supports_promotion_vs_both_controls": scores[selected]["auc"] > max(scores["v9_frozen"]["auc"], scores["weight_0"]["auc"]),
        "day17_SYNTHETIC_LABEL_IN_SAMPLE_TRAINING_DIAGNOSTIC_ONLY": synthetic,
        "desired_real_platform_auc": .83, "real_day17_platform_auc": None,
        "real_083_target_verified": False, "synthetic_metric_is_not_real_platform_auc": True,
        "pseudo_rows": len(pseudo17), "pseudo_ones_assumed": int(pseudo17.sum()), "true_source_rows": len(y16),
        "residual_models_trained": len(arms) + 2, "frozen_base_models_retrained": 0,
        "final_prediction_array_hashes": final_hashes,
        "original_best_submission_preserved": True,
        "files_sha256": {name: digest(OUT / name) for name in ["simulation_day16_labels.npy", "final_pseudo_model.json", "final_true_only_model.json", "final_prediction.npy", "final_true_only_prediction.npy", "submission_pseudo_experimental.csv", "development_predictions.npz"]}}
    write(OUT / "report.json", report)
    emit("pseudo_training_submission_exported", real16_gain=report["development_gain_vs_frozen_v9"],
         synthetic17_auc=synthetic["pseudo_trained_residual"]["auc"], real_platform_auc=None, target083_verified=False)


if __name__ == "__main__":
    run()
