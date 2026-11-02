"""Compare fixed candidates and select on day15 before confirming day16."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from blend_rules import (HERE, ROOT, VERSIONS, FEEDBACK, RULES, EXPECTED_AUC,
    digest, read, write, emit, apply, metrics, development_arrays, final_arrays, paired_interval)


def run():
    if (HERE / "comparison.json").exists():
        raise FileExistsError("Preserving feedback-based candidate selection")
    archives = HERE / "scored_inputs"
    archives.mkdir(exist_ok=True)
    records = []
    final, final_hashes = final_arrays()
    for name, feedback in FEEDBACK.items():
        path = Path(feedback["path"])
        stored = ROOT / "output" / VERSIONS[name] / "submission.csv"
        if digest(path) != feedback["sha256"] or digest(stored) != feedback["sha256"]:
            raise AssertionError("Score-to-file attribution differs from the user's attached files")
        frame = pd.read_csv(path, float_precision="round_trip")
        if list(frame.columns) != ["id", "pctr"] or len(frame) != 404182:
            raise AssertionError("Scored source schema differs")
        np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(404182))
        tolerance = 5.0001e-11 if name == "v8" else 5.0001e-7
        difference = float(np.max(np.abs(frame.pctr.to_numpy() - final[name])))
        if difference > tolerance:
            raise AssertionError("Unrounded array does not reproduce the actually scored file")
        archive = archives / f"{name}_auc{str(feedback['auc']).replace('.', '')}_{feedback['sha256'][:12]}.csv"
        if archive.exists():
            if digest(archive) != feedback["sha256"]:
                raise AssertionError("Existing score archive changed")
        else:
            shutil.copy2(path, archive)
        records.append({"version": name, **feedback, "archive": str(archive),
                        "source_csv_array_rounding_error": difference})
    write(HERE / "feedback_bound.json", {"status": "passed", "scores": records,
        "file_bindings_override_earlier_ambiguous_version_names": True,
        "platform_scores_are_human_reported": True, "platform_labels_available": False})
    sources = {str(HERE / name): digest(HERE / name) for name in ["blend_rules.py", "analyze.py", "export_verify.py"]}
    protocol = {"status": "fixed_before_day15_selection", "rules": RULES,
        "source_sha256": sources, "final_array_sha256": final_hashes,
        "feedback_manifest_sha256": digest(HERE / "feedback_bound.json"),
        "selection": "Largest full-day15 AUC among the three fixed rules beating v6, with Logloss worsening<=1e-4",
        "confirmation": "Selected rule must also beat v6 full-day16 AUC",
        "no_rule_coefficients_fit": True, "final_labels_read": False,
        "reason": "Anchor platform-best v6 and test the v8-v7 correction that improved platform feedback",
        "caveat": "Development days reused; three platform scores cannot identify final row labels or guarantee improvement"}
    write(HERE / "protocol.json", protocol)
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d", "label", "user_id", "task_id"], dtype="int32")
    all_results, bundles = {}, {}
    for day in [15, 16]:
        indices, predictions, hashes = development_arrays(day)
        np.testing.assert_array_equal(indices, np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day))
        labels = raw.label.to_numpy()[indices]
        users = raw.user_id.to_numpy()[indices]
        baseline = {name: metrics(labels, values) for name, values in predictions.items()}
        for name, measured in baseline.items():
            if abs(measured["auc"] - EXPECTED_AUC[day][name]) > 1e-12:
                raise AssertionError("Historical development prediction does not rescore as recorded")
        measured = {}
        for rule in RULES:
            values = apply(rule, predictions)
            measured[rule] = {**metrics(labels, values), "versus_v6": paired_interval(labels, values, predictions["v6"], users)}
        raw_delta = predictions["v6"] + predictions["v8"] - predictions["v7"]
        diagnostic = {"raw_additive_delta_auc": float(roc_auc_score(labels, raw_delta)),
            "outside_probability_range": int(((raw_delta <= 0) | (raw_delta >= 1)).sum()),
            "not_an_export_candidate": True}
        all_results[str(day)] = {"rows": len(indices), "clicks": int(labels.sum()), "baselines": baseline,
            "candidates": measured, "additive_diagnostic": diagnostic, "development_arrays_sha256": hashes}
        bundles[day] = (indices, predictions, labels, users)
        if day == 15:
            eligible = [rule for rule in RULES if measured[rule]["auc"] > baseline["v6"]["auc"]
                        and measured[rule]["logloss"] <= baseline["v6"]["logloss"] + .0001]
            if not eligible:
                raise RuntimeError("None of the fixed candidates improves day15; do not claim improvement")
            chosen = max(eligible, key=lambda rule: measured[rule]["auc"])
            write(HERE / "selection_day15.json", {"rule": chosen, "candidates": measured,
                "day16_used_for_rule_selection": False, "protocol_sha256": digest(HERE / "protocol.json")})
        else:
            if measured[chosen]["auc"] <= baseline["v6"]["auc"]:
                raise RuntimeError("The day15-selected candidate failed day16 confirmation")
        emit("feedback_candidate_development", day=day, **all_results[str(day)])
    final_selected = apply(chosen, final)
    report = {"status": "selected_and_confirmed", "selected_rule": chosen, "development": all_results,
        "final_prediction_min": float(final_selected.min()), "final_prediction_max": float(final_selected.max()),
        "final_prediction_mean": float(final_selected.mean()), "final_rows": len(final_selected),
        "final_auc": None, "platform_improvement_verified": False,
        "selection_day15_sha256": digest(HERE / "selection_day15.json"), "models_trained": 0}
    write(HERE / "comparison.json", report)
    pd.DataFrame([{"day": int(day), "rule": rule, **{key: value for key, value in values.items() if key != "versus_v6"}}
        for day, record in all_results.items() for rule, values in record["candidates"].items()]).to_csv(HERE / "comparison.csv", index=False)
    emit("feedback_candidate_selected", rule=chosen, final_rows=len(final_selected), final_auc=None)


if __name__ == "__main__":
    run()
