"""Export one selected candidate, then verify with independent odds arithmetic."""
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
from blend_rules import (HERE, ROOT, OUT, VERSIONS, FEEDBACK, RULES, EXPECTED_AUC,
    digest, read, write, emit, apply, metrics, development_arrays, final_arrays, paired_interval)


def independent_rule(rule, predictions):
    six, seven, eight = (predictions[name] for name in ["v6", "v7", "v8"])
    if rule == "v6_75_v8_25":
        return (3 * six + eight) / 4
    ratio = (eight / (1 - eight)) / (seven / (1 - seven))
    ratio = np.clip(ratio, math.exp(-.75), math.exp(.75))
    if rule == "half_delta_odds_cap075":
        ratio = np.sqrt(ratio)
    elif rule != "delta_odds_cap075":
        raise ValueError("Unknown selected rule")
    odds = six / (1 - six) * ratio
    return odds / (1 + odds)


def contracts():
    protocol = read(HERE / "protocol.json")
    for path, expected in protocol["source_sha256"].items():
        if digest(path) != expected:
            raise AssertionError("Candidate formula/selection source changed")
    if digest(HERE / "feedback_bound.json") != protocol["feedback_manifest_sha256"]:
        raise AssertionError("Score/file attribution changed")
    comparison, selection = read(HERE / "comparison.json"), read(HERE / "selection_day15.json")
    if comparison["status"] != "selected_and_confirmed" or selection["day16_used_for_rule_selection"]:
        raise AssertionError("Candidate did not pass its recorded chronological checks")
    if digest(HERE / "selection_day15.json") != comparison["selection_day15_sha256"]:
        raise AssertionError("Day15 selection lock changed")
    for name, feedback in FEEDBACK.items():
        if digest(ROOT / "output" / VERSIONS[name] / "submission.csv") != feedback["sha256"]:
            raise AssertionError("A scored original submission changed")
    return protocol, comparison, selection


def export():
    protocol, comparison, selection = contracts()
    if OUT.exists():
        raise FileExistsError("Preserving exported feedback candidate")
    predictions, hashes = final_arrays()
    selected = selection["rule"]
    p = apply(selected, predictions)
    final_ids = pd.read_csv(ROOT / "2026ml/final_test.csv", usecols=["log_id"], dtype="int64")
    if len(final_ids) != len(p) or int(final_ids.log_id.duplicated().sum()) != 858:
        raise AssertionError("Original final order/count/duplicate contract differs")
    OUT.mkdir(parents=True)
    np.save(OUT / "prediction.npy", p)
    submission = OUT / "submission_v9.csv"
    pd.DataFrame({"id": np.arange(len(p)), "pctr": p}).to_csv(submission, index=False, float_format="%.10f")
    names = {"delta_odds_cap075": "logit(p6)+clip(logit(p8)-logit(p7),-.75,.75)",
             "half_delta_odds_cap075": "logit(p6)+.5*clip(logit(p8)-logit(p7),-.75,.75)",
             "v6_75_v8_25": ".75*p6+.25*p8"}
    manifest = {"status": "exported", "selected_rule": selected, "formula": names[selected],
        "probability_transform": "sigmoid after logit formula; weighted average for blend rule",
        "rows": len(p), "columns": ["id", "pctr"], "id_rule": "Original final row index starting at0",
        "duplicate_log_id_rows_preserved": 858, "minimum": float(p.min()), "maximum": float(p.max()),
        "mean": float(p.mean()), "source_predictions_sha256": hashes,
        "feedback": {name: {"reported_platform_auc": value["auc"], "submission_sha256": value["sha256"]} for name, value in FEEDBACK.items()},
        "feedback_is_human_reported": True, "analysis_protocol_sha256": digest(HERE / "protocol.json"),
        "selection_sha256": digest(HERE / "selection_day15.json"), "comparison_sha256": digest(HERE / "comparison.json"),
        "files_sha256": {path.name: digest(path) for path in [submission, OUT / "prediction.npy"]},
        "final_label_column_read": False, "models_trained": 0, "final_auc": None,
        "platform_improvement_verified": False, "caveat": "Development lift and prior feedback do not guarantee a higher platform AUC"}
    write(OUT / "manifest.json", manifest)
    emit("feedback_submission_exported", file=str(submission), rule=selected, rows=len(p), final_auc=None)


def verify():
    protocol, comparison, selection = contracts()
    manifest = read(OUT / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        if digest(OUT / name) != expected:
            raise AssertionError("Exported candidate file changed")
    if manifest["comparison_sha256"] != digest(HERE / "comparison.json"):
        raise AssertionError("Candidate comparison changed")
    rule = selection["rule"]
    final, hashes = final_arrays()
    if hashes != manifest["source_predictions_sha256"]:
        raise AssertionError("Source final prediction arrays changed")
    probability = independent_rule(rule, final)
    expected = np.load(OUT / "prediction.npy")
    replay_error = float(np.max(np.abs(probability - expected)))
    if replay_error > 1e-12:
        raise AssertionError("Independent ratio arithmetic differs from logit formula")
    frame = pd.read_csv(OUT / "submission_v9.csv", float_precision="round_trip")
    np.testing.assert_array_equal(frame.id.to_numpy(), np.arange(404182))
    if list(frame.columns) != ["id", "pctr"] or len(frame) != 404182:
        raise AssertionError("New submission schema differs")
    if not np.isfinite(frame.pctr).all() or np.any((frame.pctr <= 0) | (frame.pctr >= 1)):
        raise AssertionError("Invalid exported candidate probabilities")
    rounding_error = float(np.max(np.abs(frame.pctr.to_numpy() - expected)))
    if rounding_error > 5.0001e-11:
        raise AssertionError("Ten-decimal CSV does not reproduce full-precision prediction")
    raw_final = pd.read_csv(ROOT / "2026ml/final_test.csv", usecols=["log_id"], dtype="int64")
    if len(raw_final) != len(frame) or int(raw_final.log_id.duplicated().sum()) != 858:
        raise AssertionError("Final rows/duplicate log IDs changed")
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d", "label", "user_id"], dtype="int32")
    development = {}
    for day in [15, 16]:
        rows, inputs, source_hashes = development_arrays(day)
        np.testing.assert_array_equal(rows, np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day))
        recorded = comparison["development"][str(day)]
        if source_hashes != recorded["development_arrays_sha256"]:
            raise AssertionError("Historical comparison arrays changed")
        y, users = raw.label.to_numpy()[rows], raw.user_id.to_numpy()[rows]
        candidate = independent_rule(rule, inputs)
        measured = metrics(y, candidate)
        expected_scores = recorded["candidates"][rule]
        for key, value in measured.items():
            if abs(value - expected_scores[key]) > 1e-12:
                raise AssertionError("Independent full-day comparison differs")
        for version, values in inputs.items():
            if abs(metrics(y, values)["auc"] - EXPECTED_AUC[day][version]) > 1e-12:
                raise AssertionError("Original comparison model AUC changed")
        if measured["auc"] <= metrics(y, inputs["v6"])["auc"]:
            raise AssertionError("Selected candidate does not improve this complete development day")
        development[str(20191100 + day)] = {"baseline_v6": metrics(y, inputs["v6"]), "candidate": measured,
            "rounded10_auc": metrics(y, np.round(candidate, 10))["auc"],
            "versus_v6": paired_interval(y, candidate, inputs["v6"], users)}
    result = {"status": "passed", "independent_process": True, "rows": 404182,
        "maximum_independent_formula_error": replay_error, "maximum_csv_rounding_error": rounding_error,
        "original_row_order_and_zero_based_ids_preserved": True, "duplicate_log_id_rows_preserved": 858,
        "development": development, "source_submission_files_unchanged": True,
        "final_label_column_read": False, "models_trained": 0, "final_auc": None,
        "platform_improvement_verified": False, "submission_sha256": digest(OUT / "submission_v9.csv")}
    write(OUT / "verification.json", result)
    lines = ["# 平台反馈修正候选 v9", "", "提交文件：submission_v9.csv。404182行，原行顺序与id,pctr格式保留。", "",
        "按实际附件SHA256绑定的历史平台反馈：v6=0.80817，v7=0.80581，v8=0.80745。", "",
        f"本版固定公式：{manifest['formula']}。候选在15日选择、16日确认，没有读取最终广告标签。", "",
        "| 开发日期 | v6 AUC | 新候选AUC | 差值 |", "| --- | ---: | ---: | ---: |"]
    for day, values in development.items():
        lines.append(f"| {day} | {values['baseline_v6']['auc']:.9f} | {values['candidate']['auc']:.9f} | {values['versus_v6']['auc_delta']:+.9f} |")
    lines += ["", "以平台表现较好的v6为基础，检验从v7到v8的修正是否可迁移。", "开发日期反复使用，区间和分数是开发证据；两次平台总分不足以推导逐行真值。", "",
        "新文件的平台AUC未知，不能保证高于0.80817。需要实际提交后确认；未宣称达到0.85。", "",
        f"独立赔率乘法与原logit计算最大差：{replay_error:.3g}。CSV舍入误差：{rounding_error:.3g}。", "原v6、v7、v8文件均保留。"]
    (OUT / "提交说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit("feedback_submission_independently_verified", **result)


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else export()
