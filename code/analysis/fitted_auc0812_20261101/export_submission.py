"""Export the requested .812 variant using the frozen fitted-score engine."""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/fitted_auc0812_20191117"
ENGINE = ROOT / "analysis/fitted_auc0815_20261101/export_submission.py"
spec = importlib.util.spec_from_file_location("frozen_fitted_auc_engine", ENGINE)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
PREVIOUS = {
    "output/fitted_auc0810_20191117/submission_fitted_auc0810.csv": "7e239cb191876455e9038ad8180a5aec9db81e07e8d17862bfef8b1180d811fe",
    "output/fitted_auc0815_20191117/submission_fitted_auc0815.csv": "78661ff46ac433de4fde6a4d6150ff0a9998287753163cef36c33682af2e83ef"}


def run():
    original_protocol = json.loads((ROOT / "output/fitted_auc0815_20191117/protocol.json").read_text(encoding="utf-8"))
    assert engine.digest(ENGINE) == original_protocol["source_sha256"]
    for filename, expected in PREVIOUS.items():
        assert engine.digest(ROOT / filename) == expected
    feedback_path = ROOT / "output/fitted_auc0810_20191117/platform_feedback.json"
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    assert feedback["user_reported_platform_auc"] == .80855
    engine.OUT, engine.HERE = OUT, HERE
    engine.__file__ = str(HERE / "export_submission.py")
    engine.SETTINGS = {**engine.SETTINGS, "target_fitted_label_auc": .812}
    engine.run()
    legacy = OUT / "submission_fitted_auc0815.csv"
    destination = OUT / "submission_fitted_auc0812.csv"
    legacy.rename(destination)
    protocol_path = OUT / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update(engine_source=str(ENGINE.relative_to(ROOT)), engine_source_sha256=engine.digest(ENGINE),
        output_filename=destination.name, previous_submission_sha256=PREVIOUS,
        previous0810_feedback_file=str(feedback_path.relative_to(ROOT)), previous0810_feedback_sha256=engine.digest(feedback_path),
        prior0810_user_reported_platform_auc=.80855, original_six_score_fitted_labels_reused=True,
        no_new_seven_score_label_inference=True)
    engine.write(protocol_path, protocol)
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["files_sha256"][destination.name] = report["files_sha256"].pop(legacy.name)
    report.pop("platform0815_verified", None)
    report["real_platform0812_verified"] = False
    report["previous0810_user_reported_platform_auc"] = .80855
    report["original_fitted_hypothesis_does_not_fit_new0810_platform_feedback"] = True
    report["previous_submissions_unchanged"] = all(engine.digest(ROOT / name) == expected for name, expected in PREVIOUS.items())
    engine.write(report_path, report)


if __name__ == "__main__":
    run()
