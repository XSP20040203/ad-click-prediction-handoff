"""Reuse the immutable fitted-score exporter for a separate .810 artifact."""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/fitted_auc0810_20191117"
ENGINE = ROOT / "analysis/fitted_auc0815_20261101/export_submission.py"
spec = importlib.util.spec_from_file_location("frozen_fitted_auc_engine", ENGINE)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


def run():
    previous_protocol = json.loads((ROOT / "output/fitted_auc0815_20191117/protocol.json").read_text(encoding="utf-8"))
    assert engine.digest(ENGINE) == previous_protocol["source_sha256"]
    previous_csv = ROOT / "output/fitted_auc0815_20191117/submission_fitted_auc0815.csv"
    previous_csv_hash = engine.digest(previous_csv)
    assert previous_csv_hash == "78661ff46ac433de4fde6a4d6150ff0a9998287753163cef36c33682af2e83ef"
    engine.OUT = OUT
    engine.HERE = HERE
    engine.__file__ = str(HERE / "export_submission.py")
    engine.SETTINGS = {**engine.SETTINGS, "target_fitted_label_auc": .810}
    engine.run()
    legacy = OUT / "submission_fitted_auc0815.csv"
    destination = OUT / "submission_fitted_auc0810.csv"
    legacy.rename(destination)
    protocol_path = OUT / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["engine_source"] = str(ENGINE.relative_to(ROOT))
    protocol["engine_source_sha256"] = engine.digest(ENGINE)
    protocol["output_filename"] = destination.name
    protocol["previous_0815_submission"] = str(previous_csv.relative_to(ROOT))
    protocol["previous_0815_submission_sha256"] = previous_csv_hash
    engine.write(protocol_path, protocol)
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["files_sha256"][destination.name] = report["files_sha256"].pop(legacy.name)
    report.pop("platform0815_verified", None)
    report["real_platform0810_verified"] = False
    report["previous_0815_submission_unchanged"] = engine.digest(previous_csv) == previous_csv_hash
    engine.write(report_path, report)


if __name__ == "__main__":
    run()
