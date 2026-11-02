"""Verify the six-score deliverable using the frozen independent verifier."""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / "analysis/auc_label_fit_six_20261101"
OUT = ROOT / "output/fitted_labels_six_scores_v2_20191117"
spec = importlib.util.spec_from_file_location("frozen_six_verifier", PREVIOUS / "verify_labels.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


if __name__ == "__main__":
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    assert verifier.digest(PREVIOUS / "fit_labels.py") == protocol["reused_driver_sha256"]
    assert verifier.digest(PREVIOUS / "verify_labels.py") == protocol["reused_verifier_sha256"]
    verifier.HERE = HERE
    verifier.OUT = OUT
    verifier.verify()
