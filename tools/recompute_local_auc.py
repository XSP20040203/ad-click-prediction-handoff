"""Recompute local AUC for every packaged submission against the selected hypothesis."""
from pathlib import Path
import argparse
import csv
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def run(root):
    labels = pd.read_csv(root / "labels/l06_selected_six_score_hypothesis.csv", usecols=["id", "label"])
    catalog = pd.read_csv(root / "scores/submission_catalog.csv", keep_default_na=False)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["submission_id", "platform_auc_user_reported", "local_auc_against_l06_hypothesis_NOT_platform_auc"])
    for row in catalog.itertuples(index=False):
        submitted = pd.read_csv(root / row.archive_csv_path, float_precision="round_trip")
        np.testing.assert_array_equal(submitted.id.to_numpy(), labels.id.to_numpy())
        writer.writerow([row.submission_id, row.platform_auc_user_reported,
                         f"{roc_auc_score(labels.label, submitted.pctr):.12f}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run(args.root.resolve())
