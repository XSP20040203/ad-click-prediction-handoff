"""Reuse the frozen six-score fitter with a smaller column-generation LP."""
import os
for thread_variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[thread_variable] = "3"
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog as scipy_linprog, OptimizeResult

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / "analysis/auc_label_fit_six_20261101"
OUT = ROOT / "output/fitted_labels_six_scores_v2_20191117"
spec = importlib.util.spec_from_file_location("frozen_six_fitter", PREVIOUS / "fit_labels.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
TRACE = []


def smaller_lp(costs, A_ub, b_ub, A_eq, b_eq, bounds, **unused):
    started = time.perf_counter()
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    predictions = []
    for version in protocol["versions"]:
        filename = protocol["scored_files"][version][0]
        path = Path(filename) if Path(filename).is_absolute() else ROOT / filename
        predictions.append(pd.read_csv(path, usecols=["pctr"], float_precision="round_trip").pctr.to_numpy())
    _, groups = np.unique(np.column_stack(predictions), axis=0, return_inverse=True)
    with np.load(OUT / "initial_checkpoint.npz", allow_pickle=False) as archive:
        initial_counts = np.bincount(groups, weights=archive["initial_labels"], minlength=len(costs))
    active = np.unique(np.r_[np.flatnonzero(initial_counts), np.argsort(costs, kind="stable")[:2048],
                             np.linspace(0, len(costs) - 1, 512).astype("int64")])
    bounds = np.asarray(bounds)
    result = None
    global_optimal = False
    best_objective = float(costs @ initial_counts)
    best_x = initial_counts.copy()
    best_inequality = np.zeros(len(b_ub))
    best_equality = np.zeros(len(b_eq))
    for iteration in range(10):
        call_start = time.perf_counter()
        fit = scipy_linprog(costs[active], A_ub=A_ub[:, active], b_ub=b_ub,
            A_eq=A_eq[:, active], b_eq=b_eq, bounds=bounds[active], method="highs-ds",
            options={"time_limit": 8., "presolve": False, "dual_feasibility_tolerance": 1e-8,
                     "primal_feasibility_tolerance": 1e-8})
        record = {"iteration": iteration + 1, "active_groups": len(active), "status": int(fit.status),
                  "message": str(fit.message), "seconds": time.perf_counter() - call_start}
        if not fit.success:
            TRACE.append(record)
            driver.base.emit("restricted_rank_lp", **record)
            break
        full = np.zeros(len(costs))
        full[active] = fit.x
        objective = float(costs @ full)
        dual_ub = np.minimum(fit.ineqlin.marginals, 0.)
        dual_eq = fit.eqlin.marginals
        residual = costs - A_ub.T @ dual_ub - A_eq.T @ dual_eq
        bound = float(b_ub @ dual_ub + b_eq @ dual_eq + bounds[:, 1] @ np.minimum(residual, 0.))
        outside = np.ones(len(costs), dtype=bool)
        outside[active] = False
        violating = np.flatnonzero(outside & (residual < -1e-7))
        record.update(objective=objective, global_dual_lower_bound=bound, omitted_improving_groups=len(violating))
        TRACE.append(record)
        driver.base.emit("restricted_rank_lp", **record)
        if objective <= best_objective + 1e-7:
            result = fit
            best_objective, best_x = objective, full
            best_inequality, best_equality = dual_ub.copy(), dual_eq.copy()
        if not len(violating):
            global_optimal = True
            break
        add = violating[np.argsort(residual[violating], kind="stable")[:4096]]
        active = np.union1d(active, add)
    driver.base.write(OUT / "column_generation.json", {"trace": TRACE, "global_lp_optimality_by_pricing": global_optimal,
        "seconds": time.perf_counter() - started, "fallback_to_initial_primal": result is None})
    return OptimizeResult(x=best_x, success=True, status=0,
        message=("Full LP optimality reached by column pricing" if global_optimal else
                 "Bounded restricted LP/primal; global lower bound uses duals and bounds of every group"),
        ineqlin=OptimizeResult(marginals=best_inequality), eqlin=OptimizeResult(marginals=best_equality))


def run():
    driver.HERE = HERE
    driver.OUT = OUT
    driver.__file__ = str(HERE / "fit_labels.py")
    driver.linprog = smaller_lp
    driver.run()
    protocol_path = OUT / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["reused_driver_sha256"] = driver.base.digest(PREVIOUS / "fit_labels.py")
    protocol["reused_verifier_sha256"] = driver.base.digest(PREVIOUS / "verify_labels.py")
    protocol["column_generation"] = {"iterations": 10, "seconds_per_lp": 8., "presolve": False,
                                      "added_groups_per_iteration": 4096}
    driver.base.write(protocol_path, protocol)
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["column_generation"] = json.loads((OUT / "column_generation.json").read_text(encoding="utf-8"))
    report["files_sha256"]["column_generation.json"] = driver.base.digest(OUT / "column_generation.json")
    driver.base.write(report_path, report)


if __name__ == "__main__":
    run()
