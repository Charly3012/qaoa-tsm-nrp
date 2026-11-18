#!/usr/bin/env python3
"""ILP exacto para TSM (Test Suite Minimization).

Minimiza el número de tests seleccionados tal que todo requisito quede
cubierto al menos una vez (problema de cobertura de conjuntos, resuelto con
PuLP/CBC).
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import pulp

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_io import evaluate_selection, load_tsm  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def solve_ilp(df_coverage, df_tests):
    """Resuelve el TSM de forma exacta. Devuelve (evaluación, tiempo)."""
    tests = df_coverage.index.tolist()
    n_reqs = len(df_coverage.columns)

    prob = pulp.LpProblem("TSM", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("test", tests, cat=pulp.LpBinary)

    # Función objetivo: minimizar número de tests seleccionados
    prob += pulp.lpSum([x[t] for t in tests])

    # Restricciones de cobertura: cada requisito cubierto al menos una vez
    for j in range(n_reqs):
        prob += (
            pulp.lpSum(
                [int(df_coverage.iloc[i, j]) * x[t] for i, t in enumerate(tests)]
            )
            >= 1,
            f"cov_R{j}",
        )

    start = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    elapsed = time.perf_counter() - start

    selected = [t for t in tests if pulp.value(x[t]) == 1.0]
    eval_ = evaluate_selection(selected, df_coverage, df_tests)
    eval_["status"] = pulp.LpStatus[prob.status]
    return eval_, elapsed


def _save(eval_, elapsed, dataset, repeat, method="ilp"):
    path = RESULTS_DIR / method / dataset / str(repeat)
    path.mkdir(parents=True, exist_ok=True)

    df_cov, df_tests = load_tsm(dataset)
    selected = eval_["selected"]
    result = df_tests.loc[selected][["Test", "time"]].reset_index(drop=True)
    result.to_csv(path / "result.csv", index=False)

    summary = {
        "method": method,
        "dataset": dataset,
        "repeat": repeat,
        "selected_case_num": eval_["selected_case_num"],
        "total_time": eval_["total_time"],
        "covered_reqs": eval_["covered_reqs"],
        "total_reqs": eval_["total_reqs"],
        "feasible": eval_["feasible"],
        "objective": eval_["objective"],
        "execution_time": elapsed,
        "optimal": "yes",
        "status": eval_.get("status", ""),
    }
    pd.DataFrame([summary]).to_csv(path / "summary.csv", index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="ILP exacto para TSM")
    parser.add_argument("dataset", type=str, help="Nombre del dataset (toy, toy2, ...)")
    parser.add_argument("--repeat", type=int, default=0, help="Número de repetición")
    args = parser.parse_args()

    df_cov, df_tests = load_tsm(args.dataset)
    eval_, elapsed = solve_ilp(df_cov, df_tests)
    path = _save(eval_, elapsed, args.dataset, args.repeat)

    print("[ILP TSM]")
    print(f"  Tests seleccionados: {eval_['selected']}")
    print(f"  Tiempo total: {eval_['total_time']:.4f}s | Factible: {eval_['feasible']}")
    print(f"  Tiempo de ejecucion: {elapsed:.4f}s")
    print(f"  Resultados en: {path}")


if __name__ == "__main__":
    main()