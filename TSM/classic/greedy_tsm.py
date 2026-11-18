#!/usr/bin/env python3
"""Greedy para TSM (Test Suite Minimization).

Selecciona iterativamente el test que cubre más requisitos aún pendientes;
en caso de empate, el más rápido (menor time).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_io import evaluate_selection, load_tsm  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def solve_greedy(df_coverage, df_tests):
    tests = df_coverage.index.tolist()
    matriz = df_coverage.values.astype(int)
    n_reqs = matriz.shape[1]
    times = {t: float(df_tests.loc[t, "time"]) for t in tests}

    reqs_por_cubrir = set(range(n_reqs))
    tests_elegidos = []

    start = time.perf_counter()
    while reqs_por_cubrir:
        mejor_test, max_cubiertos = None, -1
        for t in tests:
            if t in tests_elegidos:
                continue
            nuevos = len(set(np.where(matriz[tests.index(t)] == 1)[0]) & reqs_por_cubrir)
            if nuevos > max_cubiertos:
                max_cubiertos, mejor_test = nuevos, t
            elif nuevos == max_cubiertos and max_cubiertos > 0:
                if times[t] < times[mejor_test]:
                    mejor_test = t
        if max_cubiertos == 0 or mejor_test is None:
            break
        tests_elegidos.append(mejor_test)
        reqs_por_cubrir -= set(np.where(matriz[tests.index(mejor_test)] == 1)[0])
    elapsed = time.perf_counter() - start

    eval_ = evaluate_selection(tests_elegidos, df_coverage, df_tests)
    return eval_, elapsed


def _save(eval_, elapsed, dataset, repeat, method="greedy"):
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
        "optimal": "unknown",
        "status": "",
    }
    pd.DataFrame([summary]).to_csv(path / "summary.csv", index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Greedy para TSM")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()

    df_cov, df_tests = load_tsm(args.dataset)
    eval_, elapsed = solve_greedy(df_cov, df_tests)
    path = _save(eval_, elapsed, args.dataset, args.repeat)

    print("[GREEDY TSM]")
    print(f"  Tests seleccionados: {eval_['selected']}")
    print(f"  Tiempo total: {eval_['total_time']:.4f}s | Factible: {eval_['feasible']}")
    print(f"  Tiempo de ejecucion: {elapsed:.4f}s")
    print(f"  Resultados en: {path}")


if __name__ == "__main__":
    main()