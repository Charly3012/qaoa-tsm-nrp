#!/usr/bin/env python3
"""Recocido simulado (Simulated Annealing) para TSM.

Energía = n_tests + weight*tiempo + penalización por
requisitos no cubiertos. Estado inicial = todos los tests seleccionados
(garantiza factibilidad inicial). Criterio de Metropolis con enfriamiento
geométrico.
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_io import TIME_WEIGHT, evaluate_selection, load_tsm  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _energia(estado, matriz, tiempos, peso_tiempo=TIME_WEIGHT, peso_penalizacion=1000.0):
    """Energía: n_tests + peso*tiempo + penalización por requisitos no cubiertos.

    Cálculo vectorizado.
    """
    seleccion = np.array([estado[t] for t in tiempos.index], dtype=int)
    if seleccion.sum() == 0:
        n_tests, tiempo_total, n_no_cubiertos = 0, 0.0, matriz.shape[1]
    else:
        n_tests = int(seleccion.sum())
        tiempo_total = float(np.sum(tiempos.values * seleccion))
        cubiertos = matriz[seleccion == 1].any(axis=0)
        n_no_cubiertos = int((~cubiertos).sum())

    energia = n_tests + peso_tiempo * tiempo_total
    penalizacion = peso_penalizacion * n_no_cubiertos
    return energia + penalizacion, penalizacion == 0


def solve_sa(df_coverage, df_tests, seed=None, temp_inicial=100.0, tasa=0.995,
             iteraciones=100, temp_final=0.1):
    tests_lista = df_coverage.index.tolist()
    matriz = df_coverage.values.astype(int)
    tiempos = df_tests["time"].to_numpy(dtype=float)
    tiempos_series = pd.Series(tiempos, index=df_tests.index)
    rng = random.Random(seed)

    estado = {t: 1 for t in tests_lista}
    energia, _ = _energia(estado, matriz, tiempos_series)
    mejor_estado, mejor_energia = dict(estado), energia

    temp = temp_inicial
    start = time.perf_counter()
    while temp > temp_final:
        for _ in range(iteraciones):
            t_rand = rng.choice(tests_lista)
            vecino = dict(estado)
            vecino[t_rand] = 1 - vecino[t_rand]

            energia_vecino, es_valido = _energia(vecino, matriz, tiempos_series)
            delta = energia_vecino - energia
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                estado, energia = vecino, energia_vecino
                if es_valido and energia < mejor_energia:
                    mejor_energia = energia
                    mejor_estado = dict(estado)
        temp *= tasa
    elapsed = time.perf_counter() - start

    selected = [t for t in tests_lista if mejor_estado[t] == 1]
    eval_ = evaluate_selection(selected, df_coverage, df_tests)
    return eval_, elapsed


def _save(eval_, elapsed, dataset, repeat, method="sa"):
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
    parser = argparse.ArgumentParser(description="Recocido simulado para TSM")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    df_cov, df_tests = load_tsm(args.dataset)
    eval_, elapsed = solve_sa(df_cov, df_tests, seed=args.seed)
    path = _save(eval_, elapsed, args.dataset, args.repeat)

    print("[SA TSM]")
    print(f"  Tests seleccionados: {eval_['selected']}")
    print(f"  Tiempo total: {eval_['total_time']:.4f}s | Factible: {eval_['feasible']}")
    print(f"  Tiempo de ejecucion: {elapsed:.4f}s")
    print(f"  Resultados en: {path}")


if __name__ == "__main__":
    main()