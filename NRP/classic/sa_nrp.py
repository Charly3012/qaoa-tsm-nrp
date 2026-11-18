#!/usr/bin/env python3
"""Recocido simulado (Simulated Annealing) para NRP.

Energía = -valor + penalización por exceder presupuesto
y por violar dependencias REQUIRES / EXCLUDES. Perturbación de 1 bit, criterio
de Metropolis y enfriamiento geométrico.
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_io import evaluate_selection, load_nrp  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _energia(estado, df_features, dependencias, budget, peso_penalizacion=10000.0):
    costo = sum(float(df_features.loc[f, "Costo"]) * v for f, v in estado.items())
    valor = sum(float(df_features.loc[f, "Valor"]) * v for f, v in estado.items())

    energia_base = -valor
    penalizacion = 0.0

    if costo > budget:
        penalizacion += (costo - budget) * peso_penalizacion

    for orig, rel, dest in dependencias:
        xo, xd = estado.get(orig, 0), estado.get(dest, 0)
        if rel == "REQUIRES" and xo == 1 and xd == 0:
            penalizacion += peso_penalizacion
        elif rel == "EXCLUDES" and xo == 1 and xd == 1:
            penalizacion += peso_penalizacion

    return energia_base + penalizacion, penalizacion == 0


def solve_sa(df_features, df_deps, budget, seed=None, temp_inicial=1000.0,
             temp_final=0.1, tasa=0.99, iteraciones_por_temp=50):
    features = df_features.index.tolist()
    dependencias = [
        (row["Origen"], row["Relacion"], row["Destino"])
        for _, row in df_deps.iterrows()
    ]
    rng = random.Random(seed)

    estado = {f: rng.choice([0, 1]) for f in features}
    energia, _ = _energia(estado, df_features, dependencias, budget)

    mejor_estado, mejor_energia = None, float("inf")
    temp = temp_inicial

    start = time.perf_counter()
    while temp > temp_final:
        for _ in range(iteraciones_por_temp):
            vecino = dict(estado)
            f = rng.choice(features)
            vecino[f] = 1 - vecino[f]

            energia_vecina, es_valido = _energia(vecino, df_features, dependencias, budget)

            if es_valido and energia_vecina < mejor_energia:
                mejor_energia = energia_vecina
                mejor_estado = dict(vecino)

            delta = energia_vecina - energia
            if delta < 0:
                estado, energia = vecino, energia_vecina
            else:
                try:
                    p = math.exp(-delta / temp)
                except OverflowError:
                    p = 0.0
                if rng.random() < p:
                    estado, energia = vecino, energia_vecina
        temp *= tasa
    elapsed = time.perf_counter() - start

    if mejor_estado is None:
        raise RuntimeError("El SA no encontro ninguna solucion factible.")

    selected = [f for f in features if mejor_estado[f] == 1]
    eval_ = evaluate_selection(selected, df_features, df_deps, budget)
    return eval_, elapsed


def _save(eval_, elapsed, dataset, repeat, budget, method="sa"):
    path = RESULTS_DIR / method / dataset / str(repeat)
    path.mkdir(parents=True, exist_ok=True)

    df_features, _ = load_nrp(dataset)
    selected = eval_["selected"]
    result = df_features.loc[selected][["Nombre", "Costo", "Valor"]].copy()
    result.to_csv(path / "result.csv")

    summary = {
        "method": method,
        "dataset": dataset,
        "repeat": repeat,
        "budget": budget,
        "selected_case_num": eval_["selected_case_num"],
        "total_cost": eval_["total_cost"],
        "total_value": eval_["total_value"],
        "budget_used": eval_["budget_used"],
        "feasible": eval_["feasible"],
        "objective": eval_["objective"],
        "execution_time": elapsed,
        "optimal": "unknown",
        "status": "Heuristic",
    }
    pd.DataFrame([summary]).to_csv(path / "summary.csv", index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Recocido simulado para NRP")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    df_features, df_deps = load_nrp(args.dataset)
    eval_, elapsed = solve_sa(df_features, df_deps, args.budget, seed=args.seed)
    path = _save(eval_, elapsed, args.dataset, args.repeat, args.budget)

    print("[SA NRP]")
    print(f"  Features seleccionadas: {eval_['selected']}")
    print(f"  Costo: {eval_['total_cost']} | Valor: {eval_['total_value']} | Factible: {eval_['feasible']}")
    print(f"  Tiempo de ejecucion: {elapsed:.4f}s")
    print(f"  Resultados en: {path}")


if __name__ == "__main__":
    main()