#!/usr/bin/env python3
"""ILP exacto para NRP (Next Release Problem).

Maximiza el valor total de features seleccionadas sujeto al presupuesto y a
las dependencias lógicas (REQUIRES / EXCLUDES). Resuelto con PuLP/CBC.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pulp

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_io import evaluate_selection, load_nrp  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def solve_ilp(df_features, df_deps, budget):
    features = df_features.index.tolist()

    prob = pulp.LpProblem("Next_Release_Problem", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("feature", features, cat=pulp.LpBinary)

    prob += pulp.lpSum([df_features.loc[f, "Valor"] * x[f] for f in features]), "Maximizar_Valor"

    prob += (
        pulp.lpSum([df_features.loc[f, "Costo"] * x[f] for f in features]) <= budget,
        "Limite_Presupuesto",
    )

    for _, row in df_deps.iterrows():
        origen, relacion, destino = row["Origen"], row["Relacion"], row["Destino"]
        if origen not in features or destino not in features:
            continue
        if relacion == "REQUIRES":
            prob += x[origen] - x[destino] <= 0
        elif relacion == "EXCLUDES":
            prob += x[origen] + x[destino] <= 1

    start = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    elapsed = time.perf_counter() - start

    selected = [f for f in features if pulp.value(x[f]) == 1.0]
    eval_ = evaluate_selection(selected, df_features, df_deps, budget)
    eval_["status"] = pulp.LpStatus[prob.status]
    return eval_, elapsed


def _save(eval_, elapsed, dataset, repeat, budget, method="ilp"):
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
        "optimal": "yes",
        "status": eval_.get("status", ""),
    }
    pd.DataFrame([summary]).to_csv(path / "summary.csv", index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="ILP exacto para NRP")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--budget", type=int, default=100)
    args = parser.parse_args()

    df_features, df_deps = load_nrp(args.dataset)
    eval_, elapsed = solve_ilp(df_features, df_deps, args.budget)
    path = _save(eval_, elapsed, args.dataset, args.repeat, args.budget)

    print("[ILP NRP]")
    print(f"  Features seleccionadas: {eval_['selected']}")
    print(f"  Costo: {eval_['total_cost']} | Valor: {eval_['total_value']} | Factible: {eval_['feasible']}")
    print(f"  Tiempo de ejecucion: {elapsed:.4f}s")
    print(f"  Resultados en: {path}")


if __name__ == "__main__":
    main()