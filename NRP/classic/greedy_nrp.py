#!/usr/bin/env python3
"""Greedy para NRP (Next Release Problem).

Ordena las features por ratio Valor/Costo y las instala si caben en el
presupuesto, arrastrando las dependencias REQUIRES y bloqueando las EXCLUDES.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_io import evaluate_selection, load_nrp  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def solve_greedy(df_features, df_deps, budget):
    df = df_features.copy()
    df["Ratio"] = df["Valor"] / df["Costo"]
    df_sorted = df.sort_values(by=["Ratio", "Valor"], ascending=[False, False])

    requires, excludes = {}, {}
    for _, row in df_deps.iterrows():
        origen, relacion, destino = row["Origen"], row["Relacion"], row["Destino"]
        if relacion == "REQUIRES":
            requires.setdefault(origen, []).append(destino)
        elif relacion == "EXCLUDES":
            excludes.setdefault(origen, []).append(destino)
            excludes.setdefault(destino, []).append(origen)

    presupuesto_restante = budget
    seleccionadas, prohibidas = set(), set()

    start = time.perf_counter()
    for feature, row in df_sorted.iterrows():
        if presupuesto_restante <= 0:
            break
        if feature in prohibidas or feature in seleccionadas:
            continue

        costo_necesario = float(row["Costo"])
        paquete = {feature}
        es_valido = True

        if feature in requires:
            for dep in requires[feature]:
                if dep not in seleccionadas:
                    if dep in prohibidas:
                        es_valido = False
                        break
                    costo_necesario += float(df_features.loc[dep, "Costo"])
                    paquete.add(dep)

        if es_valido and costo_necesario <= presupuesto_restante:
            seleccionadas.update(paquete)
            presupuesto_restante -= costo_necesario
            for item in paquete:
                if item in excludes:
                    prohibidas.update(excludes[item])
    elapsed = time.perf_counter() - start

    eval_ = evaluate_selection(list(seleccionadas), df_features, df_deps, budget)
    return eval_, elapsed


def _save(eval_, elapsed, dataset, repeat, budget, method="greedy"):
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
        "status": "",
    }
    pd.DataFrame([summary]).to_csv(path / "summary.csv", index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Greedy para NRP")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--budget", type=int, default=100)
    args = parser.parse_args()

    df_features, df_deps = load_nrp(args.dataset)
    eval_, elapsed = solve_greedy(df_features, df_deps, args.budget)
    path = _save(eval_, elapsed, args.dataset, args.repeat, args.budget)

    print("[GREEDY NRP]")
    print(f"  Features seleccionadas: {eval_['selected']}")
    print(f"  Costo: {eval_['total_cost']} | Valor: {eval_['total_value']} | Factible: {eval_['feasible']}")
    print(f"  Tiempo de ejecucion: {elapsed:.4f}s")
    print(f"  Resultados en: {path}")


if __name__ == "__main__":
    main()