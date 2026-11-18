#!/usr/bin/env python3
"""Benchmark de NRP: compara ILP, Greedy, SA y QAOA (AerSimulator).

Uso:
    python benchmark_nrp.py --datasets toy,toy2 --methods ilp,greedy,sa,qaoa_sim

Genera results/benchmark_nrp.csv (resumen consolidado) y
results/benchmark_nrp.json (incluye metadatos del QAOA).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from classic.greedy_nrp import _save as save_greedy, solve_greedy
from classic.ilp_nrp import _save as save_ilp, solve_ilp
from classic.sa_nrp import _save as save_sa, solve_sa
from data_io import load_nrp
from quantum.qaoa_nrp_simulator import _save as save_qaoa, run_qaoa

RESULTS_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_BUDGETS = {"toy": 100, "toy2": 500, "demo": 60}


def _run_ilp(dataset, repeat, budget):
    df_f, df_d = load_nrp(dataset)
    eval_, elapsed = solve_ilp(df_f, df_d, budget)
    path = save_ilp(eval_, elapsed, dataset, repeat, budget)
    return pd.read_csv(path / "summary.csv").iloc[0]


def _run_greedy(dataset, repeat, budget):
    df_f, df_d = load_nrp(dataset)
    eval_, elapsed = solve_greedy(df_f, df_d, budget)
    path = save_greedy(eval_, elapsed, dataset, repeat, budget)
    return pd.read_csv(path / "summary.csv").iloc[0]


def _run_sa(dataset, repeat, budget, seed):
    df_f, df_d = load_nrp(dataset)
    eval_, elapsed = solve_sa(df_f, df_d, budget, seed=seed)
    path = save_sa(eval_, elapsed, dataset, repeat, budget)
    return pd.read_csv(path / "summary.csv").iloc[0]


def _run_qaoa(dataset, repeat, budget, qaoa_opts):
    eval_, metadata = run_qaoa(dataset, budget, repeat=repeat, **qaoa_opts)
    path = save_qaoa(eval_, metadata, dataset, repeat)
    return pd.read_csv(path / "summary.csv").iloc[0]


def main():
    parser = argparse.ArgumentParser(description="Benchmark NRP")
    parser.add_argument("--datasets", type=str, default="toy,toy2")
    parser.add_argument("--methods", type=str, default="ilp,greedy,sa,qaoa_sim")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budgets", type=str, default="",
                        help="Presupuesto por dataset, ej. 'toy:100,toy2:500'")
    parser.add_argument("--qaoa-reps", type=int, default=1)
    parser.add_argument("--qaoa-shots", type=int, default=4000)
    parser.add_argument("--qaoa-final-shots", type=int, default=8000)
    parser.add_argument("--qaoa-maxiter", type=int, default=100)
    parser.add_argument("--qaoa-starts", type=int, default=1)
    parser.add_argument("--qaoa-method", type=str, default="statevector")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]

    budgets = {}
    for pair in [p for p in args.budgets.split(",") if p]:
        name, value = pair.split(":")
        budgets[name.strip()] = int(value)

    qaoa_opts = {
        "reps": args.qaoa_reps,
        "shots": args.qaoa_shots,
        "final_shots": args.qaoa_final_shots,
        "maxiter": args.qaoa_maxiter,
        "starts": args.qaoa_starts,
        "seed": args.seed,
        "method": args.qaoa_method,
    }

    rows = []
    meta_all = {}
    start_total = time.time()

    for dataset in datasets:
        budget = budgets.get(dataset, DEFAULT_BUDGETS.get(dataset, 100))
        print(f"\n=== Dataset: {dataset} (budget={budget}) ===")
        for method in methods:
            print(f"  [benchmark] {method}...", end=" ", flush=True)
            t0 = time.time()
            try:
                if method == "ilp":
                    row = _run_ilp(dataset, args.repeat, budget)
                elif method == "greedy":
                    row = _run_greedy(dataset, args.repeat, budget)
                elif method == "sa":
                    row = _run_sa(dataset, args.repeat, budget, args.seed)
                elif method == "qaoa_sim":
                    row = _run_qaoa(dataset, args.repeat, budget, qaoa_opts)
                else:
                    print(f"metodo desconocido '{method}'")
                    continue
                rows.append(row.to_dict())
                meta_path = RESULTS_DIR / method / dataset / str(args.repeat) / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as fh:
                        meta_all[f"{method}/{dataset}/{args.repeat}"] = json.load(fh)
                print(f"OK ({(time.time() - t0):.1f}s)")
            except Exception as exc:  # noqa: BLE001
                print(f"FALLIDO: {exc}")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "benchmark_nrp.csv", index=False)
    with open(RESULTS_DIR / "benchmark_nrp.json", "w") as fh:
        json.dump(
            {
                "summary": df.to_dict(orient="records"),
                "metadata": meta_all,
                "total_time": time.time() - start_total,
                "options": vars(args),
            },
            fh,
            indent=2,
            default=str,
        )

    print("\n" + "=" * 70)
    if df.empty:
        print("Sin resultados: ningun metodo se ejecuto correctamente.")
    else:
        cols = [c for c in ["method", "dataset", "selected_case_num", "total_cost",
                            "total_value", "feasible", "objective", "optimal"] if c in df.columns]
        print(df[cols].to_string(index=False))
    print(f"\nResumen en: {RESULTS_DIR / 'benchmark_nrp.csv'}")


if __name__ == "__main__":
    main()