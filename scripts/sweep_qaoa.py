#!/usr/bin/env python3
"""Barrido de configuraciones QAOA en simulador (gratis, sin QPU).

Recorre reps x starts sobre los datasets indicados y produce una tabla
comparando contra el optimo ILP (que debe estar ya ejecutado).

Uso:
    python scripts/sweep_qaoa.py tsm --datasets toy,toy2,demo --reps 1,2,3 --starts 1,3,5
    python scripts/sweep_qaoa.py nrp --datasets toy,demo --reps 1,2,3 --starts 1,3,5
"""
import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

DEFAULT_BUDGETS = {"toy": 100, "toy2": 500, "demo": 60}


def _ilp_optimum(problem, dataset):
    p = ROOT / problem.upper() / "results" / "ilp" / dataset / "0" / "summary.csv"
    if not p.exists():
        return None
    row = pd.read_csv(p).iloc[0]
    if problem == "tsm":
        return row["objective"]
    return row["total_value"]


CONFIGS = {   # Fase 4 (solo TSM): formulacion, objetivo, inicializacion
    "base": dict(formulation="slack", objective="energy", init="random"),
    "F": dict(formulation="unbalanced", objective="energy", init="random"),
    "FC": dict(formulation="unbalanced", objective="cvar", init="random"),
    "FCI": dict(formulation="unbalanced", objective="cvar", init="grid"),
}


def run_tsm(dataset, reps, starts, maxiter, seed, shots, final_shots,
            config="base", alpha=0.25, grid=(36, 18)):
    sys.path.append(str(ROOT / "TSM"))
    from quantum.qaoa_tsm_simulator import run_qaoa

    eval_, meta = run_qaoa(
        dataset, reps=reps, starts=starts, maxiter=maxiter, seed=seed,
        shots=shots, final_shots=final_shots, method="statevector",
        alpha=alpha, grid=grid, **CONFIGS[config],
    )
    return eval_, meta


def run_nrp(dataset, reps, starts, maxiter, seed, shots, final_shots,
            config="base", alpha=0.25, grid=(36, 18)):
    sys.path.append(str(ROOT / "NRP"))
    from quantum.qaoa_nrp_simulator import run_qaoa

    eval_, meta = run_qaoa(
        dataset, DEFAULT_BUDGETS[dataset], reps=reps, starts=starts,
        maxiter=maxiter, seed=seed, shots=shots, final_shots=final_shots,
        method="statevector", alpha=alpha, grid=grid, **CONFIGS[config],
    )
    return eval_, meta


def main():
    ap = argparse.ArgumentParser(description="Barrido QAOA en simulador")
    ap.add_argument("problem", choices=["tsm", "nrp"])
    ap.add_argument("--datasets", type=str, default="toy,toy2,demo")
    ap.add_argument("--reps", type=str, default="1,2,3")
    ap.add_argument("--starts", type=str, default="1,3,5")
    ap.add_argument("--maxiter", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--final-shots", type=int, default=8000)
    # Fase 4 (solo tsm)
    ap.add_argument("--config", choices=list(CONFIGS), default="base")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--grid", default="36,18")
    ap.add_argument("--seeds", default=None, help="varias semillas, p. ej. 42,43,44 (anula --seed)")
    ap.add_argument("--out-dir", default=None, help="carpeta de salida (default <PROBLEMA>/results)")
    args = ap.parse_args()
    grid = tuple(int(x) for x in args.grid.split(","))
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    datasets = [d.strip() for d in args.datasets.split(",")]
    reps_list = [int(x) for x in args.reps.split(",")]
    starts_list = [int(x) for x in args.starts.split(",")]

    if args.out_dir:
        suffix = args.config + (f"_a{args.alpha:g}" if args.config in ("FC", "FCI") else "")
        out = Path(args.out_dir) / f"sweep_{args.problem}_{suffix}.csv"
    else:
        out = ROOT / f"{args.problem.upper()}" / "results" / f"sweep_{args.problem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["problem", "dataset", "config", "alpha", "seed", "reps", "starts", "metric",
                  "optimal_metric", "optimal", "feasible", "wall_s", "optimizer_fun",
                  "num_qubo_vars", "p_best", "p_feasible"]
    rows = []

    # Escribir incrementalmente (header + fila tras fila) para no perder progreso.
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        fh.flush()

        for dataset in datasets:
            opt = _ilp_optimum(args.problem, dataset)
            for reps, starts, seed in itertools.product(reps_list, starts_list, seeds):
                    t0 = time.time()
                    try:
                        if args.problem == "tsm":
                            eval_, meta = run_tsm(dataset, reps, starts, args.maxiter,
                                                  seed, args.shots, args.final_shots,
                                                  config=args.config, alpha=args.alpha, grid=grid)
                            metric = eval_["objective"]
                            feasible = eval_["feasible"]
                        else:
                            eval_, meta = run_nrp(dataset, reps, starts, args.maxiter,
                                                  seed, args.shots, args.final_shots,
                                                  config=args.config, alpha=args.alpha, grid=grid)
                            metric = eval_["total_value"]
                            feasible = eval_["feasible"]
                    except Exception as exc:  # noqa: BLE001
                        print(f"[FALLO] {args.problem} {dataset} reps={reps} starts={starts}: {exc}", flush=True)
                        continue

                    if opt is not None:
                        if args.problem == "tsm":
                            optimal = "yes" if metric <= opt + 1e-9 else "no"
                        else:
                            optimal = "yes" if metric >= opt - 1e-9 else "no"
                    else:
                        optimal = "unknown"

                    row = {
                        "problem": args.problem, "dataset": dataset, "config": args.config,
                        "alpha": args.alpha if args.config in ("FC", "FCI") else "",
                        "seed": seed, "reps": reps,
                        "starts": starts, "metric": round(float(metric), 3),
                        "optimal_metric": round(float(opt), 3) if opt is not None else "",
                        "optimal": optimal, "feasible": feasible,
                        "wall_s": round(time.time() - t0, 1),
                        "optimizer_fun": round(float(meta.get("optimizer_fun") or 0), 1),
                        "num_qubo_vars": meta.get("num_qubo_vars"),
                        # Masa de probabilidad de la solucion devuelta / factible
                        "p_best": round(float(meta.get("p_best", 0.0)), 6),
                        "p_feasible": round(float(meta.get("p_feasible", 0.0)), 4),
                    }
                    rows.append(row)
                    w.writerow(row)
                    fh.flush()
                    tag = "OPT" if optimal == "yes" else ("fact" if feasible else "INFEAS")
                    print(f"[{tag:>6}] {args.problem} {dataset:5} reps={reps} starts={starts} "
                          f"metric={metric:.3f} opt={opt} t={time.time()-t0:.0f}s", flush=True)

    print(f"\nTabla escrita en: {out}")


if __name__ == "__main__":
    main()
