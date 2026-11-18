#!/usr/bin/env python3
"""QAOA local para TSM sobre AerSimulator.

Resuelve el Test Suite Minimization traduciendo el QuadraticProgram a QUBO
(penalizando las restricciones de cobertura) y ejecutando QAOA con un
simulador local de Qiskit (AerSimulator).

Uso:
    python qaoa_tsm_simulator.py toy --reps 2 --shots 4000 --maxiter 100 --seed 42
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
from data_io import evaluate_selection, is_feasible, load_tsm  # noqa: E402
from qaoa_engine import (  # noqa: E402
    AerBackend,
    build_ansatz,
    build_value_lookup,
    decode_full,
    evaluate_energy,
    extract_best,
    initial_params,
    prepare_circuit,
)
import tsm_qubo  # noqa: E402
from tsm_qubo import build_qp, build_qubo, build_qubo_unbalanced, qubo_value  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_qaoa(dataset, repeat=0, reps=1, shots=4000, final_shots=8000, maxiter=100,
             seed=42, method="statevector", penalty=None, starts=1,
             formulation="slack", lambda1=None, lambda2=None,
             objective="energy", alpha=0.25, init="random", init_from=None,
             grid=(36, 18)):
    """Ejecuta QAOA sobre un dataset TSM en AerSimulator.

    Los valores por defecto reproducen la configuración Base (experimentos E1 y E2):
    QUBO con holguras (`formulation="slack"`, penalización 10(n+1)), energía
    esperada como objetivo e inicialización aleatoria. Opciones de las
    configuraciones F, F+C y F+C+I:
      formulation="unbalanced" : penalización desbalanceada sin holguras
                                 (lambda1/lambda2; None = valores calibrados
                                 en tsm_qubo.DEFAULT_LAMBDA*).
      objective="cvar", alpha   : CVaR_alpha como objetivo de COBYLA.
      init="grid"               : el primer reinicio parte del mejor punto de
                                 la malla gamma x beta (p=1, extendido con
                                 INTERP si reps > 1); los demás, aleatorios.
      init="transfer", init_from: el primer reinicio parte de los parámetros
                                 guardados en otro metadata.json (p. ej. de
                                 una instancia menor); INTERP si hacen falta
                                 más capas.
    """
    df_coverage, df_tests = load_tsm(dataset)
    original_names = df_coverage.index.tolist()

    qp, _ = build_qp(df_coverage, df_tests)
    if formulation == "slack":
        # Penalización por defecto: debe dominar el rango del objetivo (n_tests).
        if penalty is None:
            penalty = 10.0 * (len(original_names) + 1)
        qubo = build_qubo(qp, penalty=penalty)
        lambda1 = lambda2 = None
    elif formulation == "unbalanced":
        lambda1 = tsm_qubo.DEFAULT_LAMBDA1 if lambda1 is None else lambda1
        lambda2 = tsm_qubo.DEFAULT_LAMBDA2 if lambda2 is None else lambda2
        if lambda1 is None or lambda2 is None:
            raise ValueError("formulation='unbalanced' requiere lambda1 y lambda2 "
                             "(o DEFAULT_LAMBDA* calibrados en tsm_qubo.py)")
        qubo = build_qubo_unbalanced(qp, lambda1, lambda2)
        penalty = None
    else:
        raise ValueError(f"formulation desconocida: {formulation}")
    operator, offset = qubo.to_ising()
    n_qubits = operator.num_qubits

    # Expectativa exacta (statevector) si el método lo permite; si no, muestreo.
    exact = method == "statevector"
    if n_qubits > 24 and exact:
        raise ValueError(
            f"Metodo 'statevector' no es viable con {n_qubits} qubits. "
            f"Usa --method matrix_product_state o qasm (muestreo), o un dataset menor."
        )

    ansatz = build_ansatz(operator, reps)
    backend = AerBackend(method=method, seed=seed)
    circuit = prepare_circuit(backend, ansatz)
    value_lookup = build_value_lookup(qubo) if exact else None
    value_order = (np.argsort(value_lookup, kind="stable")
                   if (exact and objective == "cvar") else None)

    def make_energy_fn(circ):
        def energy_fn(p):
            return evaluate_energy(backend, circ, p, qubo, qubo_value, shots,
                                   operator=operator, offset=offset, exact=exact,
                                   value_lookup=value_lookup, objective=objective,
                                   alpha=alpha, value_order=value_order)
        return energy_fn

    energy_fn = make_energy_fn(circuit)

    start = time.perf_counter()
    # Punto inicial informado (solo para el primer reinicio)
    warm, grid_info = None, None
    if init == "grid":
        circuit1 = circuit if reps == 1 else prepare_circuit(backend, build_ansatz(operator, 1))
        t0 = time.perf_counter()
        warm = initial_params(1, None, init="grid", objective_fn=make_energy_fn(circuit1), grid=grid)
        grid_info = {"grid": list(grid), "params": warm.tolist(),
                     "value": float(make_energy_fn(circuit1)(warm)),
                     "secs": time.perf_counter() - t0}
    elif init == "transfer":
        if init_from is None:
            raise ValueError("init='transfer' requiere init_from (metadata.json)")
        with open(init_from) as fh:
            warm = np.asarray(json.load(fh)["qaoa_params"], dtype=np.float64)
    elif init != "random":
        raise ValueError(f"init desconocido: {init}")

    best_result = None
    best_trajectory = None
    for s in range(starts):
        rng = np.random.default_rng(seed + s)
        if s == 0 and warm is not None:
            x0 = initial_params(reps, rng, init="transfer", warm=warm)
        else:
            x0 = initial_params(reps, rng)
        traj = []

        def cobyla_objective(p):
            e = energy_fn(p)
            traj.append(float(e))
            return e

        result = minimize(cobyla_objective, x0, method="COBYLA",
                          options={"maxiter": maxiter})
        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_trajectory = traj
    result = best_result
    trajectory = best_trajectory
    elapsed = time.perf_counter() - start

    assignment, qubo_best, feasible, counts = extract_best(
        backend, circuit, result.x, qubo, original_names, qubo_value,
        lambda sel: is_feasible(sel, df_coverage), final_shots,
        objective=lambda sel: evaluate_selection(sel, df_coverage, df_tests)["objective"],
    )

    selected = [t for t in original_names if assignment.get(t, 0) == 1]
    eval_ = evaluate_selection(selected, df_coverage, df_tests)

    # Comparación contra el óptimo exacto (ILP) si ya fue ejecutado
    optimal = "unknown"
    ilp_path = RESULTS_DIR / "ilp" / dataset / str(repeat) / "summary.csv"
    if ilp_path.exists():
        ilp = pd.read_csv(ilp_path).iloc[0]
        optimal = "yes" if eval_["objective"] <= ilp["objective"] + 1e-9 else "no"

    # Diagnóstico: que fracción de las muestras finales es la solución devuelta
    # y que fracción es factible. Sirve para distinguir una distribución QAOA
    # concentrada de un simple filtrado clásico sobre muestras ~uniformes.
    n_shots = sum(counts.values())
    p_best = p_feasible = 0.0
    for bitstring, count in counts.items():
        full = decode_full(bitstring, qubo)
        orig = {name: full[name] for name in original_names}
        sel_i = [name for name in original_names if orig[name] == 1]
        if is_feasible(sel_i, df_coverage):
            p_feasible += count / n_shots
            if orig == assignment:
                p_best += count / n_shots

    metadata = {
        "dataset": dataset,
        "repeat": repeat,
        "reps": reps,
        "shots": shots,
        "final_shots": final_shots,
        "maxiter": maxiter,
        "starts": starts,
        "seed": seed,
        "method": method,
        "penalty": penalty,
        "formulation": formulation,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "objective": objective,
        "alpha": alpha if objective == "cvar" else None,
        "init": init,
        "init_from": str(init_from) if init_from else None,
        "grid_init": grid_info,
        "num_decision_vars": len(original_names),
        "num_qubo_vars": qubo.get_num_vars(),
        "optimizer": "COBYLA",
        "optimizer_success": bool(result.success),
        "optimizer_fun": float(result.fun),
        "best_qubo_value": float(qubo_best),
        "optimal": optimal,
        "execution_time": elapsed,
        "samples_measured": sum(counts.values()),
        "qaoa_params": result.x.tolist(),
        "trajectory": trajectory,
        "p_best": p_best,
        "p_feasible": p_feasible,
    }
    return eval_, metadata


def _save(eval_, metadata, dataset, repeat, method="qaoa_sim"):
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
        "execution_time": metadata["execution_time"],
        "optimal": metadata.get("optimal", "unknown"),
    }
    pd.DataFrame([summary]).to_csv(path / "summary.csv", index=False)

    meta_save = {k: v for k, v in metadata.items()}
    meta_save["selected"] = eval_["selected"]
    with open(path / "metadata.json", "w") as fh:
        json.dump(meta_save, fh, indent=2, default=str)
    return path


def main():
    parser = argparse.ArgumentParser(description="QAOA local (AerSimulator) para TSM")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--reps", type=int, default=1, help="Capas QAOA (p)")
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--final-shots", type=int, default=8000)
    parser.add_argument("--maxiter", type=int, default=100, help="Iteraciones COBYLA")
    parser.add_argument("--starts", type=int, default=1, help="Reinicios aleatorios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="statevector",
                        choices=["statevector", "matrix_product_state", "automatic"])
    parser.add_argument("--penalty", type=float, default=None,
                        help="Peso de penalización QUBO (None = automático)")
    # Opciones de las configuraciones F, F+C y F+C+I
    parser.add_argument("--formulation", choices=["slack", "unbalanced"], default="slack",
                        help="slack = QUBO con holguras (Base); unbalanced = sin holguras")
    parser.add_argument("--lambda1", type=float, default=None)
    parser.add_argument("--lambda2", type=float, default=None)
    parser.add_argument("--objective", choices=["energy", "cvar"], default="energy")
    parser.add_argument("--alpha", type=float, default=0.25, help="alpha de CVaR")
    parser.add_argument("--init", choices=["random", "grid", "transfer"], default="random")
    parser.add_argument("--init-from", type=str, default=None,
                        help="metadata.json con qaoa_params (init=transfer)")
    parser.add_argument("--grid", type=str, default="36,18", help="malla gamma,beta (init=grid)")
    parser.add_argument("--tag", type=str, default=None,
                        help="sufijo de la carpeta de resultados (results/qaoa_sim_<tag>)")
    args = parser.parse_args()

    eval_, metadata = run_qaoa(
        args.dataset, repeat=args.repeat, reps=args.reps, shots=args.shots,
        final_shots=args.final_shots, maxiter=args.maxiter, seed=args.seed,
        method=args.method, penalty=args.penalty, starts=args.starts,
        formulation=args.formulation, lambda1=args.lambda1, lambda2=args.lambda2,
        objective=args.objective, alpha=args.alpha, init=args.init,
        init_from=args.init_from, grid=tuple(int(x) for x in args.grid.split(",")),
    )
    method_tag = "qaoa_sim" + (f"_{args.tag}" if args.tag else "")
    path = _save(eval_, metadata, args.dataset, args.repeat, method=method_tag)

    print("[QAOA SIM TSM]")
    print(f"  Tests seleccionados: {eval_['selected']}")
    print(f"  Tiempo total: {eval_['total_time']:.4f}s | Factible: {eval_['feasible']}")
    print(f"  Variables QUBO: {metadata['num_qubo_vars']} | Valor QUBO: {metadata['best_qubo_value']:.4f}")
    print(f"  Tiempo de ejecucion: {metadata['execution_time']:.4f}s")
    print(f"  Resultados en: {path}")


if __name__ == "__main__":
    main()