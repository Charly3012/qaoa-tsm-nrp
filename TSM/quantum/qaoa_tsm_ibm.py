#!/usr/bin/env python3
"""QAOA para TSM sobre hardware real de IBM Quantum (Qiskit Runtime Service).

Las credenciales se leen desde el archivo .env ubicado en la raíz del
proyecto (no se incluye en el repositorio). Defina allí IBM_QUANTUM_TOKEN y, opcionalmente,
IBM_QUANTUM_INSTANCE, IBM_QUANTUM_CHANNEL e IBM_QUANTUM_BACKEND.

Uso:
    python qaoa_tsm_ibm.py toy --reps 2 --shots 4000

Experimento E3: codificación sin holguras y parámetros transferidos de la
configuración F+C+I del simulador:
    python qaoa_tsm_ibm.py tsm_aleatoria_n16 --data-dir report/scale_n/data \
        --formulation unbalanced --params-from report/fase4/scale_tsm_aleatoria_FCI_transfer_a0.25.csv \
        --tag fase4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
from data_io import evaluate_selection, is_feasible, load_tsm  # noqa: E402
from qaoa_engine import build_ansatz, decode_full, evaluate_energy, extract_best, initial_params  # noqa: E402
import tsm_qubo  # noqa: E402
from tsm_qubo import build_qp, build_qubo, build_qubo_unbalanced, qubo_value  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _job_quantum_seconds(job, retries=12, wait=5.0, sleep=None):
    """Segundos de QPU facturados al job (job.metrics()['usage']['quantum_seconds']).

    `job.usage()` devolvía 0.0 en qiskit-ibm-runtime 0.39; `metrics()` sí expone
    el campo, pero IBM lo llena unos segundos DESPUÉS de que `job.result()`
    regresa (deuda técnica #8: los 7 jobs de la Fase 4 quedaron en 0.0). Por eso
    se reintenta hasta `retries` veces, esperando `wait` segundos, mientras el
    uso no esté completo (`usage.status != 'complete'` o quantum_seconds en 0).
    Devuelve None si no está disponible.
    """
    import time as _time
    sleep = sleep or _time.sleep
    last = None
    for attempt in range(max(1, retries)):
        try:
            usage = (job.metrics() or {}).get("usage", {}) or {}
            qs = usage.get("quantum_seconds")
            status = usage.get("status")
            if qs is not None:
                last = float(qs)
                if last > 0 and status in (None, "complete", "completed"):
                    return last
        except Exception:
            pass
        if attempt < retries - 1:
            sleep(wait)
    if last is not None:
        return last
    try:
        u = job.usage()
        if isinstance(u, dict):
            return float(u.get("quantum_seconds", 0.0))
        return float(getattr(u, "quantum_seconds", 0.0))
    except Exception:
        return None


def load_transferred_params(source, reps, seed=42):
    """Parámetros [gamma_0, beta_0, ...] desde un metadata.json del simulador o
    desde un CSV de scripts/scale_n.py (fila method=qaoa de la semilla dada)."""
    source = Path(source)
    if source.suffix == ".json":
        with open(source) as fh:
            meta = json.load(fh)
        params = np.asarray(meta["qaoa_params"], dtype=float)
    else:
        df = pd.read_csv(source)
        # La columna seed mezcla enteros, "" y "grid": comparar numéricamente.
        seeds = pd.to_numeric(df.seed, errors="coerce")
        rows = df[(df.method == "qaoa") & (seeds == float(seed))]
        if rows.empty:
            raise ValueError(f"{source}: no hay fila method=qaoa con seed={seed}")
        return rows, None
    if len(params) != 2 * reps:
        raise ValueError(f"{source}: {len(params) // 2} capas, se piden {reps}")
    return None, params


class RuntimeBackend:
    """Adaptador del SamplerV2 de Qiskit Runtime a la interfaz sample()."""

    def __init__(self, backend, default_shots=4000):
        from qiskit_ibm_runtime import SamplerV2

        self.backend = backend
        self.sampler = SamplerV2(mode=backend, options={"default_shots": default_shots})
        # Mitigación básica: dynamical decoupling en los tiempos muertos del
        # circuito (no cambia el número de puertas de 2 qubits).
        self.sampler.options.dynamical_decoupling.enable = True
        self.num_jobs = 0
        self.job_ids = []
        self.total_quantum_seconds = 0.0
        self.last_depth = None
        self.last_2q = None

    def sample(self, circuit, shots):
        from qiskit import transpile

        # Nivel 3 reduce ~25% la profundidad frente al nivel 1 en heavy-hex.
        # Se registra profundidad y puertas de 2 qubits.
        isa_circuit = transpile(circuit, backend=self.backend, optimization_level=3,
                                seed_transpiler=42)
        ops = isa_circuit.count_ops()
        self.last_depth = isa_circuit.depth()
        self.last_2q = int(sum(c for g, c in ops.items() if g in ("cz", "cx", "ecr")))
        job = self.sampler.run([(isa_circuit, None, shots)])
        self.job_ids.append(job.job_id())
        print(f"[IBM] job {job.job_id()} enviado ({shots} muestras, {self.last_2q} puertas 2q, "
              f"profundidad {self.last_depth}); esperando resultado...", flush=True)
        result = job.result()
        self.num_jobs += 1
        qs = _job_quantum_seconds(job)
        if qs is not None:
            self.total_quantum_seconds += qs
        print(f"[IBM] job {job.job_id()} terminado: quantum_seconds={qs}", flush=True)
        return result[0].data.meas.get_counts()


def _get_service(instance=None):
    token = (os.getenv("IBM_QUANTUM_TOKEN") or "").strip()
    if not token:
        raise RuntimeError(
            "No se configuro el API token de IBM Quantum.\n"
            f"1) Defina IBM_QUANTUM_TOKEN en {PROJECT_ROOT / '.env'}, o\n"
            "2) exporte IBM_QUANTUM_TOKEN=su_token"
        )
    channel = (os.getenv("IBM_QUANTUM_CHANNEL") or "ibm_cloud").strip()
    instance = instance or (os.getenv("IBM_QUANTUM_INSTANCE") or "").strip() or None
    from qiskit_ibm_runtime import QiskitRuntimeService

    return QiskitRuntimeService(channel=channel, token=token, instance=instance)


def run_qaoa_ibm(dataset, repeat=0, reps=1, shots=4000, final_shots=8000,
                 maxiter=0, seed=42, backend_name=None, instance=None,
                 params_from_sim=True, params_from_sim_repeat=0,
                 formulation="slack", lambda1=None, lambda2=None,
                 params_from=None, params_seed=42, data_dir=None):
    """QAOA en hardware IBM. Por defecto reproduce la ejecución del experimento
    E1 (configuración Base: QUBO con holguras, penalización 10(n+1), parámetros de results/qaoa_sim).

    Experimento E3: formulation="unbalanced" (lambda1/lambda2 = DEFAULT_LAMBDA* si None)
    y params_from = metadata.json del simulador o CSV de scale_n.py (fila
    method=qaoa, seed=params_seed). data_dir permite cargar las instancias de
    escalamiento (report/scale_n/data).
    """
    backend_name = backend_name or (os.getenv("IBM_QUANTUM_BACKEND") or "").strip() or None
    df_coverage, df_tests = load_tsm(dataset, data_dir=data_dir)
    original_names = df_coverage.index.tolist()

    qp, _ = build_qp(df_coverage, df_tests)
    if formulation == "slack":
        # Misma penalización que qaoa_tsm_simulator.py: antes se usaba la
        # heurística automática de qiskit (otro QUBO), lo que impedía comparar
        # simulador y hardware sobre el mismo paisaje de energía.
        penalty = 10.0 * (len(original_names) + 1)
        qubo = build_qubo(qp, penalty=penalty)
        lambda1 = lambda2 = None
    elif formulation == "unbalanced":
        lambda1 = tsm_qubo.DEFAULT_LAMBDA1 if lambda1 is None else lambda1
        lambda2 = tsm_qubo.DEFAULT_LAMBDA2 if lambda2 is None else lambda2
        qubo = build_qubo_unbalanced(qp, lambda1, lambda2)
        penalty = None
    else:
        raise ValueError(f"formulation desconocida: {formulation}")
    operator, _ = qubo.to_ising()
    n_qubits = operator.num_qubits

    # Óptimo exacto (ILP) para etiquetar el resultado y contar muestras óptimas.
    from classic.ilp_tsm import solve_ilp
    ilp_ev, _ = solve_ilp(df_coverage, df_tests)
    opt_tests = int(ilp_ev["selected_case_num"])

    service = _get_service(instance=instance)
    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(
            min_num_qubits=n_qubits, operational=True, simulator=False
        )
    print(f"[IBM] Backend seleccionado: {backend.name}")

    rtb = RuntimeBackend(backend, default_shots=shots)
    ansatz = build_ansatz(operator, reps)
    rng = np.random.default_rng(seed)
    x0 = initial_params(reps, rng)
    params_source = "random"
    sim_ref = {}
    if params_from is not None:
        rows, params = load_transferred_params(params_from, reps, seed=params_seed)
        if rows is not None:
            row = rows[rows.instance == dataset] if "instance" in rows.columns else rows
            if row.empty:
                raise ValueError(f"{params_from}: no hay fila de {dataset} con seed={params_seed}")
            row = row.iloc[0]
            params = np.asarray([float(v) for v in str(row["params"]).split()], dtype=float)
            if len(params) != 2 * reps:
                raise ValueError(f"{params_from}: {len(params) // 2} capas en la fila, se piden {reps}")
            # Referencia del simulador para la misma instancia y parámetros.
            sim_ref = {k: (float(row[k]) if k in row and str(row[k]) != "" else None)
                       for k in ("p_opt", "p_feasible", "p_best", "metric", "optimum", "energy")}
            sim_ref["config"] = str(row.get("config", ""))
        x0 = params
        params_from_sim = False
        params_source = f"transfer:{Path(params_from).as_posix()}:seed{params_seed}"
    if params_from_sim:
        # Transferir los parámetros ya optimizados en simulador (mismo QUBO y
        # misma penalización) para no gastar jobs en optimizar en la QPU.
        sim_meta = RESULTS_DIR / "qaoa_sim" / dataset / str(params_from_sim_repeat) / "metadata.json"
        if sim_meta.exists():
            with open(sim_meta) as fh:
                sim = json.load(fh)
            if int(sim.get("reps", -1)) == reps and len(sim.get("qaoa_params", [])) == 2 * reps:
                x0 = np.asarray(sim["qaoa_params"], dtype=float)
                params_source = f"simulator:{sim_meta.relative_to(RESULTS_DIR.parent)}"
            else:
                print(f"[IBM] metadata del simulador con reps={sim.get('reps')} != {reps}; se usan parametros aleatorios")
        else:
            print(f"[IBM] no hay resultado de simulador en {sim_meta}; se usan parametros aleatorios")
    print(f"[IBM] Parametros iniciales ({params_source}): {np.round(x0, 4).tolist()}")

    start = time.perf_counter()
    trajectory = []
    if maxiter > 0:
        # Optimización clásica breve; CADA evaluación envía un job a IBM.
        def objective(p):
            e = evaluate_energy(rtb, ansatz, p, qubo, qubo_value, shots)
            trajectory.append(float(e))
            return e

        result = minimize(
            objective,
            x0,
            method="COBYLA",
            options={"maxiter": maxiter},
        )
        params = result.x
        optimizer_fun = float(result.fun)
    else:
        params = x0
        optimizer_fun = None
    elapsed = time.perf_counter() - start

    assignment, qubo_best, feasible, counts = extract_best(
        rtb, ansatz, params, qubo, original_names, qubo_value,
        lambda sel: is_feasible(sel, df_coverage), final_shots,
        objective=lambda sel: evaluate_selection(sel, df_coverage, df_tests)["objective"],
    )

    selected = [t for t in original_names if assignment.get(t, 0) == 1]
    eval_ = evaluate_selection(selected, df_coverage, df_tests)

    # Diagnóstico (como en el simulador): masa de la solución devuelta, masa
    # factible y masa en CUALQUIER asignación óptima (p_opt muestreado, comparable
    # con el p_opt exacto del simulador y con la línea base uniforme de scale_n).
    n_shots = sum(counts.values())
    p_best = p_feasible = p_opt_sampled = 0.0
    for bitstring, count in counts.items():
        full = decode_full(bitstring, qubo)
        orig = {name: full[name] for name in original_names}
        sel_i = [t for t in original_names if orig[t] == 1]
        if is_feasible(sel_i, df_coverage):
            p_feasible += count / n_shots
            if orig == assignment:
                p_best += count / n_shots
            if len(sel_i) == opt_tests:
                p_opt_sampled += count / n_shots

    metadata = {
        "dataset": dataset,
        "repeat": repeat,
        "reps": reps,
        "shots": shots,
        "final_shots": final_shots,
        "maxiter": maxiter,
        "seed": seed,
        "penalty": penalty,
        "formulation": formulation,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "data_dir": str(data_dir) if data_dir else None,
        "optimum_tests": opt_tests,
        "optimal": "yes" if (eval_["feasible"] and eval_["selected_case_num"] == opt_tests) else "no",
        "p_opt_sampled": p_opt_sampled,
        "sim_reference": sim_ref,
        "backend": backend.name,
        "instance": instance,
        "num_decision_vars": len(original_names),
        "num_qubo_vars": qubo.get_num_vars(),
        "optimizer": "COBYLA" if maxiter > 0 else "fixed-random",
        "optimizer_fun": optimizer_fun,
        "best_qubo_value": float(qubo_best),
        "execution_time": elapsed,
        "samples_measured": sum(counts.values()),
        "qaoa_params": np.asarray(params).tolist(),
        "trajectory": trajectory,
        "num_jobs": rtb.num_jobs,
        "job_ids": rtb.job_ids,
        "total_quantum_seconds": rtb.total_quantum_seconds,
        "params_source": params_source,
        "p_best": p_best,
        "p_feasible": p_feasible,
        "isa_depth": rtb.last_depth,
        "isa_2q_gates": rtb.last_2q,
        "transpile_optimization_level": 3,
        "dynamical_decoupling": True,
    }
    return eval_, metadata


def _save(eval_, metadata, dataset, repeat, method="qaoa_ibm", data_dir=None):
    path = RESULTS_DIR / method / dataset / str(repeat)
    path.mkdir(parents=True, exist_ok=True)

    df_cov, df_tests = load_tsm(dataset, data_dir=data_dir)
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
    parser = argparse.ArgumentParser(description="QAOA en hardware IBM Quantum para TSM")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--final-shots", type=int, default=8000)
    parser.add_argument("--maxiter", type=int, default=0,
                        help="Iteraciones COBYLA en hardware (0 = una sola muestra)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", type=str, default=None, help="Nombre del backend IBM")
    parser.add_argument("--instance", type=str, default=None,
                        help="Instancia IBM (ej. ibm-q/open/main)")
    parser.add_argument("--random-params", action="store_true",
                        help="Usar parámetros aleatorios en vez de los optimizados en simulador")
    parser.add_argument("--sim-repeat", type=int, default=0,
                        help="Repetición del simulador de la que tomar qaoa_params")
    # Opciones del experimento E3
    parser.add_argument("--formulation", choices=["slack", "unbalanced"], default="slack")
    parser.add_argument("--lambda1", type=float, default=None)
    parser.add_argument("--lambda2", type=float, default=None)
    parser.add_argument("--params-from", type=str, default=None,
                        help="metadata.json del simulador o CSV de scale_n.py con los parámetros a transferir")
    parser.add_argument("--params-seed", type=int, default=42,
                        help="semilla de la fila del CSV (scale_n) de la que tomar los parámetros")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="carpeta de instancias (p. ej. report/scale_n/data)")
    parser.add_argument("--tag", type=str, default=None,
                        help="sufijo de la carpeta de resultados (results/qaoa_ibm_<tag>)")
    args = parser.parse_args()

    try:
        _run(args)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


def _run(args):
    data_dir = Path(args.data_dir) if args.data_dir else None
    eval_, metadata = run_qaoa_ibm(
        args.dataset, repeat=args.repeat, reps=args.reps, shots=args.shots,
        final_shots=args.final_shots, maxiter=args.maxiter, seed=args.seed,
        backend_name=args.backend, instance=args.instance,
        params_from_sim=not args.random_params, params_from_sim_repeat=args.sim_repeat,
        formulation=args.formulation, lambda1=args.lambda1, lambda2=args.lambda2,
        params_from=args.params_from, params_seed=args.params_seed, data_dir=data_dir,
    )
    method = "qaoa_ibm" + (f"_{args.tag}" if args.tag else "")
    path = _save(eval_, metadata, args.dataset, args.repeat, method=method, data_dir=data_dir)

    print("[QAOA IBM TSM]")
    print(f"  Backend: {metadata['backend']} | formulacion: {metadata['formulation']} | "
          f"parametros: {metadata['params_source']}")
    print(f"  Tests seleccionados: {eval_['selected']} (optimo ILP: {metadata['optimum_tests']} tests "
          f"-> {metadata['optimal']})")
    print(f"  Tiempo total: {eval_['total_time']:.4f}s | Factible: {eval_['feasible']}")
    print(f"  p_best: {metadata['p_best']:.5f} | p_opt muestreado: {metadata['p_opt_sampled']:.5f} | "
          f"p_feasible: {metadata['p_feasible']:.4f} | profundidad ISA: {metadata['isa_depth']} | "
          f"puertas 2q: {metadata['isa_2q_gates']}")
    if metadata.get("sim_reference"):
        s = metadata["sim_reference"]
        print(f"  referencia simulador ({s.get('config')}): p_opt={s.get('p_opt')} "
              f"p_feasible={s.get('p_feasible')} p_best={s.get('p_best')}")
    print(f"  Jobs: {metadata['num_jobs']} | segundos QPU: {metadata['total_quantum_seconds']}")
    print(f"  Resultados en: {path}")


if __name__ == "__main__":
    main()