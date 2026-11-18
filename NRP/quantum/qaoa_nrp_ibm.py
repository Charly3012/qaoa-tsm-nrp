#!/usr/bin/env python3
"""QAOA para NRP sobre hardware real de IBM Quantum (Qiskit Runtime Service).

Las credenciales se leen desde el archivo .env ubicado en la raíz del
proyecto (no se incluye en el repositorio). Defina allí IBM_QUANTUM_TOKEN y, opcionalmente,
IBM_QUANTUM_INSTANCE, IBM_QUANTUM_CHANNEL e IBM_QUANTUM_BACKEND.

Uso:
    python qaoa_nrp_ibm.py toy --budget 100 --reps 1 --shots 4000
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
from data_io import evaluate_selection, is_feasible, load_nrp  # noqa: E402
from nrp_qubo import build_qp, build_qubo, qubo_value  # noqa: E402
from qaoa_engine import build_ansatz, decode_full, evaluate_energy, extract_best, initial_params  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _job_quantum_seconds(job, retries=12, wait=5.0, sleep=None):
    """Segundos de QPU facturados al job (misma lógica que qaoa_tsm_ibm.py).

    IBM llena job.metrics()['usage'] unos segundos después de job.result()
    (deuda técnica #8), así que se reintenta mientras el uso no esté completo.
    """
    sleep = sleep or time.sleep
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


def run_qaoa_ibm(dataset, budget, repeat=0, reps=1, shots=4000, final_shots=8000,
                 maxiter=0, seed=42, backend_name=None, instance=None,
                 params_from_sim=True, params_from_sim_repeat=0):
    backend_name = backend_name or (os.getenv("IBM_QUANTUM_BACKEND") or "").strip() or None
    df_features, df_deps = load_nrp(dataset)
    original_names = df_features.index.tolist()

    qp, _ = build_qp(df_features, df_deps, budget)
    qubo = build_qubo(qp, penalty=float(len(original_names) + 1.0))
    operator, _ = qubo.to_ising()
    n_qubits = operator.num_qubits

    # Óptimo exacto (ILP) para contar las muestras óptimas (p_opt muestreado).
    from classic.ilp_nrp import solve_ilp
    ilp_ev, _ = solve_ilp(df_features, df_deps, budget)
    opt_value = float(ilp_ev["total_value"])

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
        lambda sel: is_feasible(sel, df_features, df_deps, budget), final_shots,
        objective=lambda sel: evaluate_selection(sel, df_features, df_deps, budget)["objective"],
    )

    selected = [f for f in original_names if assignment.get(f, 0) == 1]
    eval_ = evaluate_selection(selected, df_features, df_deps, budget)

    # Diagnóstico (como en el simulador): masa de la solución devuelta, masa
    # factible y masa en cualquier asignación óptima (comparable con el p_opt exacto).
    n_shots = sum(counts.values())
    p_best = p_feasible = p_opt_sampled = 0.0
    for bitstring, count in counts.items():
        full = decode_full(bitstring, qubo)
        orig = {name: full[name] for name in original_names}
        sel_i = [f for f in original_names if orig[f] == 1]
        if is_feasible(sel_i, df_features, df_deps, budget):
            p_feasible += count / n_shots
            if orig == assignment:
                p_best += count / n_shots
            if abs(float(df_features.loc[sel_i, "Valor"].sum()) - opt_value) < 1e-9:
                p_opt_sampled += count / n_shots

    metadata = {
        "dataset": dataset,
        "repeat": repeat,
        "budget": budget,
        "reps": reps,
        "shots": shots,
        "final_shots": final_shots,
        "maxiter": maxiter,
        "seed": seed,
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
        "optimum_value": opt_value,
        "optimal": "yes" if (eval_["feasible"] and abs(float(eval_["total_value"]) - opt_value) < 1e-9) else "no",
        "p_opt_sampled": p_opt_sampled,
        "params_source": params_source,
        "p_best": p_best,
        "p_feasible": p_feasible,
        "isa_depth": rtb.last_depth,
        "isa_2q_gates": rtb.last_2q,
        "transpile_optimization_level": 3,
        "dynamical_decoupling": True,
    }
    return eval_, metadata


def _save(eval_, metadata, dataset, repeat, method="qaoa_ibm"):
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
        "budget": metadata["budget"],
        "selected_case_num": eval_["selected_case_num"],
        "total_cost": eval_["total_cost"],
        "total_value": eval_["total_value"],
        "budget_used": eval_["budget_used"],
        "feasible": eval_["feasible"],
        "objective": eval_["objective"],
        "execution_time": metadata["execution_time"],
        "optimal": "unknown",
    }
    pd.DataFrame([summary]).to_csv(path / "summary.csv", index=False)

    meta_save = {k: v for k, v in metadata.items()}
    meta_save["selected"] = eval_["selected"]
    with open(path / "metadata.json", "w") as fh:
        json.dump(meta_save, fh, indent=2, default=str)
    return path


def main():
    parser = argparse.ArgumentParser(description="QAOA en hardware IBM Quantum para NRP")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--final-shots", type=int, default=8000)
    parser.add_argument("--maxiter", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--instance", type=str, default=None)
    parser.add_argument("--random-params", action="store_true",
                        help="Usar parámetros aleatorios en vez de los optimizados en simulador")
    parser.add_argument("--sim-repeat", type=int, default=0,
                        help="Repetición del simulador de la que tomar qaoa_params")
    args = parser.parse_args()

    try:
        _run(args)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


def _run(args):
    eval_, metadata = run_qaoa_ibm(
        args.dataset, args.budget, repeat=args.repeat, reps=args.reps,
        shots=args.shots, final_shots=args.final_shots, maxiter=args.maxiter,
        seed=args.seed, backend_name=args.backend, instance=args.instance,
        params_from_sim=not args.random_params, params_from_sim_repeat=args.sim_repeat,
    )
    path = _save(eval_, metadata, args.dataset, args.repeat)

    print("[QAOA IBM NRP]")
    print(f"  Backend: {metadata['backend']}")
    print(f"  Features seleccionadas: {eval_['selected']}")
    print(f"  Costo: {eval_['total_cost']} | Valor: {eval_['total_value']} | Factible: {eval_['feasible']}")
    print(f"  p_best: {metadata['p_best']:.5f} | p_feasible: {metadata['p_feasible']:.4f} | "
          f"profundidad ISA: {metadata['isa_depth']} | puertas 2q: {metadata['isa_2q_gates']}")
    print(f"  Jobs: {metadata['num_jobs']} | segundos QPU: {metadata['total_quantum_seconds']}")
    print(f"  Resultados en: {path}")


if __name__ == "__main__":
    main()