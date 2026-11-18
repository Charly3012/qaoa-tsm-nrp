#!/usr/bin/env python3
"""Replicas en hardware de la ronda confirmatoria (cap. 5 de la tesis).

Bloques:
  A  3 replicas nuevas de los 7 trabajos de la Fase 4 (TSM, sin holguras, F+C+I)
  B  3 replicas de las 6 instancias de referencia con la configuracion Base
     (con holguras), para comparar las dos codificaciones en el mismo procesador
  C  instancias nuevas del experimento confirmatorio; se definen con --jobs
     (CSV) una vez elegidas tras el simulador

Para cada trabajo se calcula ANTES de enviar nada la referencia EXACTA del
simulador con los mismos parametros (p_opt y p_feasible, por vector de estado),
que es contra lo que se compara el hardware en H5. Si la instancia no cabe en el
simulador (mas de MAX_SIM_QUBITS qubits, p. ej. NRP toy2 con 37) la referencia
queda vacia y NO se construyen las tablas de 2^N estados (deuda tecnica #10: con
las instancias grandes del bloque B el ensayo en seco llego a 12.9 GB de RAM).

Proteccion del presupuesto de QPU: el script suma los segundos que IBM reporta
por trabajo y se detiene antes de enviar uno nuevo si la ronda llego al tope
(--max-qpu, 240 s por defecto). Cada trabajo se escribe al CSV en cuanto termina,
asi que una ejecucion interrumpida se puede reanudar (--resume salta lo hecho).

Uso (desde codigo/):
    python scripts/hardware_replicas.py --blocks A,B --dry-run          # plan + referencias, sin IBM
    python scripts/hardware_replicas.py --blocks A --only toy2 --replicas 1   # tanda piloto
    python scripts/hardware_replicas.py --blocks A --resume             # tanda A completa
    python scripts/hardware_replicas.py --jobs report/confirmatorio/bloque_c.csv --resume
Salida: report/confirmatorio/hardware_replicas.csv (formato que lee analisis_estadistico.py)

Columnas del CSV de --jobs: block, problem (tsm|nrp), instance, data_dir (vacio =
datos del problema), budget (solo nrp), formulation (unbalanced|slack),
params_from (CSV de scale_n.py, metadata.json o vacio = simulador/aleatorio),
params_seed, label.
"""
import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCALE_DATA = ROOT / "report" / "scale_n" / "data"
FASE4 = ROOT / "report" / "fase4"
OUT = ROOT / "report" / "confirmatorio" / "hardware_replicas.csv"
# Referencia exacta solo hasta 24 qubits: cubre los bloques A (maximo 24) y C
# (instancias del lote, N <= 22) con memoria acotada (~1 GB en 24 qubits).
MAX_SIM_QUBITS = 24

FCI_CICLO = FASE4 / "scale_tsm_ciclo_FCI_transfer_a0.25.csv"
FCI_ALEA = FASE4 / "scale_tsm_aleatoria_FCI_transfer_a0.25.csv"
TOY2_FCI = ROOT / "TSM" / "results" / "qaoa_sim_fase4" / "toy2" / "0" / "metadata.json"

# (block, problem, instance, data_dir, budget, formulation, params_from, label)
BLOCK_A = [("A", "tsm", "toy2", None, None, "unbalanced", TOY2_FCI, "F+C+I")] + [
    ("A", "tsm", inst, SCALE_DATA, None, "unbalanced", src, "F+C+I")
    for inst, src in [("tsm_ciclo_n16", FCI_CICLO), ("tsm_ciclo_n20", FCI_CICLO), ("tsm_ciclo_n24", FCI_CICLO),
                      ("tsm_aleatoria_n16", FCI_ALEA), ("tsm_aleatoria_n18", FCI_ALEA),
                      ("tsm_aleatoria_n22", FCI_ALEA)]]
BLOCK_B = [("B", "tsm", ds, None, None, "slack", None, "Base") for ds in ("toy", "toy2", "demo")] + [
    ("B", "nrp", ds, None, b, "slack", None, "Base") for ds, b in (("toy", 100), ("toy2", 500), ("demo", 60))]

FIELDS = ["block", "label", "problem", "instance", "replica", "backend", "job_ids", "shots", "qubits",
          "isa_depth", "isa_2q_gates", "quantum_seconds", "optimal", "feasible", "p_best", "p_opt_sampled",
          "p_feasible", "sim_p_opt", "sim_p_feasible", "params", "params_source", "wall_s", "timestamp"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SCALE = _load(ROOT / "scripts" / "scale_n.py", "scale_n_hw")

# Los runners de TSM y NRP importan modulos con el MISMO nombre (data_io,
# classic.*, qaoa_engine, ...). En un solo proceso, el segundo runner recibiria
# los del primero desde sys.modules. Antes de usar un runner se purgan esos
# nombres y se pone su carpeta al frente de sys.path.
_SHARED = ("data_io", "classic", "qaoa_engine", "tsm_qubo", "nrp_qubo")
_RUNNERS = {}
_ACTIVE = [None]


def activate(problem):
    """Deja sys.modules/sys.path listos para el runner de `problem` y lo devuelve."""
    if _ACTIVE[0] != problem:
        for name in list(sys.modules):
            if name in _SHARED or name.startswith("classic."):
                del sys.modules[name]
        base = ROOT / problem.upper()
        for p in (str(base), str(base / "quantum")):
            while p in sys.path:
                sys.path.remove(p)
        sys.path[:0] = [str(base / "quantum"), str(base)]
        _ACTIVE[0] = problem
        _RUNNERS.pop(problem, None)
    if problem not in _RUNNERS:
        _RUNNERS[problem] = _load(ROOT / problem.upper() / "quantum" / f"qaoa_{problem}_ibm.py",
                                  f"qaoa_{problem}_ibm")
    return _RUNNERS[problem]


def job_key(job):
    """Identificador de la instancia en el CSV: dos codificaciones de una misma
    instancia (p. ej. toy2 con Base y con F+C+I) no deben mezclarse en H5."""
    return f"{job['problem']}:{job['instance']}[{job['label']}]"


def read_jobs_csv(path):
    jobs = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            jobs.append(dict(
                block=r["block"], problem=r["problem"], instance=r["instance"],
                data_dir=Path(r["data_dir"]) if r.get("data_dir") else None,
                budget=int(r["budget"]) if r.get("budget") else None,
                formulation=r.get("formulation") or "unbalanced",
                params_from=Path(r["params_from"]) if r.get("params_from") else None,
                params_seed=int(r.get("params_seed") or 42), label=r.get("label") or "F+C+I"))
    return jobs


def default_jobs(blocks):
    jobs = []
    for block in blocks:
        for b, problem, inst, ddir, budget, form, src, label in {"A": BLOCK_A, "B": BLOCK_B}[block]:
            jobs.append(dict(block=b, problem=problem, instance=inst, data_dir=ddir, budget=budget,
                             formulation=form, params_from=src, params_seed=42, label=label))
    return jobs


# ---------------------------------------------------------------------------
# Parametros y referencia exacta del simulador
# ---------------------------------------------------------------------------
def resolve_params(job, eng, reps=1):
    """Los mismos parametros que usara el runner de hardware."""
    src = job["params_from"]
    if src is not None and Path(src).suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(src)
        seeds = pd.to_numeric(df.seed, errors="coerce")
        rows = df[(df.method == "qaoa") & (seeds == float(job["params_seed"])) & (df.instance == job["instance"])]
        if rows.empty:
            raise ValueError(f"{src}: sin fila qaoa de {job['instance']} con seed={job['params_seed']}")
        return np.asarray([float(v) for v in str(rows.iloc[0]["params"]).split()]), f"transfer:{Path(src).name}"
    meta = Path(src) if src is not None else \
        ROOT / job["problem"].upper() / "results" / "qaoa_sim" / job["instance"] / "0" / "metadata.json"
    if meta.exists():
        with open(meta) as fh:
            return np.asarray(json.load(fh)["qaoa_params"], dtype=float), f"simulator:{meta.relative_to(ROOT)}"
    # Igual que el runner cuando no hay resultado de simulador: inicio aleatorio con semilla 42.
    return np.asarray(eng.initial_params(reps, np.random.default_rng(42)), dtype=float), "random(seed=42)"


def build_job_qubo(job, M):
    """QUBO del trabajo y una funcion que construye, solo si se pide, las tablas
    de factibilidad y optimalidad sobre los 2^N estados de decision."""
    dio, qm = M["data_io"], M["qubo"]
    kw = {"data_dir": job["data_dir"]} if job["data_dir"] is not None else {}
    if job["problem"] == "tsm":
        df_a, df_b = dio.load_tsm(job["instance"], **kw)
        qp, _ = qm.build_qp(df_a, df_b)
        n = len(df_a.index)
        qubo = qm.build_qubo(qp, penalty=10.0 * (n + 1)) if job["formulation"] == "slack" else \
            qm.build_qubo_unbalanced(qp, qm.DEFAULT_LAMBDA1, qm.DEFAULT_LAMBDA2)

        def tables():
            ilp_ev, _ = M["ilp"].solve_ilp(df_a, df_b)
            return SCALE.uniform_tsm(df_a, ilp_ev["selected_case_num"])
    else:
        df_a, df_b = dio.load_nrp(job["instance"], **kw)
        qp, _ = qm.build_qp(df_a, df_b, job["budget"])
        n = len(df_a.index)
        if job["formulation"] != "slack":
            raise ValueError("NRP en hardware solo con la configuracion Base (con holguras)")
        qubo = qm.build_qubo(qp, penalty=float(n + 1))

        def tables():
            ilp_ev, _ = M["ilp"].solve_ilp(df_a, df_b, job["budget"])
            return SCALE.uniform_nrp(df_a, df_b, job["budget"], ilp_ev["total_value"])
    return qubo, df_a.index.tolist(), tables


def sim_reference(job, params):
    """p_opt y p_feasible exactos del estado final con esos parametros (vector de estado)."""
    M = SCALE.load_problem(job["problem"])
    eng = M["engine"]
    qubo, names, tables = build_job_qubo(job, M)
    n_q = qubo.get_num_vars()
    if n_q > MAX_SIM_QUBITS:
        return None, None, n_q
    feas_dec, opt_dec = tables()
    operator, _ = qubo.to_ising()
    backend = eng.AerBackend(method="statevector", seed=42)
    circuit = eng.prepare_circuit(backend, eng.build_ansatz(operator, len(params) // 2))
    bound = eng.bind_params(circuit, params).copy()
    bound.save_statevector()
    sv = backend.backend.run(bound).result().get_statevector()
    probs = np.abs(np.asarray(sv.data)) ** 2
    del sv
    var_names = [v.name for v in qubo.variables]
    idx = np.arange(1 << n_q, dtype=np.uint32)
    key = np.zeros(idx.size, dtype=np.uint32)
    bit = np.empty_like(idx)
    for k, nm in enumerate(names):      # operaciones en sitio: sin temporales de 2^n
        np.right_shift(idx, var_names.index(nm), out=bit)
        np.bitwise_and(bit, 1, out=bit)
        np.left_shift(bit, k, out=bit)
        np.bitwise_or(key, bit, out=key)
    del idx, bit
    return float(probs[opt_dec[key]].sum()), float(probs[feas_dec[key]].sum()), n_q


# ---------------------------------------------------------------------------
def done_keys(path):
    if not path.exists():
        return set(), 0.0
    keys, spent = set(), 0.0
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            keys.add((r["instance"], int(r["replica"])))
            spent += float(r["quantum_seconds"] or 0.0)
    return keys, spent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--blocks", default="", help="bloques predefinidos, p. ej. A,B")
    ap.add_argument("--jobs", default=None, help="CSV con trabajos adicionales (bloque C)")
    ap.add_argument("--only", default=None, help="solo instancias cuyo nombre contenga estos textos (coma)")
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--max-qpu", type=float, default=240.0,
                    help="tope de segundos de QPU de la ronda (incluye lo ya escrito en el CSV)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--resume", action="store_true", help="salta (instancia, replica) ya presentes en el CSV")
    ap.add_argument("--dry-run", action="store_true", help="muestra el plan y calcula referencias; no usa IBM")
    args = ap.parse_args()

    jobs = default_jobs([b for b in args.blocks.split(",") if b])
    if args.jobs:
        jobs += read_jobs_csv(args.jobs)
    if args.only:
        pats = args.only.split(",")
        jobs = [j for j in jobs if any(p in j["instance"] for p in pats)]
    if not jobs:
        sys.exit("No hay trabajos: indique --blocks y/o --jobs")

    out = Path(args.out)
    done, spent = done_keys(out) if args.resume or args.dry_run else (set(), 0.0)

    print(f"Procesador: {args.backend} | {args.shots} mediciones por trabajo | replicas: {args.replicas}")
    print(f"QPU ya registrada en {out.name}: {spent:.1f} s | tope de la ronda: {args.max_qpu:.0f} s\n")
    plan = []
    for job in jobs:
        M = SCALE.load_problem(job["problem"])
        params, source = resolve_params(job, M["engine"])
        p_opt, p_feas, n_q = sim_reference(job, params)
        job.update(params=params, params_source=source, sim_p_opt=p_opt, sim_p_feasible=p_feas, qubits=n_q)
        pending = [r for r in range(1, args.replicas + 1) if (job_key(job), r) not in done]
        plan.append((job, pending))
        ref = (f"p_opt={p_opt:.3e}  p_feasible={p_feas:.3f}  esperados en {args.replicas * args.shots}: "
               f"{p_opt * args.replicas * args.shots:.1f}") if p_opt is not None else "sin referencia (no simulable)"
        print(f"  [{job['block']}] {job_key(job):38s} {n_q:3d} qubits | {source:42s} | {ref} | "
              f"pendientes: {pending}")
    n_pending = sum(len(p) for _, p in plan)
    print(f"\n{n_pending} trabajos pendientes; QPU estimada ~{3 * n_pending} s (3 s por trabajo en la Fase 4).")
    if args.dry_run:
        return
    if spent + 3 * n_pending > args.max_qpu:
        print(f"AVISO: lo pendiente rebasaria el tope ({spent:.0f} + ~{3 * n_pending} > {args.max_qpu:.0f} s); "
              f"el script se detendra al alcanzarlo.")

    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for job, pending in plan:
            for replica in pending:
                if spent >= args.max_qpu:
                    print(f"\nTOPE DE QPU alcanzado ({spent:.1f} s >= {args.max_qpu:.0f} s): no se envian mas trabajos.")
                    return
                t0 = time.time()
                print(f"\n=== [{job['block']}] {job_key(job)} replica {replica} ===", flush=True)
                if job["problem"] == "tsm":
                    tsm_ibm = activate("tsm")
                    src = job["params_from"]
                    ev, meta = tsm_ibm.run_qaoa_ibm(
                        job["instance"], reps=1, shots=args.shots, final_shots=args.shots, maxiter=0,
                        backend_name=args.backend, formulation=job["formulation"],
                        params_from=str(src) if src is not None else None, params_seed=job["params_seed"],
                        params_from_sim=src is None, data_dir=job["data_dir"])
                else:
                    nrp_ibm = activate("nrp")
                    ev, meta = nrp_ibm.run_qaoa_ibm(
                        job["instance"], job["budget"], reps=1, shots=args.shots, final_shots=args.shots,
                        maxiter=0, backend_name=args.backend, params_from_sim=True)
                used = np.asarray(meta["qaoa_params"], dtype=float)
                if not np.allclose(used, job["params"], atol=1e-6):
                    print(f"AVISO: el runner uso parametros {used} distintos de la referencia {job['params']}")
                qs = float(meta.get("total_quantum_seconds") or 0.0)
                spent += qs
                w.writerow(dict(
                    block=job["block"], label=job["label"], problem=job["problem"], instance=job_key(job),
                    replica=replica, backend=meta["backend"], job_ids=" ".join(meta.get("job_ids", [])),
                    shots=args.shots, qubits=meta["num_qubo_vars"], isa_depth=meta.get("isa_depth"),
                    isa_2q_gates=meta.get("isa_2q_gates"), quantum_seconds=qs, optimal=meta.get("optimal"),
                    feasible=ev.get("feasible"), p_best=meta.get("p_best"),
                    p_opt_sampled=meta.get("p_opt_sampled"), p_feasible=meta.get("p_feasible"),
                    sim_p_opt=job["sim_p_opt"], sim_p_feasible=job["sim_p_feasible"],
                    params=" ".join(f"{v:.6f}" for v in used), params_source=job["params_source"],
                    wall_s=round(time.time() - t0, 1), timestamp=time.strftime("%Y-%m-%d %H:%M:%S")))
                fh.flush()
                print(f"  -> p_feasible={meta.get('p_feasible'):.4f} p_opt={meta.get('p_opt_sampled'):.5f} | "
                      f"QPU {qs:.1f} s | acumulado {spent:.1f} s", flush=True)
    print(f"\nTerminado. QPU acumulada de la ronda: {spent:.1f} s. Tabla en {out}")


if __name__ == "__main__":
    main()
