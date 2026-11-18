#!/usr/bin/env python3
"""Fase 4 en hardware IBM (`ibm_fez`): 7 jobs aprobados el 2026-10-30.

Cada job: QUBO sin holguras (lambda1 = lambda2 = 1), p = 1, parametros
transferidos de la configuracion F+C+I del simulador, maxiter = 0 (sin
optimizar en la QPU), 4000 muestras, transpilacion nivel 3 + dynamical
decoupling. Presupuesto estimado: 35-60 s de QPU en total.

| # | instancia            | parametros desde                                  | 2q (FakeFez) |
|---|----------------------|---------------------------------------------------|--------------|
| 1 | toy2                 | TSM/results/qaoa_sim_fase4/toy2/0/metadata.json   | 53           |
| 2 | tsm_ciclo_n16        | report/fase4/scale_tsm_ciclo_FCI_transfer_a0.25   | 62           |
| 3 | tsm_ciclo_n20        | idem                                              | 40           |
| 4 | tsm_ciclo_n24        | idem                                              | 48           |
| 5 | tsm_aleatoria_n16    | report/fase4/scale_tsm_aleatoria_FCI_transfer_... | 176          |
| 6 | tsm_aleatoria_n18    | idem                                              | 220          |
| 7 | tsm_aleatoria_n22    | idem                                              | 302          |

Uso (desde codigo/):
    python scripts/hardware_fase4.py                 # los 7 jobs
    python scripts/hardware_fase4.py --only 1,2      # subconjunto
    python scripts/hardware_fase4.py --dry-run       # solo muestra el plan
Salida: TSM/results/qaoa_ibm_fase4/<instancia>/0/ y report/fase4/hardware_fase4.csv
"""
import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TSM = ROOT / "TSM"
SCALE_DATA = ROOT / "report" / "scale_n" / "data"
FASE4 = ROOT / "report" / "fase4"

JOBS = [
    (1, "toy2", None, TSM / "results" / "qaoa_sim_fase4" / "toy2" / "0" / "metadata.json"),
    (2, "tsm_ciclo_n16", SCALE_DATA, FASE4 / "scale_tsm_ciclo_FCI_transfer_a0.25.csv"),
    (3, "tsm_ciclo_n20", SCALE_DATA, FASE4 / "scale_tsm_ciclo_FCI_transfer_a0.25.csv"),
    (4, "tsm_ciclo_n24", SCALE_DATA, FASE4 / "scale_tsm_ciclo_FCI_transfer_a0.25.csv"),
    (5, "tsm_aleatoria_n16", SCALE_DATA, FASE4 / "scale_tsm_aleatoria_FCI_transfer_a0.25.csv"),
    (6, "tsm_aleatoria_n18", SCALE_DATA, FASE4 / "scale_tsm_aleatoria_FCI_transfer_a0.25.csv"),
    (7, "tsm_aleatoria_n22", SCALE_DATA, FASE4 / "scale_tsm_aleatoria_FCI_transfer_a0.25.csv"),
]

FIELDS = ["job", "instance", "backend", "job_ids", "qubits", "isa_depth", "isa_2q_gates",
          "quantum_seconds", "optimum_tests", "result_tests", "optimal", "feasible",
          "p_best", "p_opt_sampled", "p_feasible", "sim_p_opt", "sim_p_feasible", "sim_p_best",
          "wall_s", "params"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="numeros de job, p. ej. 1,2,5")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--backend", default="ibm_fez")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    only = {int(x) for x in args.only.split(",")} if args.only else None
    jobs = [j for j in JOBS if only is None or j[0] in only]

    for n, inst, data_dir, src in jobs:
        ok = src.exists() and (data_dir is None or (data_dir / f"tsm_{inst}_coverage.csv").exists())
        print(f"  job {n}: {inst:20s} params <- {src.relative_to(ROOT)}  {'OK' if ok else 'FALTA'}")
        if not ok:
            sys.exit(f"Falta la fuente de parametros o la instancia del job {n}")
    if args.dry_run:
        return

    sys.path.insert(0, str(TSM))
    sys.path.insert(0, str(TSM / "quantum"))
    ibm = _load(TSM / "quantum" / "qaoa_tsm_ibm.py", "qaoa_tsm_ibm")

    FASE4.mkdir(parents=True, exist_ok=True)
    out = FASE4 / "hardware_fase4.csv"
    mode = "a" if args.append and out.exists() else "w"
    total_qs = 0.0
    with open(out, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if mode == "w":
            w.writeheader()
        for n, inst, data_dir, src in jobs:
            print(f"\n=== job {n}/{len(JOBS)}: {inst} ===", flush=True)
            t0 = time.time()
            eval_, meta = ibm.run_qaoa_ibm(
                inst, reps=1, shots=args.shots, final_shots=args.shots, maxiter=0,
                backend_name=args.backend, formulation="unbalanced",
                params_from=str(src), params_seed=42, data_dir=data_dir,
            )
            ibm._save(eval_, meta, inst, 0, method="qaoa_ibm_fase4", data_dir=data_dir)
            s = meta.get("sim_reference") or {}
            row = dict(
                job=n, instance=inst, backend=meta["backend"], job_ids=" ".join(meta["job_ids"]),
                qubits=meta["num_qubo_vars"], isa_depth=meta["isa_depth"],
                isa_2q_gates=meta["isa_2q_gates"], quantum_seconds=meta["total_quantum_seconds"],
                optimum_tests=meta["optimum_tests"], result_tests=eval_["selected_case_num"],
                optimal=meta["optimal"], feasible=eval_["feasible"], p_best=meta["p_best"],
                p_opt_sampled=meta["p_opt_sampled"], p_feasible=meta["p_feasible"],
                sim_p_opt=s.get("p_opt"), sim_p_feasible=s.get("p_feasible"), sim_p_best=s.get("p_best"),
                wall_s=round(time.time() - t0, 1),
                params=" ".join(f"{v:.6f}" for v in meta["qaoa_params"]),
            )
            w.writerow(row)
            fh.flush()
            total_qs += float(meta["total_quantum_seconds"] or 0.0)
            print(f"  -> {eval_['selected_case_num']} tests (optimo {meta['optimum_tests']}, "
                  f"{meta['optimal']}) p_opt={meta['p_opt_sampled']:.5f} p_feasible={meta['p_feasible']:.4f} "
                  f"| sim p_opt={s.get('p_opt')} | QPU acumulada {total_qs:.1f} s", flush=True)
    print(f"\nTabla en {out}; QPU total {total_qs:.1f} s")


if __name__ == "__main__":
    main()
