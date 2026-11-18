#!/usr/bin/env python3
"""Auditoria de la formulacion QUBO y del QAOA (simulador) para TSM y NRP.

Para cada instancia calcula, SIN tocar hardware real:

  1. Estadisticas del QUBO: variables de decision vs. holgura, penalizacion,
     rango de coeficientes y escala del paisaje de energia.
  2. Fuerza bruta sobre el QUBO (2**n): el argmin del QUBO debe coincidir con
     el optimo del ILP y debe ser factible (la penalizacion domina).
  3. Linea base uniforme: probabilidad de que un estado uniforme (|+>^n, el
     estado inicial de QAOA) caiga en el optimo / en una solucion factible, y
     probabilidad de "ver" el optimo al muestrear `final_shots` veces.
  4. QAOA p=1 (COBYLA, mismos valores por defecto que los scripts): p_opt,
     p_factible y energia esperada con los parametros optimizados, comparados
     con la linea base uniforme.
  5. Barrido en malla (gamma, beta) para p=1: mejor energia y mejor p_opt
     alcanzables, para saber si COBYLA encontro algo cercano al mejor punto.
  6. Orden real de los parametros en el circuito (gamma/beta).

Uso:
    python scripts/audit_qaoa.py                 # todas las instancias
    python scripts/audit_qaoa.py --only tsm:toy2 nrp:demo
    python scripts/audit_qaoa.py --grid 24 12    # malla gamma x beta mas gruesa
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "report" / "audit"

# Instancias y sus parametros tal como se usan en los benchmarks.
INSTANCES = [
    ("tsm", "toy", None),
    ("tsm", "toy2", None),
    ("tsm", "demo", None),
    ("nrp", "toy", 100),
    ("nrp", "toy2", 500),
    ("nrp", "demo", 60),
]


def _load(path, name):
    """Importa un modulo por ruta con un nombre unico (TSM y NRP repiten nombres)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_problem(problem):
    base = ROOT / problem.upper()
    # data_io debe ser importable como `data_io` para los modulos del problema.
    sys.path.insert(0, str(base))
    sys.path.insert(0, str(base / "quantum"))
    data_io = _load(base / "data_io.py", "data_io")
    engine = _load(base / "quantum" / "qaoa_engine.py", f"{problem}_qaoa_engine")
    qubo_mod = _load(base / "quantum" / f"{problem}_qubo.py", f"{problem}_qubo")
    sys.path.pop(0)
    sys.path.pop(0)
    return data_io, engine, qubo_mod


def build_instance(problem, dataset, budget):
    data_io, engine, qubo_mod = load_problem(problem)
    if problem == "tsm":
        df_cov, df_tests = data_io.load_tsm(dataset)
        names = df_cov.index.tolist()
        qp, _ = qubo_mod.build_qp(df_cov, df_tests)
        penalty_sim = 10.0 * (len(names) + 1)       # qaoa_tsm_simulator.py
        validate = lambda sel: data_io.is_feasible(sel, df_cov)
        objective = lambda sel: data_io.evaluate_selection(sel, df_cov, df_tests)["objective"]
        ilp_obj = _ilp_tsm(df_cov)
        sense = 1
    else:
        df_feat, df_deps = data_io.load_nrp(dataset)
        names = df_feat.index.tolist()
        qp, _ = qubo_mod.build_qp(df_feat, df_deps, budget)
        penalty_sim = float(len(names) + 1.0)       # qaoa_nrp_simulator.py
        validate = lambda sel: data_io.is_feasible(sel, df_feat, df_deps, budget)
        objective = lambda sel: -data_io.evaluate_selection(sel, df_feat, df_deps, budget)["total_value"]
        ilp_obj = _ilp_nrp(df_feat, df_deps, budget)
        sense = 1
    return dict(
        problem=problem, dataset=dataset, budget=budget, names=names, qp=qp,
        penalty_sim=penalty_sim, validate=validate, objective=objective,
        ilp_obj=ilp_obj, engine=engine, qubo_mod=qubo_mod, sense=sense,
    )


def _ilp_tsm(df_cov):
    import pulp
    tests = df_cov.index.tolist()
    prob = pulp.LpProblem("tsm", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("t", tests, cat=pulp.LpBinary)
    prob += pulp.lpSum(x[t] for t in tests)
    for r in df_cov.columns:
        prob += pulp.lpSum(int(df_cov.loc[t, r]) * x[t] for t in tests) >= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return float(pulp.value(prob.objective))


def _ilp_nrp(df_feat, df_deps, budget):
    import pulp
    feats = df_feat.index.tolist()
    prob = pulp.LpProblem("nrp", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("f", feats, cat=pulp.LpBinary)
    prob += pulp.lpSum(float(df_feat.loc[f, "Valor"]) * x[f] for f in feats)
    prob += pulp.lpSum(float(df_feat.loc[f, "Costo"]) * x[f] for f in feats) <= budget
    for _, row in df_deps.iterrows():
        o, rel, d = row["Origen"], row["Relacion"], row["Destino"]
        if o not in feats or d not in feats:
            continue
        if rel == "REQUIRES":
            prob += x[o] - x[d] <= 0
        elif rel == "EXCLUDES":
            prob += x[o] + x[d] <= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return -float(pulp.value(prob.objective))


# ---------------------------------------------------------------------------
# 1-3. Estadisticas del QUBO y fuerza bruta
# ---------------------------------------------------------------------------
def analyze_qubo(inst, qubo, label):
    engine = inst["engine"]
    names = inst["names"]
    n = qubo.get_num_vars()
    var_names = [v.name for v in qubo.variables]
    slack = [v for v in var_names if v not in names]

    lin = np.array(list(qubo.objective.linear.to_dict().values()) or [0.0])
    quad = np.array(list(qubo.objective.quadratic.to_dict().values()) or [0.0])

    values = engine.build_value_lookup(qubo)      # 2**n valores
    size = values.size
    idx = np.arange(size, dtype=np.uint64)
    bits = ((idx[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.uint8)
    dec_cols = [var_names.index(nm) for nm in names]

    # Factibilidad y objetivo original sobre las variables de decision
    # (2**n_decision combinaciones), proyectadas luego a los 2**n estados QUBO.
    n_dec = len(names)
    feas_dec = np.zeros(1 << n_dec, dtype=bool)
    obj_dec = np.full(1 << n_dec, np.inf)
    for key in range(1 << n_dec):
        sel = [names[k] for k in range(n_dec) if (key >> k) & 1]
        if inst["validate"](sel):
            feas_dec[key] = True
            obj_dec[key] = inst["objective"](sel)
    keys = np.zeros(size, dtype=np.int64)
    for k, col in enumerate(dec_cols):
        keys |= bits[:, col].astype(np.int64) << k
    feasible = feas_dec[keys]
    orig_obj = obj_dec[keys]

    argmin = int(np.argmin(values))
    qubo_min = float(values[argmin])
    argmin_feasible = bool(feasible[argmin])
    best_orig = float(orig_obj[feasible].min()) if feasible.any() else np.nan
    tol = 1e-6
    opt_states = feasible & (orig_obj <= best_orig + tol)
    # Estados cuyo valor QUBO coincide con el minimo (degeneracion)
    qubo_min_states = values <= qubo_min + tol

    min_infeasible = float(values[~feasible].min()) if (~feasible).any() else np.nan
    max_feasible = float(values[feasible].max()) if feasible.any() else np.nan

    uniform_p_opt = float(opt_states.mean())
    uniform_p_feas = float(feasible.mean())

    return dict(
        label=label,
        n_qubo=n, n_decision=len(names), n_slack=len(slack),
        constant=float(qubo.objective.constant),
        lin_min=float(lin.min()), lin_max=float(lin.max()),
        quad_min=float(quad.min()), quad_max=float(quad.max()),
        qubo_min=qubo_min, argmin_feasible=argmin_feasible,
        argmin_orig_obj=float(orig_obj[argmin]) if feasible[argmin] else np.nan,
        brute_opt_obj=best_orig, ilp_opt_obj=inst["ilp_obj"],
        brute_matches_ilp=bool(abs(best_orig - inst["ilp_obj"]) < 1e-6),
        n_opt_states=int(opt_states.sum()),
        n_qubo_min_states=int(qubo_min_states.sum()),
        min_infeasible_qubo=min_infeasible, max_feasible_qubo=max_feasible,
        penalty_dominates=bool(min_infeasible > qubo_min + tol) if not np.isnan(min_infeasible) else True,
        uniform_p_opt=uniform_p_opt, uniform_p_feas=uniform_p_feas,
        uniform_energy=float(values.mean()),
        p_see_opt_8000_uniform=1.0 - (1.0 - uniform_p_opt) ** 8000,
        _values=values, _opt_states=opt_states, _feasible=feasible, _orig_obj=orig_obj,
    )


# ---------------------------------------------------------------------------
# 4-5. QAOA p=1 y malla
# ---------------------------------------------------------------------------
def _statevector_probs(sim, circuit, params, engine):
    # Misma asignacion que el motor (por nombre: [gamma_0, beta_0, ...])
    bound = engine.bind_params(circuit, params)
    bound.save_statevector()
    sv = sim.run(bound).result().get_statevector()
    return np.abs(np.asarray(sv.data)) ** 2


def qaoa_p1(inst, qubo, stats, maxiter=100, seed=42, grid=(36, 18)):
    from qiskit_aer import AerSimulator
    engine = inst["engine"]
    operator, offset = qubo.to_ising()
    ansatz = engine.build_ansatz(operator, 1)
    circuit = engine.prepare_circuit(None, ansatz)
    param_names = [p.name for p in circuit.parameters]
    sim = AerSimulator(method="statevector", seed_simulator=seed)
    values = stats["_values"]
    opt_states = stats["_opt_states"]
    feas = stats["_feasible"]

    def energy(p):
        return float(np.dot(_statevector_probs(sim, circuit, p, engine), values))

    # COBYLA desde el mismo punto inicial que los scripts (seed=42, start 0)
    rng = np.random.default_rng(seed)
    x0 = engine.initial_params(1, rng)
    t0 = time.perf_counter()
    res = minimize(energy, x0, method="COBYLA", options={"maxiter": maxiter})
    t_cobyla = time.perf_counter() - t0
    probs = _statevector_probs(sim, circuit, res.x, engine)
    cob = dict(
        cobyla_energy=float(res.fun), cobyla_nfev=int(res.nfev),
        cobyla_p_opt=float(probs[opt_states].sum()),
        cobyla_p_feas=float(probs[feas].sum()),
        cobyla_x0=x0.tolist(), cobyla_x=res.x.tolist(), cobyla_secs=t_cobyla,
        # Probabilidad de VER el optimo al menos una vez en 8000 muestras
        cobyla_p_see_opt_8000=1.0 - (1.0 - float(probs[opt_states].sum())) ** 8000,
    )

    # Malla (gamma, beta): el circuito asigna por orden de `circuit.parameters`
    ng, nb = grid
    gammas = np.linspace(0, 2 * np.pi, ng, endpoint=False)
    betas = np.linspace(0, np.pi, nb, endpoint=False)
    best_e, best_p = (np.inf, None), (-1.0, None)
    t0 = time.perf_counter()
    for g in gammas:
        for b in betas:
            p = [g, b]
            pr = _statevector_probs(sim, circuit, p, engine)
            e = float(np.dot(pr, values))
            po = float(pr[opt_states].sum())
            if e < best_e[0]:
                best_e = (e, (float(g), float(b)))
            if po > best_p[0]:
                best_p = (po, (float(g), float(b)))
    t_grid = time.perf_counter() - t0
    grid_res = dict(
        grid_best_energy=best_e[0], grid_best_energy_at=best_e[1],
        grid_best_p_opt=best_p[0], grid_best_p_opt_at=best_p[1],
        grid_points=ng * nb, grid_secs=t_grid,
    )
    return dict(param_order=param_names, **cob, **grid_res)


def fmt(x):
    if isinstance(x, float):
        if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0):
            return f"{x:.3e}"
        return f"{x:.4f}"
    return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="p.ej. tsm:toy2 nrp:demo")
    parser.add_argument("--grid", nargs=2, type=int, default=[36, 18])
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--max-qubits", type=int, default=24)
    parser.add_argument("--no-qaoa", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for problem, dataset, budget in INSTANCES:
        if args.only and f"{problem}:{dataset}" not in args.only:
            continue
        inst = build_instance(problem, dataset, budget)
        qubo_mod = inst["qubo_mod"]
        qubo_sim = qubo_mod.build_qubo(inst["qp"], penalty=inst["penalty_sim"])
        # Penalizacion que usa el script de hardware (TSM: automatica; NRP: n+1)
        qubo_hw = qubo_mod.build_qubo(inst["qp"], penalty=None if problem == "tsm" else inst["penalty_sim"])
        hw_penalty = None if problem == "tsm" else inst["penalty_sim"]
        try:
            # qiskit expone la penalizacion efectiva usada por el convertidor
            from qiskit_optimization.converters import QuadraticProgramToQubo
            conv = QuadraticProgramToQubo(penalty=None)
            conv.convert(inst["qp"])
            hw_penalty = conv.penalty
        except Exception:
            pass

        n = qubo_sim.get_num_vars()
        print(f"\n=== {problem.upper()} / {dataset} (budget={budget}) — {n} qubits ===")
        if n > args.max_qubits:
            print(f"  [omitido: {n} qubits > {args.max_qubits}; fuerza bruta/statevector no viable]")
            rows.append(dict(problem=problem, dataset=dataset, n_qubo=n, skipped=True))
            continue

        stats = analyze_qubo(inst, qubo_sim, "sim")
        same_hw = (qubo_hw.objective.linear.to_array().tolist() == qubo_sim.objective.linear.to_array().tolist()
                   and qubo_hw.objective.quadratic.to_array().tolist() == qubo_sim.objective.quadratic.to_array().tolist())
        row = dict(problem=problem, dataset=dataset, budget=budget,
                   penalty_sim=inst["penalty_sim"], penalty_hw=hw_penalty,
                   qubo_hw_equals_sim=same_hw)
        row.update({k: v for k, v in stats.items() if not k.startswith("_")})
        if not args.no_qaoa:
            row.update(qaoa_p1(inst, qubo_sim, stats, maxiter=args.maxiter, grid=tuple(args.grid)))
        rows.append(row)

        for k, v in row.items():
            if k in ("problem", "dataset"):
                continue
            print(f"  {k:28s} {fmt(v)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "audit_qaoa.csv", index=False)
    with open(OUT_DIR / "audit_qaoa.json", "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print(f"\nGuardado en {OUT_DIR}")


if __name__ == "__main__":
    main()
