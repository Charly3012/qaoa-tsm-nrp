#!/usr/bin/env python3
"""Fase 2: escalamiento gradual de N en simulador (sin hardware).

Para cada familia de instancias y cada tamano N:
  * genera la instancia (scripts/generate_dataset.py, semilla fija) en
    report/scale_n/data/ para no tocar TSM/data ni NRP/data; con una
    --data-seed distinta de 42 va a report/scale_n/data/ds<seed>/, con la
    semilla en el nombre y generada a partir de (data_seed, N);
  * cuenta qubits del QUBO (decision + holgura);
  * resuelve con ILP (optimo), greedy y SA (formulacion original);
  * calcula la linea base uniforme EXACTA (fraccion de asignaciones factibles
    y optimas) enumerando las 2**N decisiones de forma vectorizada;
  * baseline "aleatorio + filtro": mismas `final_shots` muestras uniformes y
    el mismo criterio que extract_best (mejor factible por objetivo original);
  * QAOA p=1 (COBYLA, statevector exacto) con varias semillas: resultado,
    p_best, p_feasible, p_opt exacto del estado final, evaluaciones, tiempo.

Familias:
  tsm_ciclo      tests = reqs = N, cada requisito cubierto por 2 tests
                 consecutivos (sin variables de holgura: qubits = N).
  tsm_aleatoria  tests = reqs = N, cada requisito cubierto por 3 tests al
                 azar (genera holguras: qubits > N).
  nrp            N features, budget = 40% del costo total,
                 REQUIRES = max(1, N//5), EXCLUDES = max(1, N//6).

Las instancias cuyo QUBO excede --max-qubits solo registran el conteo de
qubits y los clasicos (QAOA no viable en statevector).

Uso:
    python scripts/scale_n.py --family tsm_ciclo --sizes 4,6,8,...
    python scripts/scale_n.py --family nrp --sizes 4,6,8 --seeds 42,43,44
Salida incremental: report/scale_n/scale_<family>.csv

Fase 4 (experimento E3): --config selecciona
  base : QUBO con holguras, energia esperada, inicio aleatorio (= Fase 2)
  F    : penalizacion desbalanceada sin holguras (lambda1/lambda2)
  FC   : F + CVaR_alpha como objetivo
  FCI  : FC + inicializacion informada (--init grid: mejor punto de la malla
         gamma x beta, una corrida por instancia con seed="grid";
         --init transfer: cada semilla parte de sus parametros optimizados
         en el N anterior de la misma corrida; el primer N es aleatorio)
    python scripts/scale_n.py --family tsm_aleatoria --sizes 4,...,24 --config FC --alpha 0.1 --out-dir report/fase4
Salida: <out-dir>/scale_<family>_<config>[_<init>][_a<alpha>].csv
"""
import argparse
import csv
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "scale_n"
DATA = OUT / "data"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_problem(problem):
    """Carga data_io, motor, QUBO y clasicos de un problema (TSM/NRP)."""
    base = ROOT / problem.upper()
    data_io = _load(base / "data_io.py", "data_io")      # lo importan los clasicos
    mods = dict(
        data_io=data_io,
        engine=_load(base / "quantum" / "qaoa_engine.py", f"{problem}_engine"),
        qubo=_load(base / "quantum" / f"{problem}_qubo.py", f"{problem}_qubo"),
        ilp=_load(base / "classic" / f"ilp_{problem}.py", f"{problem}_ilp"),
        greedy=_load(base / "classic" / f"greedy_{problem}.py", f"{problem}_greedy"),
        sa=_load(base / "classic" / f"sa_{problem}.py", f"{problem}_sa"),
    )
    return mods


GEN = _load(ROOT / "scripts" / "generate_dataset.py", "generate_dataset")


# ---------------------------------------------------------------------------
# Generacion de instancias
# ---------------------------------------------------------------------------
LEGACY_DATA_SEED = 42   # semilla de las corridas exploratorias (E2, E3 y hardware)


def instance_dir(seed):
    """Carpeta de las instancias de una semilla de datos.

    La semilla 42 conserva report/scale_n/data/ (la leen hardware_fase4.py y
    diag_landscape.py); cualquier otra va a su subcarpeta ds<seed>/ para que
    generar instancias nuevas nunca sobrescriba las existentes.
    """
    return DATA if seed == LEGACY_DATA_SEED else DATA / f"ds{seed}"


def instance_seed(seed, n):
    """Semilla con la que se genera la instancia de tamano n.

    Con la semilla 42 se usa tal cual (reproduce las instancias exploratorias).
    Con cualquier otra se deriva de (seed, n): asi las instancias de distinto
    tamano no comparten el inicio de la secuencia aleatoria.
    """
    return seed if seed == LEGACY_DATA_SEED else [seed, n]


def make_instance(family, n, seed):
    legacy = seed == LEGACY_DATA_SEED
    name = f"{family}_n{n:02d}" if legacy else f"{family}_n{n:02d}_d{seed}"
    out = instance_dir(seed)
    gseed = instance_seed(seed, n)
    if family == "tsm_ciclo":
        GEN.gen_tsm(name, n, n, 2, gseed, out_dir=out)
        return "tsm", name, None
    if family == "tsm_aleatoria":
        GEN.gen_tsm(name, n, n, 3, gseed, out_dir=out)
        return "tsm", name, None
    if family == "nrp":
        costs = np.random.default_rng(gseed).integers(5, 40, n)   # mismo rng que gen_nrp
        budget = int(round(0.4 * costs.sum()))
        GEN.gen_nrp(name, n, budget, gseed, requires=max(1, n // 5),
                    excludes=max(1, n // 6), out_dir=out)
        return "nrp", name, budget
    raise ValueError(family)


# ---------------------------------------------------------------------------
# Linea base uniforme exacta sobre las 2**N decisiones
# ---------------------------------------------------------------------------
def _popcount(x):
    x = x.astype(np.uint32)
    c = np.zeros(x.shape, dtype=np.uint8)
    for k in range(32):
        c += ((x >> k) & 1).astype(np.uint8)
    return c


def uniform_tsm(df_cov, opt_tests):
    names = df_cov.index.tolist()
    n = len(names)
    idx = np.arange(1 << n, dtype=np.uint32)
    feas = np.ones(idx.size, dtype=bool)
    for r in df_cov.columns:
        mask = 0
        for k, t in enumerate(names):
            if int(df_cov.loc[t, r]) == 1:
                mask |= 1 << k
        feas &= (idx & np.uint32(mask)) != 0
    ntests = _popcount(idx)
    opt = feas & (ntests == opt_tests)
    return feas, opt


def uniform_nrp(df_feat, df_deps, budget, opt_value):
    names = df_feat.index.tolist()
    n = len(names)
    idx = np.arange(1 << n, dtype=np.uint32)
    cost = np.zeros(idx.size, dtype=np.float64)
    value = np.zeros(idx.size, dtype=np.float64)
    for k, f in enumerate(names):
        b = (idx >> k) & 1
        cost += float(df_feat.loc[f, "Costo"]) * b
        value += float(df_feat.loc[f, "Valor"]) * b
    feas = cost <= budget
    pos = {f: k for k, f in enumerate(names)}
    for _, row in df_deps.iterrows():
        o, rel, d = row["Origen"], row["Relacion"], row["Destino"]
        if o not in pos or d not in pos:
            continue
        xo, xd = (idx >> pos[o]) & 1, (idx >> pos[d]) & 1
        if rel == "REQUIRES":
            feas &= ~((xo == 1) & (xd == 0))
        elif rel == "EXCLUDES":
            feas &= ~((xo == 1) & (xd == 1))
    opt = feas & (np.abs(value - opt_value) < 1e-9)
    return feas, opt


# ---------------------------------------------------------------------------
# Experimento por instancia
# ---------------------------------------------------------------------------
def run_instance(family, n, args, writer, fh):
    problem, name, budget = make_instance(family, n, args.data_seed)
    data_dir = instance_dir(args.data_seed)
    M = load_problem(problem)
    dio, eng, qm = M["data_io"], M["engine"], M["qubo"]

    if problem == "tsm":
        df_a, df_b = dio.load_tsm(name, data_dir=data_dir)
        names = df_a.index.tolist()
        qp, _ = qm.build_qp(df_a, df_b)
        penalty = 10.0 * (len(names) + 1)
        validate = lambda sel: dio.is_feasible(sel, df_a)
        objective = lambda sel: dio.evaluate_selection(sel, df_a, df_b)["objective"]
        metric_of = lambda ev: ev["selected_case_num"]
        better = lambda a, b: a < b
        t0 = time.perf_counter(); ilp_ev, _ = M["ilp"].solve_ilp(df_a, df_b); t_ilp = time.perf_counter() - t0
        t0 = time.perf_counter(); gr_ev, _ = M["greedy"].solve_greedy(df_a, df_b); t_gr = time.perf_counter() - t0
        t0 = time.perf_counter(); sa_ev, _ = M["sa"].solve_sa(df_a, df_b, seed=args.data_seed); t_sa = time.perf_counter() - t0
        eval_sel = lambda sel: dio.evaluate_selection(sel, df_a, df_b)
    else:
        df_a, df_b = dio.load_nrp(name, data_dir=data_dir)
        names = df_a.index.tolist()
        qp, _ = qm.build_qp(df_a, df_b, budget)
        penalty = float(len(names) + 1.0)
        validate = lambda sel: dio.is_feasible(sel, df_a, df_b, budget)
        objective = lambda sel: dio.evaluate_selection(sel, df_a, df_b, budget)["objective"]
        metric_of = lambda ev: ev["total_value"]
        better = lambda a, b: a > b
        t0 = time.perf_counter(); ilp_ev, _ = M["ilp"].solve_ilp(df_a, df_b, budget); t_ilp = time.perf_counter() - t0
        t0 = time.perf_counter(); gr_ev, _ = M["greedy"].solve_greedy(df_a, df_b, budget); t_gr = time.perf_counter() - t0
        t0 = time.perf_counter(); sa_ev, _ = M["sa"].solve_sa(df_a, df_b, budget, seed=args.data_seed); t_sa = time.perf_counter() - t0
        eval_sel = lambda sel: dio.evaluate_selection(sel, df_a, df_b, budget)

    opt = metric_of(ilp_ev)
    cfg = args.config
    p_logic = None
    if cfg == "base":
        qubo = qm.build_qubo(qp, penalty=penalty)
        l1 = l2 = None
    else:
        l1 = qm.DEFAULT_LAMBDA1 if args.lambda1 is None else args.lambda1
        l2 = qm.DEFAULT_LAMBDA2 if args.lambda2 is None else args.lambda2
        if l1 is None or l2 is None:
            raise ValueError(f"Faltan lambda1/lambda2 (o DEFAULT_LAMBDA* en {problem}_qubo.py)")
        if problem == "nrp":      # penalizacion de dependencias propia (solo NRP)
            p_logic = qm.resolve_penalty_logic(l1, l2, args.penalty_logic)
            qubo = qm.build_qubo_unbalanced(qp, l1, l2, penalty_logic=p_logic)
        else:
            qubo = qm.build_qubo_unbalanced(qp, l1, l2)
        penalty = None
    use_cvar = cfg in ("FC", "FCI")
    init = args.init if cfg == "FCI" else "random"
    n_qubo = qubo.get_num_vars()
    var_names = [v.name for v in qubo.variables]
    n_dec = len(names)
    base = dict(family=family, problem=problem, n=n, instance=name, budget=budget,
                n_decision=n_dec, n_qubo=n_qubo, n_slack=n_qubo - n_dec, optimum=opt,
                config=cfg, lambda1=l1, lambda2=l2, data_seed=args.data_seed,
                penalty_logic=p_logic,
                objective="cvar" if use_cvar else "energy",
                alpha=args.alpha if use_cvar else "", init=init, reps=args.reps)

    def emit(**row):
        full = {**base, **row}
        writer.writerow({k: full.get(k, "") for k in FIELDS})
        fh.flush()
        tag = f"{row['method']:>10}"
        print(f"[{family} N={n:2d} q={n_qubo:2d}] {tag} metric={row.get('metric')} "
              f"opt={opt} {('p_best=%.2e' % row['p_best']) if row.get('p_best') not in (None, '') else ''} "
              f"t={row.get('secs', 0):.1f}s", flush=True)

    for meth, ev, t in [("ilp", ilp_ev, t_ilp), ("greedy", gr_ev, t_gr), ("sa", sa_ev, t_sa)]:
        m = metric_of(ev)
        emit(method=meth, seed=args.data_seed, metric=m, optimal=int(m == opt),
             feasible=int(ev["feasible"]), secs=t)

    # Linea base uniforme exacta (solo si 2**N es enumerable)
    p_opt_u = p_feas_u = None
    if n_dec <= args.max_enum:
        if problem == "tsm":
            feas_dec, opt_dec = uniform_tsm(df_a, opt)
        else:
            feas_dec, opt_dec = uniform_nrp(df_a, df_b, budget, opt)
        p_opt_u, p_feas_u = float(opt_dec.mean()), float(feas_dec.mean())
        p_see = 1.0 - (1.0 - p_opt_u) ** args.final_shots
        emit(method="uniforme", seed="", p_opt=p_opt_u, p_feasible=p_feas_u, p_see_opt=p_see,
             n_opt_states=int(opt_dec.sum()), secs=0.0)
    else:
        feas_dec = opt_dec = None

    # Aleatorio + filtro: final_shots muestras uniformes, mejor factible por objetivo.
    # No depende de la configuracion: con --no-random se omite (basta calcularlo
    # una vez por instancia, en una sola de las configuraciones).
    for s in ([] if args.no_random else args.seeds):
        rng = np.random.default_rng(s)
        t0 = time.perf_counter()
        samples = rng.integers(0, 2, size=(args.final_shots, n_dec), dtype=np.uint8)
        uniq = np.unique(samples, axis=0)
        best, best_obj, n_feas = None, None, 0
        for row in uniq:
            sel = [names[k] for k in range(n_dec) if row[k]]
            if validate(sel):
                n_feas += 1
                o = objective(sel)
                if best_obj is None or o < best_obj:
                    best_obj, best = o, sel
        t = time.perf_counter() - t0
        m = metric_of(eval_sel(best)) if best is not None else None
        emit(method="aleatorio", seed=s, metric=m, optimal=int(m == opt) if m is not None else 0,
             feasible=int(best is not None), secs=t)

    # QAOA p=1 (statevector exacto) si cabe
    if n_qubo > args.max_qubits:
        print(f"[{family} N={n}] QAOA omitido: {n_qubo} qubits > {args.max_qubits}", flush=True)
        return
    operator, offset = qubo.to_ising()
    ansatz = eng.build_ansatz(operator, args.reps)
    backend = eng.AerBackend(method="statevector", seed=42)
    circuit = eng.prepare_circuit(backend, ansatz)
    t0 = time.perf_counter()
    values = eng.build_value_lookup(qubo)
    order = np.argsort(values, kind="stable") if use_cvar else None
    # proyeccion estado QUBO -> asignacion de decision (para p_opt exacto)
    idx = np.arange(1 << n_qubo, dtype=np.uint32)
    key = np.zeros(idx.size, dtype=np.uint32)
    for k, nm in enumerate(names):
        key |= ((idx >> var_names.index(nm)) & 1) << k
    del idx
    opt_states = opt_dec[key] if opt_dec is not None else None
    feas_states = feas_dec[key] if feas_dec is not None else None
    del key
    t_lookup = time.perf_counter() - t0

    def probs_at(p, circ=None):
        bound = eng.bind_params(circ if circ is not None else circuit, p)
        probe = bound.copy()
        probe.save_statevector()
        sv = backend.backend.run(probe).result().get_statevector()
        return np.abs(np.asarray(sv.data)) ** 2

    def objective_of(pr):
        if use_cvar:
            return eng.cvar_from_probs(pr, values, args.alpha, order=order)
        return float(np.dot(pr, values))

    # Inicializacion informada (FCI)
    runs = [(s, None) for s in args.seeds]           # (etiqueta, warm)
    t_init = 0.0
    if init == "grid":
        circ1 = circuit if args.reps == 1 else eng.prepare_circuit(backend, eng.build_ansatz(operator, 1))
        t0 = time.perf_counter()
        warm, _ = eng.grid_search_params(lambda p: objective_of(probs_at(p, circ1)), *args.grid)
        t_init = time.perf_counter() - t0
        runs = [("grid", warm)]
    elif init == "transfer":
        runs = [(s, TRANSFER.get(s)) for s in args.seeds]
        sin_previo = [s for s, w in runs if w is None]
        if sin_previo and n == args.first_size:
            # Primer tamano de la llamada: arranca aleatorio por diseno. En el CSV
            # queda warm=0; el analisis solo acepta filas F+C+I con warm=1.
            print(f"NOTA [transfer] N={n} es el primer tamano de la cadena: inicio aleatorio "
                  f"(warm=0). Para que un tamano cuente como F+C+I debe correrse en la misma "
                  f"llamada que los anteriores.", flush=True)
        elif sin_previo:
            print(f"AVISO [transfer] N={n}: sin parametros del tamano anterior para las semillas "
                  f"{sin_previo}; arrancan con inicio aleatorio (warm=0)", flush=True)

    for s, warm in runs:
        if warm is not None:
            x0 = eng.initial_params(args.reps, None, init="transfer", warm=warm)
        else:
            x0 = eng.initial_params(args.reps, np.random.default_rng(s))
        nfev = [0]

        def energy(p):
            nfev[0] += 1
            return objective_of(probs_at(p))

        t0 = time.perf_counter()
        res = minimize(energy, x0, method="COBYLA", options={"maxiter": args.maxiter})
        t_opt = time.perf_counter() - t0 + t_init
        if init == "transfer":
            TRANSFER[s] = np.asarray(res.x)
        pr = probs_at(res.x)
        p_opt = float(pr[opt_states].sum()) if opt_states is not None else None
        p_feas_exact = float(pr[feas_states].sum()) if feas_states is not None else None
        del pr
        t1 = time.perf_counter()
        assignment, _, feasible, counts = eng.extract_best(
            backend, circuit, res.x, qubo, names, qp_value(qm), validate,
            args.final_shots, objective=objective)
        sel = [nm for nm in names if assignment.get(nm, 0) == 1]
        ev = eval_sel(sel)
        n_sh = sum(counts.values())
        p_best = sum(c for b, c in counts.items()
                     if {nm: v for nm, v in eng.decode_full(b, qubo).items() if nm in assignment} == assignment) / n_sh
        t_samp = time.perf_counter() - t1
        m = metric_of(ev)
        emit(method="qaoa", seed=s, metric=m, optimal=int(m == opt and ev["feasible"]),
             feasible=int(ev["feasible"]), p_best=p_best, p_opt=p_opt, p_feasible=p_feas_exact,
             energy=float(res.fun), nfev=nfev[0], secs=t_opt + t_samp + t_lookup,
             warm=int(warm is not None),
             secs_opt=t_opt, params=" ".join(f"{v:.6f}" for v in res.x))


def qp_value(qm):
    return qm.qubo_value


TRANSFER = {}   # semilla -> parametros optimizados en el N anterior (init=transfer)

FIELDS = ["family", "problem", "n", "instance", "budget", "n_decision", "n_qubo", "n_slack",
          "optimum", "config", "lambda1", "lambda2", "objective", "alpha", "init", "reps",
          "method", "seed", "metric", "optimal", "feasible", "p_best", "p_opt",
          "p_feasible", "p_see_opt", "n_opt_states", "energy", "nfev", "secs", "secs_opt",
          "params", "warm", "data_seed", "penalty_logic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["tsm_ciclo", "tsm_aleatoria", "nrp"])
    ap.add_argument("--sizes", required=True, help="p. ej. 4,6,8,10")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=100)
    ap.add_argument("--final-shots", type=int, default=8000)
    ap.add_argument("--max-qubits", type=int, default=24)
    ap.add_argument("--max-enum", type=int, default=26, help="N max para la base uniforme exacta")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--no-random", action="store_true",
                    help="omite la linea base aleatorio + filtro (ya calculada en otra configuracion)")
    # Fase 4
    ap.add_argument("--config", choices=["base", "F", "FC", "FCI"], default="base")
    ap.add_argument("--lambda1", type=float, default=None)
    ap.add_argument("--lambda2", type=float, default=None)
    ap.add_argument("--penalty-logic", default=None,
                    help="solo NRP sin holguras: P de REQUIRES/EXCLUDES. Por defecto "
                         "nrp_qubo.DEFAULT_PENALTY_LOGIC; 'lambda' = lambda1 + lambda2 "
                         "(reproduce el experimento exploratorio E3)")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--init", choices=["grid", "transfer"], default="grid", help="solo FCI")
    ap.add_argument("--grid", default="18,9", help="malla gamma,beta para init=grid")
    ap.add_argument("--out-dir", default=None, help="carpeta de salida (default report/scale_n)")
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",")]
    args.grid = tuple(int(x) for x in args.grid.split(","))
    sizes = [int(s) for s in args.sizes.split(",")]
    args.first_size = sizes[0]
    if args.penalty_logic not in (None, "lambda"):
        args.penalty_logic = float(args.penalty_logic)

    out_dir = Path(args.out_dir) if args.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.config == "base" and args.out_dir is None:
        out = out_dir / f"scale_{args.family}.csv"          # nombre heredado de la Fase 2
    else:
        suffix = args.config
        if args.config == "FCI":
            suffix += f"_{args.init}"
        if args.config in ("FC", "FCI"):
            suffix += f"_a{args.alpha:g}"
        if args.reps != 1:
            suffix += f"_p{args.reps}"
        out = out_dir / f"scale_{args.family}_{suffix}.csv"
    mode = "a" if args.append and out.exists() else "w"
    with open(out, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if mode == "w":
            w.writeheader()
        for n in sizes:
            run_instance(args.family, n, args, w, fh)
    print(f"\nTabla en {out}")


if __name__ == "__main__":
    main()
