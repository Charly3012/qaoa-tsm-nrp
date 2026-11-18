#!/usr/bin/env python3
"""Fase 4, paso 1: calibracion de (lambda1, lambda2) para la penalizacion
desbalanceada (sin holguras) por fuerza bruta. TSM y NRP.

Para cada instancia y cada par (l1, l2) de la malla se enumeran los 2**N
estados del QUBO desbalanceado (tsm_qubo / nrp_qubo .build_qubo_unbalanced)
para responder:

  * argmin_feasible : el estado de minima energia es factible?
  * argmin_optimal  : ademas tiene la metrica optima (= ILP)?
  * rank_opt        : cuantos estados tienen energia estrictamente menor que
                      el mejor estado optimo (0 = el optimo es el argmin).
  * margin          : min energia infactible - min energia factible.
  * n_min_states    : degeneracion del minimo.
  * coef_max        : mayor |coeficiente| del QUBO (escala del paisaje).

El QUBO es lineal en (l1, l2):  E(x) = obj(x) + l1*A(x) + l2*B(x). Se
precalculan obj, A, B y la factibilidad una vez por instancia:
  TSM: A = -sum_i h_i,  B = sum_i h_i**2,  h_i = cobertura_i - 1  (>= 0 factible)
  NRP: A = -h,          B = h**2,          h = presupuesto - costo (>= 0),
       mas P * V, con V = violaciones REQUIRES/EXCLUDES y P = --penalty-logic
       (por defecto nrp_qubo.DEFAULT_PENALTY_LOGIC; 'lambda' = l1 + l2, que es
       como se hizo la calibracion original de la Fase 4)

Protocolo contra el sobreajuste: conjunto de calibracion
(benchmark + familia N <= 12, semilla 42) y conjunto de evaluacion no visto
(N = 14..24 semilla 42; N = 8, 12, 16 semillas 43 y 44).

Uso:
    python scripts/sweep_lambda.py --problem tsm
    python scripts/sweep_lambda.py --problem nrp --grid 0.5,1,2,3,5 --grid2 0.01,0.02,0.05,0.1,0.2,0.5,1
Salida: report/fase4/sweep_lambda[_<tag>].csv y sweep_lambda_pairs[_<tag>].csv
(TSM, por compatibilidad) o report/fase4/nrp/sweep_lambda*.csv (NRP).
"""
import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GEN = _load(ROOT / "scripts" / "generate_dataset.py", "generate_dataset")


class Problem:
    """Carga de modulos, generacion de instancias y precalculo por problema."""

    def __init__(self, problem, data_dir):
        self.problem = problem
        base = ROOT / problem.upper()
        sys.path.insert(0, str(base))
        self.dio = _load(base / "data_io.py", "data_io")
        self.qm = _load(base / "quantum" / f"{problem}_qubo.py", f"{problem}_qubo")
        self.ilp = _load(base / "classic" / f"ilp_{problem}.py", f"{problem}_ilp")
        self.data_dir = data_dir

    # -- instancias ---------------------------------------------------------
    def gen(self, n, seed):
        if self.problem == "tsm":
            name = f"tsm_aleatoria_n{n:02d}_s{seed}"
            if not (self.data_dir / f"tsm_{name}_coverage.csv").exists():
                GEN.gen_tsm(name, n, n, 3, seed, out_dir=self.data_dir)
            return name, None
        name = f"nrp_n{n:02d}_s{seed}"
        costs = np.random.default_rng(seed).integers(5, 40, n)   # mismo rng que gen_nrp
        budget = int(round(0.4 * costs.sum()))
        if not (self.data_dir / f"nrp_{name}_features.csv").exists():
            GEN.gen_nrp(name, n, budget, seed, requires=max(1, n // 5),
                        excludes=max(1, n // 6), out_dir=self.data_dir)
        return name, budget

    # -- precalculo ---------------------------------------------------------
    def precompute(self, name, budget, data_dir):
        """Devuelve dict(names, qp, obj, A, B, feas, metric, opt, sense, n_reqs)."""
        if self.problem == "tsm":
            df_cov, df_tests = self.dio.load_tsm(name, data_dir=data_dir)
            qp, names = self.qm.build_qp(df_cov, df_tests)
            n = len(names)
            idx = np.arange(1 << n, dtype=np.uint32)
            bits = [((idx >> k) & 1).astype(np.int8) for k in range(n)]
            obj = np.zeros(idx.size)
            for k, t in enumerate(names):
                obj += (1.0 + self.dio.TIME_WEIGHT * float(df_tests.loc[t, "time"])) * bits[k]
            H1 = np.zeros(idx.size, dtype=np.int16)
            H2 = np.zeros(idx.size, dtype=np.int32)
            feas = np.ones(idx.size, dtype=bool)
            for r in df_cov.columns:
                h = np.full(idx.size, -1, dtype=np.int8)
                for k, t in enumerate(names):
                    if int(df_cov.loc[t, r]) == 1:
                        h += bits[k]
                feas &= h >= 0
                H1 += h
                H2 += h.astype(np.int32) ** 2
            metric = np.zeros(idx.size, dtype=np.int16)
            for b in bits:
                metric += b
            opt = int(metric[feas].min())
            ilp_ev, _ = self.ilp.solve_ilp(df_cov, df_tests)
            assert ilp_ev["selected_case_num"] == opt, (name, ilp_ev["selected_case_num"], opt)
            return dict(names=names, qp=qp, obj=obj, A=-H1.astype(np.float64),
                        B=H2.astype(np.float64), feas=feas, metric=metric, opt=opt,
                        sense="min", extra=len(df_cov.columns))

        df_feat, df_deps = self.dio.load_nrp(name, data_dir=data_dir)
        qp, names = self.qm.build_qp(df_feat, df_deps, budget)
        n = len(names)
        idx = np.arange(1 << n, dtype=np.uint32)
        bits = [((idx >> k) & 1).astype(np.int8) for k in range(n)]
        pos = {nm: k for k, nm in enumerate(names)}
        obj = np.zeros(idx.size)
        for k, nm in enumerate(names):
            obj += float(qp.objective.linear[k]) * bits[k]
        value = np.zeros(idx.size)
        for k, nm in enumerate(names):
            value += float(df_feat.loc[nm, "Valor"]) * bits[k]
        h = None
        V = np.zeros(idx.size, dtype=np.int16)
        for cons in qp.linear_constraints:
            terms = cons.linear.to_dict(use_name=True)
            lhs = np.zeros(idx.size, dtype=np.int32)
            for nm, c in terms.items():
                lhs += int(round(c)) * bits[pos[nm]].astype(np.int32)
            if cons.name == "Limite_Presupuesto":
                h = int(round(cons.rhs)) - lhs
            else:
                V += (lhs > int(round(cons.rhs))).astype(np.int16)
        feas = (h >= 0) & (V == 0)
        opt = float(value[feas].max())
        ilp_ev, _ = self.ilp.solve_ilp(df_feat, df_deps, budget)
        assert abs(float(ilp_ev["total_value"]) - opt) < 1e-9, (name, ilp_ev["total_value"], opt)
        hf = h.astype(np.float64)
        Vf = V.astype(np.float64)
        return dict(names=names, qp=qp, obj=obj, A=-hf, B=hf * hf, V=Vf, feas=feas,
                    metric=value, opt=opt, sense="max", extra=len(df_deps))

    def penalty_logic(self, l1, l2, penalty_logic):
        """P efectivo de las dependencias (solo NRP; en TSM no hay)."""
        if self.problem != "nrp":
            return 0.0
        return self.qm.resolve_penalty_logic(l1, l2, penalty_logic)

    def coef_max(self, qp, l1, l2, penalty_logic=None):
        if self.problem == "nrp":
            q = self.qm.build_qubo_unbalanced(qp, l1, l2, penalty_logic=penalty_logic)
        else:
            q = self.qm.build_qubo_unbalanced(qp, l1, l2)
        lin = list(q.objective.linear.to_dict().values()) or [0.0]
        quad = list(q.objective.quadratic.to_dict().values()) or [0.0]
        return float(max(np.abs(lin).max(), np.abs(quad).max()))


def analyze_pair(pre, l1, l2, tol=1e-9, p_logic=0.0):
    values = pre["obj"] + l1 * pre["A"] + l2 * pre["B"]
    if "V" in pre:                       # NRP: violaciones de dependencias
        values = values + p_logic * pre["V"]
    feas, metric, opt = pre["feas"], pre["metric"], pre["opt"]
    argmin = int(np.argmin(values))
    vmin = float(values[argmin])
    opt_states = feas & (np.abs(metric - opt) < 1e-9)
    best_opt = float(values[opt_states].min())
    rank = int(np.count_nonzero(values < best_opt - tol))
    ties = int(np.count_nonzero((np.abs(values - best_opt) <= tol) & ~opt_states))
    min_feas = float(values[feas].min())
    min_infeas = float(values[~feas].min()) if (~feas).any() else np.nan
    return dict(
        argmin_feasible=bool(feas[argmin]), argmin_optimal=bool(opt_states[argmin]),
        argmin_metric=float(metric[argmin]), rank_opt=rank, ties_opt=ties,
        qubo_min=vmin, best_opt_value=best_opt,
        margin=(min_infeas - min_feas) if not np.isnan(min_infeas) else np.nan,
        n_min_states=int(np.count_nonzero(values <= vmin + tol)),
        uniform_p_opt=float(opt_states.mean()), uniform_p_feas=float(feas.mean()),
    )


FIELDS = ["problem", "set", "instance", "n", "n_extra", "optimum", "lambda1", "lambda2", "coef_max",
          "argmin_feasible", "argmin_optimal", "argmin_metric", "rank_opt", "ties_opt",
          "qubo_min", "best_opt_value", "margin", "n_min_states", "uniform_p_opt",
          "uniform_p_feas", "secs"]

BENCH = {"tsm": [("toy", None), ("toy2", None), ("demo", None)],
         "nrp": [("toy", 100), ("demo", 60)]}          # nrp toy2: 30 vars, no enumerable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["tsm", "nrp"], default="tsm")
    ap.add_argument("--grid", default="0.5,1,1.5,2,3,5,10", help="valores de lambda1 (y de lambda2 si no hay --grid2)")
    ap.add_argument("--grid2", default=None, help="valores de lambda2")
    ap.add_argument("--calib-sizes", default="4,6,8,10,12")
    ap.add_argument("--eval-sizes", default="14,16,18,20,22,24")
    ap.add_argument("--eval-seeds", default="43,44")
    ap.add_argument("--eval-seed-sizes", default="8,12,16")
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--max-n", type=int, default=24)
    ap.add_argument("--no-bench", action="store_true", help="omitir las instancias del benchmark")
    ap.add_argument("--tag", default="", help="sufijo de los archivos de salida")
    ap.add_argument("--penalty-logic", default=None,
                    help="solo NRP: P de REQUIRES/EXCLUDES (por defecto DEFAULT_PENALTY_LOGIC; "
                         "'lambda' = l1 + l2, calibracion original)")
    args = ap.parse_args()
    if args.penalty_logic not in (None, "lambda"):
        args.penalty_logic = float(args.penalty_logic)
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = ROOT / "report" / "fase4" / ("" if args.problem == "tsm" else "nrp")
    data_dir = ROOT / "report" / "fase4" / "data" if args.problem == "tsm" else out_dir / "data"
    sizes = lambda s: [int(x) for x in s.split(",") if x]
    l1s = [float(x) for x in args.grid.split(",")]
    l2s = [float(x) for x in (args.grid2 or args.grid).split(",")]
    pairs = [(a, b) for a in l1s for b in l2s]

    P = Problem(args.problem, data_dir)
    plan = []
    if not args.no_bench:
        plan += [("calib", nm, bud, None) for nm, bud in BENCH[args.problem]]
    for n in sizes(args.calib_sizes):
        plan.append(("calib", *P.gen(n, args.data_seed), data_dir))
    for n in sizes(args.eval_sizes):
        plan.append(("eval", *P.gen(n, args.data_seed), data_dir))
    for s in sizes(args.eval_seeds):
        for n in sizes(args.eval_seed_sizes):
            plan.append(("eval", *P.gen(n, s), data_dir))

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out_dir / f"sweep_lambda{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for subset, name, budget, ddir in plan:
            t0 = time.perf_counter()
            try:
                pre = P.precompute(name, budget, ddir)
            except MemoryError:
                print(f"[{subset}] {name}: no enumerable (memoria)", flush=True)
                continue
            n = len(pre["names"])
            if n > args.max_n:
                print(f"[{subset}] {name}: N={n} > {args.max_n}, omitida", flush=True)
                continue
            print(f"[{subset}] {name}: N={n} opt={pre['opt']} (precalculo {time.perf_counter() - t0:.1f}s)",
                  flush=True)
            for l1, l2 in pairs:
                t0 = time.perf_counter()
                p_logic = P.penalty_logic(l1, l2, args.penalty_logic)
                res = analyze_pair(pre, l1, l2, p_logic=p_logic)
                row = dict(problem=args.problem, set=subset, instance=name, n=n, n_extra=pre["extra"],
                           optimum=pre["opt"], lambda1=l1, lambda2=l2,
                           coef_max=P.coef_max(pre["qp"], l1, l2, p_logic if args.problem == "nrp" else None), secs=time.perf_counter() - t0, **res)
                rows.append(row)
                w.writerow({k: row.get(k, "") for k in FIELDS})
            fh.flush()
            n_ok = sum(1 for r in rows if r["instance"] == name and r["argmin_optimal"])
            print(f"    pares con argmin optimo: {n_ok}/{len(pairs)}", flush=True)

    summary = []
    for l1, l2 in pairs:
        sub = [r for r in rows if r["lambda1"] == l1 and r["lambda2"] == l2]
        cal = [r for r in sub if r["set"] == "calib"]
        ev = [r for r in sub if r["set"] == "eval"]
        summary.append(dict(
            lambda1=l1, lambda2=l2,
            calib_optimal=sum(r["argmin_optimal"] for r in cal), calib_total=len(cal),
            calib_feasible=sum(r["argmin_feasible"] for r in cal),
            eval_optimal=sum(r["argmin_optimal"] for r in ev), eval_total=len(ev),
            eval_feasible=sum(r["argmin_feasible"] for r in ev),
            calib_rank_median=float(np.median([r["rank_opt"] for r in cal])) if cal else "",
            calib_rank_max=max((r["rank_opt"] for r in cal), default=""),
            eval_rank_max=max((r["rank_opt"] for r in ev), default=""),
            eval_rank_median=float(np.median([r["rank_opt"] for r in ev])) if ev else "",
            margin_min=min(r["margin"] for r in sub) if sub else "",
            coef_max=max(r["coef_max"] for r in sub) if sub else "",
        ))
    # Orden: mas instancias de calibracion con argmin optimo; luego menor rango
    # mediano del optimo en calibracion (criterio de respaldo cuando la
    # formulacion no es exacta); luego evaluacion; luego paisaje mas suave.
    summary.sort(key=lambda s: (-s["calib_optimal"], s["calib_rank_median"] or 0,
                                -s["eval_optimal"], s["coef_max"]))
    with open(out_dir / f"sweep_lambda_pairs{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\nPares ordenados por instancias de calibracion con argmin optimo:")
    print(f"{'l1':>6} {'l2':>6} | calib opt/fact/tot | rank med/max | eval opt/fact/tot | rank med/max | margen min | coef max")
    for s in summary[:20]:
        print(f"{s['lambda1']:6.3f} {s['lambda2']:6.3f} | "
              f"{s['calib_optimal']:3d}/{s['calib_feasible']:3d}/{s['calib_total']:3d}        | "
              f"{str(s['calib_rank_median']):>6}/{str(s['calib_rank_max']):<6} | "
              f"{s['eval_optimal']:3d}/{s['eval_feasible']:3d}/{s['eval_total']:3d}       | "
              f"{str(s['eval_rank_median']):>6}/{str(s['eval_rank_max']):<6} | "
              f"{s['margin_min']:10.3f} | {s['coef_max']:8.2f}")
    full = [s for s in summary if s["calib_optimal"] == s["calib_total"]]
    print(f"\nPares optimos en TODA la calibracion: {len(full)} de {len(pairs)}")
    for s in full:
        print(f"  l1={s['lambda1']}, l2={s['lambda2']}: evaluacion "
              f"{s['eval_optimal']}/{s['eval_total']} optimos, {s['eval_feasible']}/{s['eval_total']} factibles")
    print(f"\nSalida en {out_dir}")


if __name__ == "__main__":
    main()
