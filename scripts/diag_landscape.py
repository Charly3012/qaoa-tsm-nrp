#!/usr/bin/env python3
"""Diagnostico del paisaje de energia (Fase 4): por que QAOA p=1 amplifica en TSM y no en NRP.

Dos modos, siempre con las instancias de `scale_n.py` (semilla de datos 42) y QAOA p=1
(malla 36x18 + COBYLA <= 100 sobre CVaR 0.25, statevector exacto):

  python scripts/diag_landscape.py lambda --sizes 10,14,18 [--penalty-logic 2.0]
      NRP: para varios pares (lambda1, lambda2) reporta el rango del optimo ILP en la energia
      QUBO, si el argmin es factible, y la masa de QAOA en el optimo, en el argmin, en los
      factibles y en las 50 energias mas bajas. --penalty-logic fija P de REQUIRES/EXCLUDES
      (por defecto P = lambda1 + lambda2, como en build_qubo_unbalanced).

  python scripts/diag_landscape.py compare --n 14
      TSM aleatoria, TSM ciclo y NRP con sus lambdas congelados: coeficientes del QUBO,
      densidad, rango del espectro y masa de QAOA en las k energias mas bajas.

Salidas usadas en el informe: report/fase4/nrp/diag_lambda_*.txt y diag_landscape_n14.txt.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import scale_n as S  # noqa: E402

ALPHA = 0.25


def qaoa_p1(eng, qubo, values, order):
    operator, _ = qubo.to_ising()
    ansatz = eng.build_ansatz(operator, 1)
    backend = eng.AerBackend(method="statevector", seed=42)
    circuit = eng.prepare_circuit(backend, ansatz)

    def probs_at(p):
        b = eng.bind_params(circuit, p).copy()
        b.save_statevector()
        return np.abs(np.asarray(backend.backend.run(b).result().get_statevector().data)) ** 2

    obj = lambda pr: eng.cvar_from_probs(pr, values, ALPHA, order=order)
    warm, _ = eng.grid_search_params(lambda p: obj(probs_at(p)), 36, 18)
    res = minimize(lambda p: obj(probs_at(p)), warm, method="COBYLA", options={"maxiter": 100})
    return probs_at(res.x), res.x


def key_map(names, var_names, n):
    idx = np.arange(1 << n, dtype=np.uint32)
    key = np.zeros(idx.size, dtype=np.uint32)
    for k, nm in enumerate(names):
        key |= ((idx >> var_names.index(nm)) & 1) << k
    return idx, key


def load(family, n):
    problem, name, budget = S.make_instance(family, n, 42)
    M = S.load_problem(problem)
    dio, qm = M["data_io"], M["qubo"]
    if problem == "tsm":
        df_a, df_b = dio.load_tsm(name, data_dir=S.DATA)
        qp, _ = qm.build_qp(df_a, df_b)
        opt = M["ilp"].solve_ilp(df_a, df_b)[0]["selected_case_num"]
        feas, opt_dec = S.uniform_tsm(df_a, opt)
    else:
        df_a, df_b = dio.load_nrp(name, data_dir=S.DATA)
        qp, _ = qm.build_qp(df_a, df_b, budget)
        opt = M["ilp"].solve_ilp(df_a, df_b, budget)[0]["total_value"]
        feas, opt_dec = S.uniform_nrp(df_a, df_b, budget, opt)
    return M, name, budget, df_a, qp, opt, feas, opt_dec


def mode_lambda(args):
    PL = args.penalty_logic
    pairs = [(0.05, 0.005), (0.1, 0.01), (0.2, 0.02), (0.5, 0.05)] if PL is not None else \
            [(0.1, 0.01), (0.5, 0.05), (1.0, 0.1), (1.0, 1.0), (2.0, 2.0)]
    print(f"penalty_logic = {PL}")
    for n in args.sizes:
        M, name, budget, df_a, qp, opt, feas, opt_dec = load("nrp", n)
        eng, qm = M["engine"], M["qubo"]
        names = df_a.index.tolist()
        idx = np.arange(1 << n, dtype=np.uint32)
        cost = np.zeros(idx.size)
        for k, f in enumerate(names):
            cost += float(df_a.loc[f, "Costo"]) * ((idx >> k) & 1)
        p_opt_u = opt_dec.mean()
        print(f"\n=== {name} N={n} B={budget} opt={opt} n_opt={opt_dec.sum()} p_opt_u={p_opt_u:.2e} "
              f"p_feas_u={feas.mean():.3f}")
        for l1, l2 in pairs:
            qubo = qm.build_qubo_unbalanced(qp, l1, l2, penalty_logic="lambda" if PL is None else PL)
            var_names = [v.name for v in qubo.variables]
            values = eng.build_value_lookup(qubo)
            _, key = key_map(names, var_names, n)
            order = np.argsort(values, kind="stable")
            rank = np.empty_like(order); rank[order] = np.arange(order.size)
            opt_states = np.where(opt_dec[key])[0]
            am = int(order[0]); am_dec = key[am]
            topk = order[:50]
            pr, _ = qaoa_p1(eng, qubo, values, order)
            mp = int(np.argmax(pr))
            lin = np.abs(qubo.objective.linear.to_array()); quad = np.abs(qubo.objective.quadratic.to_array())
            print(f"  l=({l1:>4},{l2:>5}) coefmax={max(lin.max(), quad.max()):6.1f} rank_opt={int(rank[opt_states].min()):4d} "
                  f"argmin_feas={bool(feas[am_dec])!s:5} cost_am={cost[am_dec]:.0f}/{budget} "
                  f"feas_top50={feas[key[topk]].mean():.2f} | p_opt={pr[opt_states].sum():.2e} "
                  f"A={pr[opt_states].sum()/p_opt_u:6.2f} p_argmin={pr[am]:.2e} (x{pr[am]*2**n:6.1f}) "
                  f"p_feas={pr[feas[key]].sum():.3f} p_top50={pr[topk].sum():.3f} "
                  f"maxprob: rank={int(rank[mp])} feas={bool(feas[key[mp]])}")


def mode_compare(args):
    N = args.n
    for family in ("tsm_aleatoria", "tsm_ciclo", "nrp"):
        M, name, budget, df_a, qp, opt, feas, opt_dec = load(family, N)
        eng, qm = M["engine"], M["qubo"]
        names = df_a.index.tolist()
        qubo = qm.build_qubo_unbalanced(qp, qm.DEFAULT_LAMBDA1, qm.DEFAULT_LAMBDA2,
                                        penalty_logic="lambda")   # como en E3
        var_names = [v.name for v in qubo.variables]
        values = eng.build_value_lookup(qubo)
        _, key = key_map(names, var_names, N)
        order = np.argsort(values, kind="stable")
        rank = np.empty_like(order); rank[order] = np.arange(order.size)
        opt_states = np.where(opt_dec[key])[0]
        sv = values[order]
        lin = qubo.objective.linear.to_array(); quad = qubo.objective.quadratic.to_array()
        nz = np.abs(quad[np.triu_indices(N, 1)]); nz = nz[nz > 0]
        pr, x = qaoa_p1(eng, qubo, values, order)
        k = 2 ** N
        p_opt = pr[opt_states].sum(); p_u = opt_dec.mean()
        print(f"\n{family} N={N}: opt_states={len(opt_states)} rank_opt={rank[opt_states].min()} "
              f"argmin_feas={bool(feas[key[order[0]]])}")
        print(f"  coef: lin[{lin.min():.1f},{lin.max():.1f}] quad[{nz.min():.2f},{nz.max():.2f}] "
              f"densidad quad={nz.size/(N*(N-1)/2):.2f}  |obj coef| max={abs(qp.objective.linear.to_array()).max():.2f}")
        print(f"  espectro: rango={sv[-1]-sv[0]:.1f}  gap1={sv[1]-sv[0]:.3f}")
        for kk in (1, 10, 50, 500):
            m = pr[order[:kk]].sum()
            print(f"  masa QAOA en {kk:>3} menores energias: {m:.3e}  (uniforme {kk/k:.1e}, x{m*k/kk:.1f})")
        print(f"  p_opt={p_opt:.2e} A={p_opt/p_u:.1f}  p_feas={pr[feas[key]].sum():.3f} (unif {feas.mean():.3f})  "
              f"gamma,beta={x}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("lambda"); a.add_argument("--sizes", default="10,14,18")
    a.add_argument("--penalty-logic", type=float, default=None,
                   help="P de REQUIRES/EXCLUDES; sin indicar se usa lambda1 + lambda2 (experimento E3)")
    b = sub.add_parser("compare"); b.add_argument("--n", type=int, default=14)
    args = ap.parse_args()
    if args.mode == "lambda":
        args.sizes = [int(s) for s in args.sizes.split(",")]
        mode_lambda(args)
    else:
        mode_compare(args)


if __name__ == "__main__":
    main()
