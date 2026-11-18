#!/usr/bin/env python3
"""QAOA (configuracion F+C+I, transferencia, p=1) frente a los metodos clasicos (ILP, greedy, SA)
y a aleatorio + filtro, en benchmark y escalamiento. Tablas y figuras para el informe de resultados.

Entradas:
  report/fase4/scale_tsm_{aleatoria,ciclo}_FCI_transfer_a0.25.csv, report/fase4/nrp/scale_nrp_FCI_transfer_a0.25.csv
  report/fase4/nmax/scale_tsm_aleatoria_FCI_transfer_a0.25.csv (N = 26, 28, 30)
  report/fase4/sweep_tsm_FCI_a0.25.csv, report/fase4/nrp/sweep_nrp_FCI_a0.25.csv (benchmark QAOA)
  {TSM,NRP}/results/{ilp,greedy,sa}/<dataset>/0/summary.csv (benchmark clasicos)
Salidas:
  report/fase4/resumen_vs_clasicos.csv, report/fase4/benchmark_vs_clasicos.csv
  report/figures/fase4/c1_brecha_metodos.png, c2_tiempo_metodos.png, c3_exito_metodos.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "report" / "fase4"
FIG = ROOT / "report" / "figures" / "fase4"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

FAM = {"tsm_aleatoria": "TSM cobertura aleatoria (k=3)", "tsm_ciclo": "TSM cobertura en ciclo (k=2)",
       "nrp": "NRP (presupuesto 40 %)"}
METH = [("ilp", "ILP (CBC)", "#000000", "-"), ("greedy", "Greedy", "#8c564b", "-"),
        ("sa", "Recocido simulado", "#9467bd", "-"), ("aleatorio", "Aleatorio + filtro (8000)", "#7f7f7f", ":"),
        ("qaoa", "QAOA p=1 (F+C+I)", "#2ca02c", "-")]


def gap_pct(problem, metric, opt):
    if metric is None or np.isnan(metric):
        return np.nan
    return 100.0 * ((metric - opt) / opt if problem == "tsm" else (opt - metric) / opt)


def load_scaling():
    frames = [pd.read_csv(R / "scale_tsm_aleatoria_FCI_transfer_a0.25.csv"),
              pd.read_csv(R / "nmax" / "scale_tsm_aleatoria_FCI_transfer_a0.25.csv"),
              pd.read_csv(R / "scale_tsm_ciclo_FCI_transfer_a0.25.csv"),
              pd.read_csv(R / "nrp" / "scale_nrp_FCI_transfer_a0.25.csv")]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["family", "n", "method", "seed"], keep="last")
    rows = []
    for (fam, n), g in df.groupby(["family", "n"]):
        problem = g.problem.iloc[0]
        opt = float(g.optimum.iloc[0])
        base = dict(family=fam, problem=problem, n=int(n), n_qubits=int(g.n_qubo.iloc[0]))
        for m, _, _, _ in METH:
            h = g[g.method == m]
            if h.empty:
                rows.append(dict(base, method=m))
                continue
            met = pd.to_numeric(h.metric, errors="coerce")
            rows.append(dict(base, method=m, runs=len(h), metric_median=met.median(),
                             gap_median=gap_pct(problem, met.median(), opt),
                             gap_mean=np.nanmean([gap_pct(problem, v, opt) for v in met]),
                             optimal=int(h.optimal.fillna(0).sum()), feasible=int(h.feasible.fillna(0).sum()),
                             secs_median=h.secs.median(),
                             p_opt_median=h.p_opt.median() if m == "qaoa" else np.nan,
                             p_feasible_median=h.p_feasible.median() if m == "qaoa" else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(R / "resumen_vs_clasicos.csv", index=False)
    return out


def load_benchmark():
    rows = []
    for problem, datasets in (("tsm", ["toy", "toy2", "demo"]), ("nrp", ["toy", "demo"])):
        sweep = pd.read_csv(R / ("sweep_tsm_FCI_a0.25.csv" if problem == "tsm" else "nrp/sweep_nrp_FCI_a0.25.csv"))
        for ds in datasets:
            opt = None
            for m in ("ilp", "greedy", "sa"):
                p = ROOT / problem.upper() / "results" / m / ds / "0" / "summary.csv"
                if not p.exists():
                    continue
                s = pd.read_csv(p).iloc[0]
                metric = float(s["selected_case_num"] if problem == "tsm" else s["total_value"])
                if m == "ilp":
                    opt = metric
                rows.append(dict(problem=problem, dataset=ds, method=m, metric=metric,
                                 gap_pct=gap_pct(problem, metric, opt), secs=float(s["execution_time"]),
                                 feasible=bool(s["feasible"])))
            q = sweep[(sweep.dataset == ds) & (sweep.reps == 1) & (sweep.starts == sweep.starts.min())]
            if q.empty:
                continue
            rows.append(dict(problem=problem, dataset=ds, method="qaoa", metric=q.metric.median(),
                             gap_pct=gap_pct(problem, q.metric.median(), opt), secs=q.wall_s.median(),
                             feasible=bool(q.feasible.all()), optimal=int((q.optimal == "yes").sum()),
                             runs=len(q), qubits=int(q.num_qubo_vars.iloc[0]), p_best=q.p_best.median()))
    out = pd.DataFrame(rows)
    out.to_csv(R / "benchmark_vs_clasicos.csv", index=False)
    return out


def figures(df):
    fams = list(FAM)
    specs = [("c1_brecha_metodos.png", "gap_mean", "brecha al óptimo ILP (%) — media de corridas",
              "Calidad de la solución devuelta frente al óptimo exacto", False),
             ("c2_tiempo_metodos.png", "secs_median", "tiempo por corrida (s) — mediana, log",
              "Tiempo de ejecución (simulador statevector para QAOA)", True),
             ("c3_exito_metodos.png", "exito", "corridas que alcanzan el óptimo (%)",
              "Éxito: fracción de corridas que devuelven el óptimo exacto", False)]
    for fname, col, ylabel, title, log in specs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
        for ax, fam in zip(axes, fams):
            d = df[df.family == fam]
            for m, label, color, ls in METH:
                h = d[d.method == m].sort_values("n")
                if h.empty or h[col if col != "exito" else "optimal"].isna().all():
                    continue
                y = 100.0 * h.optimal / h.runs if col == "exito" else h[col]
                ax.plot(h.n, y, marker="o", ms=4, color=color, ls=ls, label=label)
            ax.set_title(FAM[fam], fontsize=10, fontweight="bold")
            ax.set_xlabel("N (elementos de decisión)")
            if log:
                ax.set_yscale("log")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(fontsize=8)
        fig.suptitle(title, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    df = load_scaling()
    bm = load_benchmark()
    figures(df)
    pd.set_option("display.width", 200)
    print(bm.round(4).to_string(index=False))
    print()
    show = df[df.method.isin(["greedy", "sa", "aleatorio", "qaoa"])]
    print(show[["family", "n", "n_qubits", "method", "runs", "metric_median", "gap_mean", "optimal",
                "feasible", "secs_median", "p_opt_median"]].round(4).to_string(index=False))
    print(f"\nFiguras en {FIG}: c1, c2, c3")
