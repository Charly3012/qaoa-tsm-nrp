#!/usr/bin/env python3
"""Figuras y tablas de la Fase 4 (TSM sin holguras + CVaR + inicializacion).

Entrada:
  report/fase4/scale_<familia>_<config>.csv   (scripts/scale_n.py --config ...)
  report/scale_n/scale_<familia>.csv          (Fase 2 = configuracion Base)
  report/fase4/sweep_tsm_<config>.csv         (scripts/sweep_qaoa.py, benchmark)
Salida:
  report/figures/fase4/f1_amplificacion_vs_n.png   A = p_opt(QAOA)/p_opt(uniforme), mediana y max
  report/figures/fase4/f2_popt_vs_n.png            p_opt (mediana) vs uniforme
  report/figures/fase4/f3_brecha_vs_n.png          brecha al optimo (media)
  report/figures/fase4/f4_exito_vs_n.png           semillas que alcanzan el optimo
  report/fase4/resumen_escalamiento.csv            tabla agregada por familia/config/N
  report/fase4/resumen_benchmark.csv               tabla agregada por dataset/config
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "report" / "fase4"
SRC_BASE = ROOT / "report" / "scale_n"
OUT = ROOT / "report" / "figures" / "fase4"
OUT.mkdir(parents=True, exist_ok=True)

TXT, TXT2, GRID, C_UNI = "#0b0b0b", "#52514e", "#e6e5e1", "#9a9890"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TXT2,
    "xtick.color": TXT2, "ytick.color": TXT2, "axes.titlecolor": TXT,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "lines.linewidth": 1.8,
})
FAM = {"tsm_aleatoria": "TSM cobertura aleatoria (k=3)", "tsm_ciclo": "TSM cobertura en ciclo (k=2)",
       "nrp": "NRP (presupuesto 40 %)"}
# config -> (etiqueta, color, estilo de linea)
CFG = {
    "base": ("Base: holguras + energía + aleatorio", C_UNI, (0, (4, 2))),
    "F": ("F: sin holguras", "#2a78d6", "-"),
    "FC_a0.25": ("F+C: CVaR α=0.25", "#eb6834", "-"),
    "FC_a0.1": ("F+C: CVaR α=0.1", "#b8471c", (0, (4, 2))),
    "FCI_grid_a0.25": ("F+C+I: malla γ×β", "#1baf7a", "-"),
    "FCI_transfer_a0.25": ("F+C+I: transferencia N−2", "#0e7a52", (0, (4, 2))),
}


def load_scale():
    dfs = []
    for fam in FAM:
        p = SRC_BASE / f"scale_{fam}.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["config"] = "base"
            dfs.append(d)
        src = SRC / "nrp" if fam == "nrp" else SRC
        for p in sorted(src.glob(f"scale_{fam}_*.csv")):
            m = re.match(rf"scale_{fam}_(.+)\.csv", p.name)
            d = pd.read_csv(p)
            d["config"] = m.group(1)
            dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["seed"] = df["seed"].astype(str)
    # Brecha relativa al optimo con el signo correcto segun el problema
    # (TSM minimiza tests; NRP maximiza valor).
    df["gap_pct"] = np.where(df.problem == "nrp", (df.optimum - df.metric) / df.optimum,
                             (df.metric - df.optimum) / df.optimum) * 100
    return df


def aggregate(df):
    """Una fila por (familia, config, N) con las metricas de QAOA frente al uniforme."""
    rows = []
    for (fam, cfg), d in df.groupby(["family", "config"]):
        uni = d[d.method == "uniforme"].set_index("n")
        ale = d[d.method == "aleatorio"].groupby("n")
        for n, q in d[d.method == "qaoa"].groupby("n"):
            p_u = float(uni.p_opt.get(n, np.nan))
            A = q.p_opt / p_u if p_u and not np.isnan(p_u) else np.nan
            gap = q.gap_pct
            rows.append(dict(
                family=fam, config=cfg, n=int(n), n_qubo=int(q.n_qubo.iloc[0]),
                seeds=len(q), optimal=int(q.optimal.sum()),
                aleatorio_optimal=int(ale.optimal.sum().get(n, 0)) if n in ale.groups else np.nan,
                p_opt_uniform=p_u, p_opt_median=float(q.p_opt.median()), p_opt_max=float(q.p_opt.max()),
                A_median=float(np.median(A)), A_max=float(np.max(A)),
                p_feasible_median=float(q.p_feasible.median()),
                p_feasible_uniform=float(uni.p_feasible.get(n, np.nan)),
                gap_mean=float(gap.mean()), secs_mean=float(q.secs.mean()),
            ))
    return pd.DataFrame(rows).sort_values(["family", "config", "n"])


def style(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _configs(agg):
    return [c for c in CFG if c in set(agg.config)]


def fig_amplification(agg):
    fams = [f for f in FAM if f in set(agg.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(5.2 * len(fams), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        d = agg[agg.family == fam]
        for cfg in _configs(d):
            lab, c, ls = CFG[cfg]
            g = d[d.config == cfg].sort_values("n")
            ax.plot(g.n, g.A_median.clip(lower=1e-3), color=c, linestyle=ls, marker="o",
                    markersize=4, label=lab)
            ax.scatter(g.n, g.A_max.clip(lower=1e-3), color=c, marker="^", s=18, alpha=0.6)
        ax.axhline(1, color=TXT2, linewidth=1, linestyle=":")
        ax.set_yscale("log")
        ax.set_title(FAM[fam], loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("N")
        style(ax)
    axes[0].set_ylabel("amplificación A = p_opt(QAOA) / p_opt(uniforme) — log")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Amplificación del óptimo por configuración (línea = mediana, ▲ = máximo de semillas)",
                 x=0.01, ha="left", fontsize=11, color=TXT)
    fig.tight_layout()
    fig.savefig(OUT / "f1_amplificacion_vs_n.png", dpi=160)
    plt.close(fig)


def fig_popt(agg):
    fams = [f for f in FAM if f in set(agg.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(5.2 * len(fams), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        d = agg[agg.family == fam]
        u = d.drop_duplicates("n").sort_values("n")
        ax.plot(u.n, u.p_opt_uniform * 100, color=TXT2, marker=".", linewidth=1, label="estado uniforme")
        for cfg in _configs(d):
            lab, c, ls = CFG[cfg]
            g = d[d.config == cfg].sort_values("n")
            ax.plot(g.n, g.p_opt_median.clip(lower=1e-12) * 100, color=c, linestyle=ls, marker="o",
                    markersize=4, label=lab)
        ax.axhline(100 / 8000, color=TXT2, linewidth=0.8, linestyle=":")
        ax.text(ax.get_xlim()[0] + 0.3, 100 / 8000 * 1.3, "1/8000 muestras", fontsize=7, color=TXT2)
        ax.set_yscale("log")
        ax.set_title(FAM[fam], loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("N")
        style(ax)
    axes[0].set_ylabel("p_opt (%) — mediana de semillas, log")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Probabilidad de medir el óptimo por configuración", x=0.01, ha="left",
                 fontsize=11, color=TXT)
    fig.tight_layout()
    fig.savefig(OUT / "f2_popt_vs_n.png", dpi=160)
    plt.close(fig)


def fig_gap(agg, df):
    fams = [f for f in FAM if f in set(agg.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(5.2 * len(fams), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        d = agg[agg.family == fam]
        for cfg in _configs(d):
            lab, c, ls = CFG[cfg]
            g = d[d.config == cfg].sort_values("n")
            ax.plot(g.n, g.gap_mean, color=c, linestyle=ls, marker="o", markersize=4, label=lab)
        a = df[(df.family == fam) & (df.method == "aleatorio")].drop_duplicates(["n", "seed"])
        a = a.groupby("n").gap_pct.mean().reset_index().rename(columns={"gap_pct": "gap"})
        ax.plot(a.n, a.gap, color=TXT, linestyle=(0, (1, 2)), marker="x", markersize=4,
                label="aleatorio + filtro")
        ax.set_title(FAM[fam], loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("N")
        style(ax)
    axes[0].set_ylabel("brecha al óptimo ILP (%) — media de semillas")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Calidad de la solución devuelta por configuración", x=0.01, ha="left",
                 fontsize=11, color=TXT)
    fig.tight_layout()
    fig.savefig(OUT / "f3_brecha_vs_n.png", dpi=160)
    plt.close(fig)


def fig_success(agg):
    fams = [f for f in FAM if f in set(agg.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(5.2 * len(fams), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        d = agg[agg.family == fam]
        for cfg in _configs(d):
            lab, c, ls = CFG[cfg]
            g = d[d.config == cfg].sort_values("n")
            ax.plot(g.n, g.optimal / g.seeds * 100, color=c, linestyle=ls, marker="o",
                    markersize=4, label=lab)
        a = d.drop_duplicates("n").sort_values("n")
        ax.plot(a.n, a.aleatorio_optimal / 3 * 100, color=TXT, linestyle=(0, (1, 2)), marker="x",
                markersize=4, label="aleatorio + filtro")
        ax.set_ylim(-5, 105)
        ax.set_title(FAM[fam], loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("N")
        style(ax)
    axes[0].set_ylabel("corridas que alcanzan el óptimo (%)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Éxito por configuración (óptimo encontrado tras el filtro clásico)",
                 x=0.01, ha="left", fontsize=11, color=TXT)
    fig.tight_layout()
    fig.savefig(OUT / "f4_exito_vs_n.png", dpi=160)
    plt.close(fig)


def benchmark():
    dfs = []
    for p in sorted(SRC.glob("sweep_tsm_*.csv")) + sorted((SRC / "nrp").glob("sweep_nrp_*.csv")):
        d = pd.read_csv(p)
        if not d.empty:
            dfs.append(d)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df["cfg"] = df.config + np.where(df.alpha.notna() & (df.alpha != ""),
                                     "_a" + df.alpha.astype(str).str.replace(r"\.0$", "", regex=True), "")
    g = df.groupby(["problem", "dataset", "cfg"]).agg(
        seeds=("seed", "count"), optimal=("optimal", lambda s: int((s == "yes").sum())),
        qubits=("num_qubo_vars", "first"), p_best_median=("p_best", "median"),
        p_best_max=("p_best", "max"), p_feasible_median=("p_feasible", "median"),
        wall_s=("wall_s", "mean")).reset_index()
    return g


if __name__ == "__main__":
    df = load_scale()
    agg = aggregate(df)
    agg.to_csv(SRC / "resumen_escalamiento.csv", index=False)
    fig_amplification(agg); fig_popt(agg); fig_gap(agg, df); fig_success(agg)
    b = benchmark()
    if b is not None:
        b.to_csv(SRC / "resumen_benchmark.csv", index=False)
    print(f"Figuras en {OUT}; tablas en {SRC}")
