#!/usr/bin/env python3
"""Figuras del experimento confirmatorio (cap. 6 de la tesis).

Solo LEE las tablas de report/estadistica/confirmatorio y los CSV de
report/confirmatorio; no corre QAOA ni usa IBM.

Salida (report/figures/confirmatorio/):
  c1_amplificacion_vs_n.png   A frente a N, mediana y RIC entre instancias, por configuracion
  c2_ver_optimo_vs_n.png      probabilidad de ver el optimo en 8000 mediciones: QAOA frente al azar
  c3_cajas_logA.png           cajas de log10 A por configuracion; Base frente a F en N = 8 (H7)
  c4_hardware_vs_simulador.png  p_feasible y p_opt en hardware (IC de Wilson) frente al simulador (H5)
  c5_tiempo_vs_n.png          tiempo frente a N: QAOA simulado, ILP, SA y voraz
  x1_hardware_vs_uniforme.png EXPLORATORIO: como c4, con el muestreo uniforme como referencia

    python scripts/make_confirmatorio_figures.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "report" / "confirmatorio"
EST = ROOT / "report" / "estadistica" / "confirmatorio"
OUT = ROOT / "report" / "figures" / "confirmatorio"
OUT.mkdir(parents=True, exist_ok=True)

TXT, TXT2, GRID, GRAY = "#0b0b0b", "#52514e", "#e6e5e1", "#9a9890"
plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": TXT2,
    "xtick.color": TXT2, "ytick.color": TXT2, "axes.titlecolor": TXT, "axes.titlesize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "lines.linewidth": 1.8, "savefig.dpi": 200,
})
# configuracion -> (etiqueta, color, marcador, estilo). Mismos colores que las figuras de la Fase 4;
# el marcador y el estilo de linea distinguen las series sin depender del color.
CFG = {
    "Base": ("Base (con holguras)", GRAY, "D", (0, (4, 2))),
    "F": ("F", "#2a78d6", "o", "-"),
    "F+C": ("F+C", "#eb6834", "s", "-"),
    "F+C+I": ("F+C+I", "#0e7a52", "^", "-"),
}
FAM = {"tsm_aleatoria": "TSM, cobertura aleatoria", "tsm_ciclo": "TSM, cobertura en ciclo", "nrp": "NRP"}
C_HW, C_SIM = "#2a78d6", TXT


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("  ", OUT / name)


def per_instance():
    return pd.read_csv(EST / "resumen_por_instancia.csv")


# ---------------------------------------------------------------------------
def fig_amplification(res):
    main = res[res.data_seed.between(45, 49)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
    for ax, fam in zip(axes, ("tsm_aleatoria", "nrp")):
        for cfg in ("F", "F+C", "F+C+I"):
            lab, col, mk, ls = CFG[cfg]
            g = (10 ** main[(main.family == fam) & (main.cfg == cfg)].set_index("n").logA_median).groupby("n")
            q = g.quantile([0.25, 0.5, 0.75]).unstack()
            ax.fill_between(q.index, q[0.25], q[0.75], color=col, alpha=0.15, linewidth=0)
            ax.plot(q.index, q[0.5], color=col, marker=mk, ms=5, ls=ls, label=lab)
        ax.axhline(1, color=GRAY, lw=1.1, ls=(0, (2, 2)))
        if fam == "tsm_aleatoria":
            ax.text(0.99, 1.0, "azar (A = 1)", color=TXT2, fontsize=8, ha="right", va="bottom",
                    transform=ax.get_yaxis_transform())
        ax.set_yscale("log")
        ax.set_title(FAM[fam], loc="left")
        ax.set_xlabel("N (variables de decisión)")
        ax.set_xticks(sorted(main[main.family == fam].n.unique()))
    axes[0].set_ylabel("Amplificación A (mediana y RIC entre instancias)")  # A por instancia = 10^mediana(log10 A) de 10 semillas, como en las pruebas
    axes[0].legend(loc="upper left", title="Configuración")
    save(fig, "c1_amplificacion_vs_n.png")


def fig_see_optimum(res):
    main = res[res.data_seed.between(45, 49) & (res.cfg == "F+C+I")]
    fams = ("tsm_aleatoria", "tsm_ciclo", "nrp")
    short = {"tsm_aleatoria": "TSM aleatoria", "tsm_ciclo": "TSM ciclo", "nrp": "NRP"}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.2), sharey=True)
    for ax, fam in zip(axes, fams):
        g = main[main.family == fam].groupby("n")[["p_see_qaoa", "p_see_unif"]].mean()
        ax.plot(g.index, g.p_see_unif, color=GRAY, marker="o", ms=5, ls=(0, (4, 2)), label="Muestreo uniforme")
        ax.plot(g.index, g.p_see_qaoa, color=CFG["F+C+I"][1], marker="^", ms=5, label="QAOA (F+C+I)")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(short[fam], loc="left")
        ax.set_xlabel("N")
        ax.set_xticks(range(4, int(g.index.max()) + 1, 4))
    axes[0].set_ylabel("P(ver el óptimo en 8000 mediciones)\nmedia de 5 instancias")
    h, l = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.27, wspace=0.12)
    fig.legend(h, l, loc="lower center", ncol=2)
    save(fig, "c2_ver_optimo_vs_n.png")


def _boxes(ax, groups, cfgs, rng):
    data = [g.to_numpy() for g in groups]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=TXT, lw=1.6), whiskerprops=dict(color=TXT2), capprops=dict(color=TXT2))
    for patch, cfg, y, k in zip(bp["boxes"], cfgs, data, range(1, len(data) + 1)):
        col = CFG[cfg][1]
        patch.set(facecolor=col, alpha=0.22, edgecolor=col)
        ax.scatter(k + rng.uniform(-0.16, 0.16, len(y)), y, s=11, color=col, alpha=0.8, linewidths=0, zorder=3)
    ax.axhline(0, color=GRAY, lw=1.1, ls=(0, (2, 2)))
    ax.set_xticks(range(1, len(cfgs) + 1))
    ax.set_xticklabels([f"{CFG[c][0]}\n(n = {len(d)})" for c, d in zip(cfgs, data)])
    ax.grid(axis="x", visible=False)


def fig_boxes(res):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw=dict(width_ratios=[3, 2]))
    a = res[(res.family == "tsm_aleatoria") & res.data_seed.between(45, 49) & (res.n >= 8)]
    cfgs = ["F", "F+C", "F+C+I"]
    _boxes(axes[0], [a[a.cfg == c].logA_median for c in cfgs], cfgs, rng)
    axes[0].set_title("TSM aleatoria, N = 8 a 22", loc="left")
    axes[0].set_ylabel("log$_{10}$ A por instancia (mediana de 10 semillas)")
    b = res[(res.family == "tsm_aleatoria") & (res.n == 8)]
    cfgs = ["Base", "F"]
    _boxes(axes[1], [b[b.cfg == c].logA_median for c in cfgs], cfgs, rng)
    axes[1].set_title("Codificación, N = 8 (H7)", loc="left")
    for ax in axes:
        ax.text(0.99, 0.0, "azar", color=TXT2, fontsize=8, ha="right", va="bottom", transform=ax.get_yaxis_transform())
    save(fig, "c3_cajas_logA.png")


# ---------------------------------------------------------------------------
def short_label(key):
    m = re.match(r"(tsm|nrp):(.+)\[(.+)\]", key)
    prob, inst, cfg = m.groups()
    s = re.match(r"tsm_(aleatoria|ciclo)_n(\d+)(?:_d(\d+))?", inst)
    if s:
        fam, n, d = s.groups()
        name = f"TSM {fam} N={n}" + (f" (semilla {d})" if d else "")
    else:
        # nombres del documento (cap. 5, tabla de instancias de referencia); en el código: toy, toy2, demo
        name = {("tsm", "toy"): "T5", ("tsm", "toy2"): "T10", ("tsm", "demo"): "T12",
                ("nrp", "toy"): "R10", ("nrp", "demo"): "R12", ("nrp", "toy2"): "R30"}[(prob, inst)]
    return name, cfg


def hardware_table():
    hw = pd.read_csv(EST / "hardware_wilson.csv")
    uni = pd.read_csv(EST / "hardware_vs_uniforme.csv").set_index("instancia")
    order, groups = [], []
    keys = hw.instancia.drop_duplicates().tolist()
    blocks = [("Réplicas de E3 (F+C+I)", [k for k in keys if "[F+C+I]" in k and "_d" not in k]),
              ("Instancias nuevas (F+C+I)", [k for k in keys if "_d" in k]),
              ("Codificación con holguras (Base)", [k for k in keys if "[Base]" in k])]
    for title, ks in blocks:
        groups.append((title, len(order), len(ks)))
        order += ks
    return hw, uni, order, groups


def fig_hardware(with_uniform, name):
    hw, uni, order, groups = hardware_table()
    n = len(order)
    ypos = {}
    y = 0.0
    for title, start, count in groups:
        for k in order[start:start + count]:
            ypos[k] = y
            y += 1
        y += 0.9
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.3 * n + 1.9), sharey=True)
    for ax, measure, xl in zip(axes, ("p_feasible", "p_opt"), ("p$_{feasible}$", "p$_{opt}$ (escala logarítmica)")):
        d = hw[hw.medida == measure].set_index("instancia")
        logx = measure == "p_opt"
        xmin = 1e-5 if not with_uniform else 5e-8
        for k in order:
            r, yy = d.loc[k], ypos[k]
            incon = isinstance(r.conclusion, str) and r.conclusion.startswith("no concluyente")
            col = GRAY if incon else C_HW
            lo, hi, p = r.wilson_inf, r.wilson_sup, r.p_hw
            if logx and r.k == 0:                       # sin optimos: solo la cota superior de Wilson
                ax.plot([xmin, hi], [yy, yy], color=col, lw=1.6)
                ax.plot([hi], [yy], marker="|", ms=7, color=col)
                ax.plot([xmin * 1.25], [yy], marker="<", ms=5, color=col)
            else:
                ax.plot([max(lo, xmin) if logx else lo, hi], [yy, yy], color=col, lw=1.6)
                ax.plot([p], [yy], marker="o", ms=5.5, color=col, mec="white", mew=0.8, zorder=3)
            if pd.notna(r.p_sim):
                ax.plot([r.p_sim], [yy], marker="D", ms=5, mfc="white", mec=C_SIM, mew=1.2, zorder=4)
            if with_uniform:
                u = uni.loc[k, f"unif_{measure}"]
                if pd.notna(u):
                    ax.plot([u], [yy], marker="|", ms=11, mew=1.8, color="#eb6834", zorder=2)
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(xmin, 0.05)
        else:
            ax.set_xlim(0, 1)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    labels = [short_label(k)[0] for k in order]
    axes[0].set_yticks([ypos[k] for k in order])
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    for title, start, count in groups:
        axes[0].text(0.0, ypos[order[start]] - 0.75, title, fontsize=9, color=TXT, fontweight="bold",
                     transform=axes[0].get_yaxis_transform(), ha="left", va="center")
    axes[0].set_ylim(max(ypos.values()) + 0.7, -1.3)
    H = [plt.Line2D([], [], color=C_HW, marker="o", ms=5.5, mec="white", lw=1.6, label="Hardware (IC de Wilson 95 %)"),
         plt.Line2D([], [], color=C_SIM, marker="D", ms=5, mfc="white", lw=0, label="Simulador (exacto)"),
         plt.Line2D([], [], color=GRAY, marker="o", ms=5.5, mec="white", lw=1.6,
                    label="p$_{opt}$ no concluyente (< 10 óptimos esperados)"),
         plt.Line2D([], [], color=C_HW, marker="<", ms=5, lw=1.6, label="0 óptimos observados (cota superior)")]
    if with_uniform:
        H.append(plt.Line2D([], [], color="#eb6834", marker="|", ms=11, mew=1.8, lw=0, label="Muestreo uniforme"))
    fig.subplots_adjust(bottom=0.14, top=0.99, wspace=0.2)
    fig.legend(handles=H, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.075))
    save(fig, name)


# ---------------------------------------------------------------------------
def fig_time():
    dfs = []
    for p in sorted(RUNS.glob("ds*/scale_*.csv")):
        d = pd.read_csv(p, usecols=["family", "n", "config", "method", "secs", "data_seed"], low_memory=False)
        dfs.append(d)
    d = pd.concat(dfs, ignore_index=True)
    d = d[d.data_seed.between(45, 49)]
    series = [("qaoa", "FCI", "QAOA simulado (F+C+I)", CFG["F+C+I"][1], "^", "-"),
              ("sa", "FCI", "Recocido simulado", "#eb6834", "s", "-"),
              ("ilp", "FCI", "ILP", "#2a78d6", "o", "-"),
              ("greedy", "FCI", "Voraz", TXT2, "v", (0, (4, 2)))]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
    for ax, fam in zip(axes, ("tsm_aleatoria", "nrp")):
        f = d[d.family == fam]
        if fam == "tsm_aleatoria":
            g = f[(f.method == "qaoa") & (f.config == "base")].groupby("n").secs.median()
            ax.plot(g.index, g.values, color=GRAY, marker="D", ms=5, ls=(0, (4, 2)), label="QAOA simulado (Base)")
        for method, cfg, lab, col, mk, ls in series:
            g = f[(f.method == method) & (f.config == cfg)].groupby("n").secs.median()
            ax.plot(g.index, g.values, color=col, marker=mk, ms=5, ls=ls, label=lab)
        ax.set_yscale("log")
        ax.set_title(FAM[fam], loc="left")
        ax.set_xlabel("N (variables de decisión)")
        ax.set_xticks(sorted(f.n.unique()))
    axes[0].set_ylabel("Tiempo por ejecución (s, mediana)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.91, 0.5))
    save(fig, "c5_tiempo_vs_n.png")


def main():
    res = per_instance()
    print("Figuras en:")
    fig_amplification(res)
    fig_see_optimum(res)
    fig_boxes(res)
    fig_hardware(False, "c4_hardware_vs_simulador.png")
    fig_time()
    fig_hardware(True, "x1_hardware_vs_uniforme.png")


if __name__ == "__main__":
    main()
