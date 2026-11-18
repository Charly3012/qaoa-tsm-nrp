#!/usr/bin/env python3
"""Figuras de la Fase 2 (escalamiento de N) a partir de report/scale_n/scale_*.csv.

Salida: report/figures/scale_n/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "report" / "scale_n"
OUT = ROOT / "report" / "figures" / "scale_n"
OUT.mkdir(parents=True, exist_ok=True)

# Paleta categorica validada (slots 1-3) + gris neutro para la linea base.
C1, C2, C3, C_UNI = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890"
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TXT2,
    "xtick.color": TXT2, "ytick.color": TXT2, "axes.titlecolor": TXT,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "lines.linewidth": 2,
})
FAM = {"tsm_ciclo": ("TSM cobertura en ciclo (sin holguras)", C1),
       "tsm_aleatoria": ("TSM cobertura aleatoria (k=3)", C2),
       "nrp": ("NRP (budget 40 %)", C3)}


def load():
    dfs = []
    for f in FAM:
        p = SRC / f"scale_{f}.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True)


def style(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(6, 0), va="center",
                fontsize=8, color=TXT)


def fig_qubits(df):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    q = df.drop_duplicates(["family", "n"])
    for f, (lab, c) in FAM.items():
        d = q[q.family == f].sort_values("n")
        if d.empty:
            continue
        ax.plot(d.n, d.n_qubo, color=c, marker="o", markersize=5, label=lab)
        end_label(ax, d.n.iloc[-1], d.n_qubo.iloc[-1], f"{int(d.n_qubo.iloc[-1])} q", c)
    nmax = q.n.max()
    ax.plot([0, nmax], [0, nmax], color=C_UNI, linestyle=(0, (4, 2)), linewidth=1.4,
            label="qubits = N (sin holguras)")
    ax.axhline(24, color=TXT2, linewidth=1, linestyle=":")
    ax.text(1, 24.5, "límite statevector en este equipo (~24 q)", fontsize=8, color=TXT2)
    ax.set_xlabel("N (tests / features = variables de decisión)")
    ax.set_ylabel("qubits del QUBO")
    ax.set_title("Qubits necesarios vs N: las holguras dominan el crecimiento", loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "f1_qubits_vs_n.png", dpi=160)
    plt.close(fig)


def fig_pbest(df):
    fams = [f for f in FAM if f in set(df.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, fams):
        lab, c = FAM[f]
        d = df[df.family == f]
        u = d[d.method == "uniforme"].sort_values("n")
        qa = d[d.method == "qaoa"].groupby("n").agg(p_opt=("p_opt", "mean"), lo=("p_opt", "min"),
                                                    hi=("p_opt", "max")).reset_index()
        ax.plot(u.n, u.p_opt * 100, color=C_UNI, marker="o", markersize=4, label="estado uniforme")
        if not qa.empty:
            ax.plot(qa.n, qa.p_opt * 100, color=c, marker="o", markersize=5,
                    label="QAOA p=1 (media; banda = mín–máx de 3 semillas)")
            ax.fill_between(qa.n, qa.lo * 100, qa.hi * 100, color=c, alpha=0.15, linewidth=0)
        ax.set_yscale("log")
        ax.set_title(lab, loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("N")
        style(ax)
    axes[0].set_ylabel("probabilidad de medir el óptimo (%) — log")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("p_opt de QAOA frente al estado uniforme al crecer N", x=0.01, ha="left",
                 fontsize=12, color=TXT)
    fig.tight_layout()
    fig.savefig(OUT / "f2_popt_vs_n.png", dpi=160)
    plt.close(fig)


def fig_success(df):
    fams = [f for f in FAM if f in set(df.family)]
    meths = [("ilp", "ILP", TXT), ("sa", "SA (coincide con ILP: 0 %)", C3), ("greedy", "Greedy", C_UNI),
             ("qaoa", "QAOA p=1", C1), ("aleatorio", "aleatorio + filtro", C2)]
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, fams):
        d = df[df.family == f].copy()
        d["gap"] = np.where(d.problem == "tsm", (d.metric - d.optimum) / d.optimum,
                            (d.optimum - d.metric) / d.optimum) * 100
        for m, lab, c in meths:
            g = d[d.method == m].groupby("n").gap.mean().reset_index()
            if g.empty:
                continue
            ls = (0, (4, 2)) if m == "aleatorio" else "-"
            ax.plot(g.n, g.gap, color=c, linestyle=ls, marker="o", markersize=4, label=lab)
        ax.set_title(FAM[f][0], loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("N")
        style(ax)
    axes[0].set_ylabel("brecha al óptimo ILP (%) — 0 = óptimo")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Calidad de la solución al crecer N (media de semillas)", x=0.01, ha="left",
                 fontsize=12, color=TXT)
    fig.tight_layout()
    fig.savefig(OUT / "f3_brecha_vs_n.png", dpi=160)
    plt.close(fig)


def fig_time(df):
    fams = [f for f in FAM if f in set(df.family)]
    meths = [("ilp", "ILP", TXT), ("sa", "SA", C3), ("qaoa", "QAOA p=1 (simulado)", C1)]
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, fams):
        d = df[df.family == f]
        for m, lab, c in meths:
            g = d[d.method == m].groupby("n").secs.mean().reset_index()
            if g.empty:
                continue
            ax.plot(g.n, g.secs, color=c, marker="o", markersize=4, label=lab)
        ax.set_yscale("log")
        ax.set_title(FAM[f][0], loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("N")
        style(ax)
    axes[0].set_ylabel("tiempo (s) — log")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Tiempo de ejecución al crecer N", x=0.01, ha="left", fontsize=12, color=TXT)
    fig.tight_layout()
    fig.savefig(OUT / "f4_tiempo_vs_n.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    df = load()
    fig_qubits(df); fig_pbest(df); fig_success(df); fig_time(df)
    print(f"Figuras en {OUT}")
