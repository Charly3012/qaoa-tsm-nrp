#!/usr/bin/env python3
"""Figuras adicionales de la Fase 4 (experimento E3).

  f5: benchmark, p_best mediana Base vs F+C+I (TSM toy/toy2/demo, NRP toy/demo),
      desde report/fase4/resumen_benchmark.csv.
  f6: diagnostico del paisaje en N=14: masa de QAOA p=1 en las k energias mas bajas
      relativa al estado uniforme (TSM aleatoria, TSM ciclo, NRP), desde
      report/fase4/nrp/diag_landscape_n14.txt.

Salida: report/figures/fase4/f5_benchmark_pbest.png, f6_paisaje_n14.png.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures" / "fase4"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def f5_benchmark():
    df = pd.read_csv(ROOT / "report" / "fase4" / "resumen_benchmark.csv")
    order = [("tsm", "toy"), ("tsm", "toy2"), ("tsm", "demo"), ("nrp", "toy"), ("nrp", "demo")]
    cfgs = [("base", "Base: holguras + energía + aleatorio", "#9e9e9e"),
            ("F", "F: sin holguras", "#1f77b4"),
            ("FC_a0.25", "F+C: CVaR α=0.25", "#ff7f0e"),
            ("FCI_a0.25", "F+C+I: malla γ×β", "#2ca02c")]
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.2
    for k, (cfg, label, color) in enumerate(cfgs):
        ys, xs = [], []
        for i, (prob, ds) in enumerate(order):
            row = df[(df.problem == prob) & (df.dataset == ds) & (df.cfg == cfg)]
            if row.empty:
                continue
            xs.append(i + (k - 1.5) * w)
            ys.append(100 * float(row.p_best_median.iloc[0]))
        bars = ax.bar(xs, ys, w, label=label, color=color)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y * 1.15, f"{y:.2g}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{p.upper()} {d}" for p, d in order])
    ax.set_ylabel("p_best (%) — mediana de 3 semillas, log")
    ax.set_title("Benchmark: fracción de mediciones que corresponde a la solución devuelta (p=1)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "f5_benchmark_pbest.png", dpi=160)


def f6_landscape():
    txt = (ROOT / "report" / "fase4" / "nrp" / "diag_landscape_n14.txt").read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=\w+ N=14:)", txt)
    data = {}
    for b in blocks:
        m = re.match(r"(\w+) N=14", b)
        if not m:
            continue
        fam = m.group(1)
        mult = {int(k): float(x) for k, x in re.findall(r"en\s+(\d+) menores energias: [\d.e+-]+\s+\(uniforme [\d.e+-]+, x([\d.]+)\)", b)}
        A = float(re.search(r"A=([\d.]+)", b).group(1))
        data[fam] = (mult, A)
    labels = {"tsm_aleatoria": "TSM aleatoria (k=3)", "tsm_ciclo": "TSM ciclo (k=2)", "nrp": "NRP"}
    colors = {"tsm_aleatoria": "#1f77b4", "tsm_ciclo": "#2ca02c", "nrp": "#d62728"}
    ks = [1, 10, 50, 500]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.25
    for j, fam in enumerate(["tsm_aleatoria", "tsm_ciclo", "nrp"]):
        mult, A = data[fam]
        xs = np.arange(len(ks)) + (j - 1) * w
        ys = [mult[k] for k in ks]
        bars = ax.bar(xs, ys, w, label=f"{labels[fam]} — A(óptimo) = {A:.0f}", color=colors[fam])
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y + 0.5, f"×{y:.0f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(1, color="k", ls=":", lw=1)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"{k} más bajas" for k in ks])
    ax.set_xlabel("energías del QUBO (ordenadas de menor a mayor)")
    ax.set_ylabel("masa de QAOA p=1 / masa del estado uniforme")
    ax.set_title("N=14, λ congelados: dónde concentra probabilidad QAOA p=1")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "f6_paisaje_n14.png", dpi=160)


if __name__ == "__main__":
    f5_benchmark()
    f6_landscape()
    print(f"Figuras en {FIG}: f5_benchmark_pbest.png, f6_paisaje_n14.png")
