#!/usr/bin/env python3
"""Genera las graficas del reporte leyendo los resultados reales (results/*.csv).

No hay numeros hardcodeados: se leen todos los summary.csv generados por los
benchmarks de TSM y NRP.
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLORS = {
    "ilp": "#1f77b4",
    "greedy": "#2ca02c",
    "sa": "#ff7f0e",
    "qaoa_sim": "#9467bd",
    "qaoa_ibm": "#d62728",
}
METHOD_ORDER = ["ilp", "greedy", "sa", "qaoa_sim", "qaoa_ibm"]


def load(base):
    rows = []
    for sp in glob.glob(f"{base}/results/*/*/0/summary.csv"):
        rows.append(pd.read_csv(sp).iloc[0].to_dict())
    return pd.DataFrame(rows)


def pivot(df, value, methods):
    """Devuelve {dataset: {method: valor}} para los metodos presentes."""
    out = {}
    for _, r in df.iterrows():
        ds = r["dataset"]
        if r["method"] not in methods:
            continue
        out.setdefault(ds, {})[r["method"]] = r[value]
    return out


def _bar_group(ax, datasets, methods, data, ylabel, title, annotate=True):
    x = np.arange(len(datasets))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = [data.get(d, {}).get(m) for d in datasets]
        vals = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w,
                      label=m.replace("_", " "), color=COLORS[m])
        if annotate:
            for b, v in zip(bars, vals):
                if v:
                    ax.text(b.get_x() + b.get_width() / 2, v, str(v),
                            ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return ax


# ----------------------------------------------------------------------
# TSM
# ----------------------------------------------------------------------
tsm = load(ROOT / "TSM")
tsm_datasets = ["toy", "toy2", "demo"]
tsm_sel = pivot(tsm, "selected_case_num", METHOD_ORDER)

# Figura 1: seleccion de tests por metodo (clasicos + qaoa)
fig, ax = plt.subplots(figsize=(8, 5))
_bar_group(ax, tsm_datasets, ["ilp", "greedy", "sa"], tsm_sel,
           "Tests seleccionados (menos = mejor)",
           "TSM - Tests seleccionados por metodo")
plt.tight_layout()
plt.savefig(FIG / "tsm_selected.png", dpi=150)
plt.close()

# Figura 2: QAOA vs clasicos (todos los metodos)
fig, ax = plt.subplots(figsize=(9, 5))
_bar_group(ax, tsm_datasets, METHOD_ORDER, tsm_sel,
           "Tests seleccionados", "TSM - QAOA vs clasicos")
plt.tight_layout()
plt.savefig(FIG / "tsm_qaoa_all.png", dpi=150)
plt.close()

# Figura 3: tiempo de ejecucion (clasicos, escala log)
tsm_time = pivot(tsm, "execution_time", ["ilp", "greedy", "sa"])
fig, ax = plt.subplots(figsize=(7, 5))
for m in ["ilp", "greedy", "sa"]:
    vals = [max(tsm_time.get(d, {}).get(m, 1e-4), 1e-4) for d in tsm_datasets]
    ax.plot(tsm_datasets, vals, marker="o", label=m.upper(), color=COLORS[m])
ax.set_yscale("log")
ax.set_ylabel("Tiempo de ejecucion (s, log)")
ax.set_title("TSM - Tiempo de ejecucion por metodo")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(FIG / "tsm_time.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# NRP
# ----------------------------------------------------------------------
nrp = load(ROOT / "NRP")
nrp_datasets = ["toy", "toy2", "demo"]
nrp_val = pivot(nrp, "total_value", METHOD_ORDER)

# Figura 4: valor total por metodo (clasicos)
fig, ax = plt.subplots(figsize=(8, 5))
_bar_group(ax, nrp_datasets, ["ilp", "greedy", "sa"], nrp_val,
           "Valor total (mas = mejor)",
           "NRP - Valor total por metodo")
plt.tight_layout()
plt.savefig(FIG / "nrp_value.png", dpi=150)
plt.close()

# Figura 5: QAOA vs clasicos en toy y demo (escalas comparables)
fig, ax = plt.subplots(figsize=(8, 5))
_bar_group(ax, ["toy", "demo"], METHOD_ORDER, nrp_val,
           "Valor total", "NRP - QAOA vs clasicos (toy, demo)")
plt.tight_layout()
plt.savefig(FIG / "nrp_qaoa.png", dpi=150)
plt.close()

# Figura 6: tiempo de ejecucion (clasicos, escala log)
nrp_time = pivot(nrp, "execution_time", ["ilp", "greedy", "sa"])
fig, ax = plt.subplots(figsize=(7, 5))
for m in ["ilp", "greedy", "sa"]:
    vals = [max(nrp_time.get(d, {}).get(m, 1e-4), 1e-4) for d in nrp_datasets]
    ax.plot(nrp_datasets, vals, marker="o", label=m.upper(), color=COLORS[m])
ax.set_yscale("log")
ax.set_ylabel("Tiempo de ejecucion (s, log)")
ax.set_title("NRP - Tiempo de ejecucion por metodo")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(FIG / "nrp_time.png", dpi=150)
plt.close()

print("Figuras generadas en", FIG)
