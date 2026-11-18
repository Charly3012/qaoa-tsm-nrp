"""Utilidades de carga, métricas y validación para NRP (Next Release Problem).

El dataset NRP consta de:
- nrp_{name}_features.csv    : índice = feature (F*), columnas Nombre, Costo, Valor.
- nrp_{name}_dependencies.csv: columnas Origen, Relación, Destino, Descripción.
  Relación: 'REQUIRES' (x_origen <= x_destino) o 'EXCLUDES' (x_origen + x_destino <= 1).

La función objetivo es maximizar el valor total seleccionado, respetando el
presupuesto y las dependencias lógicas.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_nrp(dataset_name, data_dir=None):
    """Carga features y dependencias. Devuelve (df_features, df_deps)."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    feat_path = data_dir / f"nrp_{dataset_name}_features.csv"
    deps_path = data_dir / f"nrp_{dataset_name}_dependencies.csv"

    if not feat_path.exists() or not deps_path.exists():
        raise FileNotFoundError(
            f"No se encontraron datasets '{dataset_name}' en {data_dir}. "
            f"Genéralo con scripts/generate_dataset.py"
        )

    df_features = pd.read_csv(feat_path, index_col=0)
    df_features.index = df_features.index.astype(str).str.strip()

    df_deps = pd.read_csv(deps_path)
    for col in ("Origen", "Destino"):
        df_deps[col] = df_deps[col].astype(str).str.strip()
    return df_features, df_deps


def evaluate_selection(selected, df_features, df_deps, budget):
    """Evalúa una selección de features.

    Devuelve dict con costo total, valor total, factibilidad y valor objetivo
    (en forma de minimización, igual que el modelo cuántico: -valor).
    """
    selected = list(selected)
    total_cost = float(df_features.loc[selected, "Costo"].sum()) if selected else 0.0
    total_value = float(df_features.loc[selected, "Valor"].sum()) if selected else 0.0

    selected_set = set(selected)
    valid = True
    for _, row in df_deps.iterrows():
        orig, rel, dest = row["Origen"], row["Relacion"], row["Destino"]
        if orig not in df_features.index or dest not in df_features.index:
            continue
        xo, xd = int(orig in selected_set), int(dest in selected_set)
        if rel == "REQUIRES" and xo == 1 and xd == 0:
            valid = False
        elif rel == "EXCLUDES" and xo == 1 and xd == 1:
            valid = False

    budget_ok = total_cost <= budget
    feasible = valid and budget_ok

    return {
        "selected": selected,
        "selected_case_num": len(selected),
        "total_cost": total_cost,
        "total_value": total_value,
        "budget": budget,
        "budget_used": f"{total_cost:.2f}/{budget}",
        "feasible": feasible,
        "objective": -total_value,
    }


def is_feasible(selected, df_features, df_deps, budget):
    """True si la selección respeta presupuesto y dependencias."""
    selected = set(selected)
    total_cost = float(df_features.loc[list(selected), "Costo"].sum())
    if total_cost > budget:
        return False
    for _, row in df_deps.iterrows():
        orig, rel, dest = row["Origen"], row["Relacion"], row["Destino"]
        if orig not in df_features.index or dest not in df_features.index:
            continue
        xo, xd = int(orig in selected), int(dest in selected)
        if rel == "REQUIRES" and xo == 1 and xd == 0:
            return False
        if rel == "EXCLUDES" and xo == 1 and xd == 1:
            return False
    return True