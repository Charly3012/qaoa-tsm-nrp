"""Utilidades de carga, métricas y validación para TSM (Test Suite Minimization).

El dataset TSM consta de:
- tsm_{name}_coverage.csv : matriz de cobertura (índice = test T*, columnas = requisito R*).
- tsm_{name}_tests.csv   : columnas Test, time (una fila por test).

La función objetivo es minimizar el número de tests seleccionados (con
desempate por tiempo total de ejecución).
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

TIME_WEIGHT = 0.001


def load_tsm(dataset_name, data_dir=None):
    """Carga cobertura y tiempos/tasa. Devuelve (df_coverage, df_tests)."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    cov_path = data_dir / f"tsm_{dataset_name}_coverage.csv"
    tests_path = data_dir / f"tsm_{dataset_name}_tests.csv"

    if not cov_path.exists() or not tests_path.exists():
        raise FileNotFoundError(
            f"No se encontraron datasets '{dataset_name}' en {data_dir}. "
            f"Genéralo con scripts/generate_dataset.py"
        )

    df_coverage = pd.read_csv(cov_path, index_col=0)
    df_coverage.index = df_coverage.index.astype(str).str.strip()
    df_coverage.columns = df_coverage.columns.astype(str).str.strip()
    df_coverage = df_coverage.astype(int)

    df_tests = pd.read_csv(tests_path)
    df_tests["Test"] = df_tests["Test"].astype(str).str.strip()
    df_tests.index = df_coverage.index
    return df_coverage, df_tests


def evaluate_selection(selected, df_coverage, df_tests, time_weight=TIME_WEIGHT):
    """Evalúa una selección de tests.

    Devuelve dict con número de tests, tiempo total, requisitos cubiertos,
    factibilidad y valor objetivo (minimización).
    """
    selected = list(selected)
    n_tests = len(selected)
    total_time = float(df_tests.loc[selected, "time"].sum()) if selected else 0.0

    covered = set()
    for t in selected:
        covered.update(df_coverage.columns[df_coverage.loc[t] == 1].tolist())
    total_reqs = len(df_coverage.columns)
    n_covered = len(covered)
    feasible = n_covered == total_reqs

    objective = n_tests + time_weight * total_time
    return {
        "selected": selected,
        "selected_case_num": n_tests,
        "total_time": total_time,
        "covered_reqs": n_covered,
        "total_reqs": total_reqs,
        "feasible": feasible,
        "objective": objective,
    }


def is_feasible(selected, df_coverage):
    """True si la selección cubre todos los requisitos."""
    covered = set()
    for t in selected:
        covered.update(df_coverage.columns[df_coverage.loc[t] == 1].tolist())
    return len(covered) == len(df_coverage.columns)