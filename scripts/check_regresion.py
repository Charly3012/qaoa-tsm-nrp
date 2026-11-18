#!/usr/bin/env python3
"""Prueba de regresion de scale_n.py contra los CSV exploratorios.

Repite, con la semilla de datos 42, seis corridas pequenas (N <= 12) hacia una
carpeta temporal y las compara fila por fila con los CSV versionados en
report/. Sirve para comprobar que un cambio al codigo que NO debe alterar
resultados efectivamente no los altera.

Se comparan las columnas deterministas: optimo, resultado de cada metodo,
evaluaciones, p_opt, p_feasible, p_see_opt, energia, p_best y parametros
optimizados. No se comparan los tiempos (varian entre corridas).

Uso (desde codigo/):
    python scripts/check_regresion.py                 # corre y compara (~5 min)
    python scripts/check_regresion.py --keep DIR      # conserva los CSV nuevos en DIR
Codigo de salida: 0 si no hay diferencias, 1 si las hay.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIZES = "4,6,8,10,12"

# (argumentos de scale_n.py, subcarpeta de salida, CSV nuevo, CSV de referencia)
CASES = [
    (["--family", "tsm_aleatoria", "--sizes", SIZES, "--config", "FCI", "--init", "transfer", "--alpha", "0.25"],
     "", "scale_tsm_aleatoria_FCI_transfer_a0.25.csv", "report/fase4/scale_tsm_aleatoria_FCI_transfer_a0.25.csv"),
    (["--family", "tsm_aleatoria", "--sizes", SIZES, "--config", "F"],
     "", "scale_tsm_aleatoria_F.csv", "report/fase4/scale_tsm_aleatoria_F.csv"),
    (["--family", "tsm_ciclo", "--sizes", SIZES, "--config", "FC", "--alpha", "0.25"],
     "", "scale_tsm_ciclo_FC_a0.25.csv", "report/fase4/scale_tsm_ciclo_FC_a0.25.csv"),
    # Los CSV exploratorios de NRP se corrieron con P = lambda1 + lambda2 (deuda tecnica #7).
    (["--family", "nrp", "--sizes", SIZES, "--config", "FCI", "--init", "transfer", "--alpha", "0.25",
      "--penalty-logic", "lambda"],
     "", "scale_nrp_FCI_transfer_a0.25.csv", "report/fase4/nrp/scale_nrp_FCI_transfer_a0.25.csv"),
    (["--family", "nrp", "--sizes", SIZES, "--config", "F", "--penalty-logic", "lambda"],
     "", "scale_nrp_F.csv", "report/fase4/nrp/scale_nrp_F.csv"),
    (["--family", "tsm_aleatoria", "--sizes", "4,6", "--config", "base"],
     "base", "scale_tsm_aleatoria_base.csv", "report/scale_n/scale_tsm_aleatoria.csv"),
]

KEY = ["n", "method", "seed"]
EXACT = ["n_qubo", "optimum", "metric", "optimal", "feasible", "n_opt_states", "nfev", "params"]
NUMERIC = ["p_opt", "p_feasible", "p_see_opt", "energy", "p_best"]
TOL = 1e-9   # diferencia relativa admitida en las columnas numericas


def compare(new_csv, ref_csv):
    """Devuelve (filas emparejadas, lista de diferencias)."""
    new, ref = pd.read_csv(new_csv), pd.read_csv(ref_csv)
    for df in (new, ref):
        df["seed"] = df["seed"].astype(str)
    merged = new.merge(ref, on=KEY, suffixes=("_new", "_ref"))
    problems = []
    if len(merged) != len(new):
        problems.append(f"{len(new) - len(merged)} filas nuevas sin pareja en la referencia")
    for col in EXACT:
        if f"{col}_ref" not in merged or f"{col}_new" not in merged:
            continue
        a, b = merged[f"{col}_new"], merged[f"{col}_ref"]
        diff = ~((a == b) | (a.isna() & b.isna()))
        if diff.any():
            ex = merged.loc[diff, KEY].head(3).to_dict("records")
            problems.append(f"{col}: {int(diff.sum())} filas distintas, p. ej. {ex}")
    for col in NUMERIC:
        if f"{col}_ref" not in merged or f"{col}_new" not in merged:
            continue
        a, b = merged[f"{col}_new"].astype(float), merged[f"{col}_ref"].astype(float)
        if (a.isna() != b.isna()).any():
            problems.append(f"{col}: valores vacios en distinto lugar")
        both = a.notna() & b.notna()
        rel = (np.abs(a - b) / np.maximum(np.abs(b), 1e-300))[both]
        if (rel > TOL).any():
            problems.append(f"{col}: {int((rel > TOL).sum())} filas con diferencia relativa "
                            f"> {TOL:g} (max {rel.max():.2e})")
    return len(merged), problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keep", default=None, help="carpeta donde conservar los CSV nuevos")
    args = ap.parse_args()

    tmp = None
    if args.keep:
        out_root = Path(args.keep).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="regresion_")
        out_root = Path(tmp.name)

    total_rows, failed = 0, 0
    for scale_args, sub, new_name, ref in CASES:
        out_dir = out_root / sub if sub else out_root
        label = " ".join(scale_args)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "scale_n.py"), *scale_args, "--out-dir", str(out_dir)],
            cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        if proc.returncode != 0:
            failed += 1
            print(f"FALLA   {label}\n        scale_n.py termino con codigo {proc.returncode}:\n"
                  f"{proc.stderr[-1500:]}")
            continue
        avisos = [ln for ln in proc.stdout.splitlines() if ln.startswith("AVISO")]
        rows, problems = compare(out_dir / new_name, ROOT / ref)
        total_rows += rows
        if problems or avisos:
            failed += 1
            print(f"DISTINTO {label}  ({rows} filas)")
            for p in problems + avisos:
                print(f"        - {p}")
        else:
            print(f"IGUAL   {label}  ({rows} filas)")

    print(f"\n{len(CASES) - failed} de {len(CASES)} casos sin diferencias; {total_rows} filas comparadas.")
    if tmp is not None:
        tmp.cleanup()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
