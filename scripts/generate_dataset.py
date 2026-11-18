#!/usr/bin/env python3
"""Generador deterministico de datasets para TSM y NRP.

Crea datasets nativos del proyecto (sin depender de fuentes externas) con una
semilla reproducible. Dos modos:

    python generate_dataset.py tsm --name demo --tests 12 --reqs 12 --cover 2 --seed 42
    python generate_dataset.py nrp --name demo --features 12 --budget 60 --seed 42

- TSM: escribe `tsm_{name}_coverage.csv` (matriz test x requisito) y
  `tsm_{name}_tests.csv` (Test, time). La cobertura en "ciclo" (cada requisito
  cubierto por exactamente 2 tests consecutivos) minimiza las variables de
  holgura del QUBO, lo que la hace apta para QAOA.
- NRP: escribe `nrp_{name}_features.csv` (Feature, Nombre, Costo, Valor) y
  `nrp_{name}_dependencies.csv` (Origen, Relacion, Destino, Descripcion), con
  dependencias REQUIRES/EXCLUDES validas.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TSM_DATA = ROOT / "TSM" / "data"
NRP_DATA = ROOT / "NRP" / "data"

NOMBRES_FEATURES = [
    "Autenticacion SSO",
    "Panel de Analiticas",
    "Motor de Reportes",
    "Exportacion CSV",
    "Notificaciones Push",
    "Busqueda Full-Text",
    "API Publica REST",
    "Modo Oscuro",
    "Internacionalizacion i18n",
    "Cache Distribuida",
    "Cola de Trabajos Async",
    "Auditoria de Accesos",
    "Integracion de Pagos",
    "Importador de Datos",
    "Roles y Permisos",
    "Backup Automatico",
]


def gen_tsm(name, tests, reqs, cover, seed, out_dir=None):
    rng = np.random.default_rng(seed)
    test_idx = [f"T{i}" for i in range(tests)]
    req_idx = [f"R{j}" for j in range(reqs)]

    matriz = np.zeros((tests, reqs), dtype=int)
    for j in range(reqs):
        if cover == 2 and reqs == tests:
            cubren = [j, (j + 1) % tests]
        else:
            cubren = rng.choice(tests, size=min(cover, tests), replace=False).tolist()
        for c in cubren:
            matriz[c, j] = 1

    cov = pd.DataFrame(matriz, index=test_idx, columns=req_idx)
    times = np.round(rng.uniform(0.5, 2.0, tests), 2)
    tests_df = pd.DataFrame({"Test": test_idx, "time": times})

    out = Path(out_dir) if out_dir else TSM_DATA
    out.mkdir(parents=True, exist_ok=True)
    cov.to_csv(out / f"tsm_{name}_coverage.csv")
    tests_df.to_csv(out / f"tsm_{name}_tests.csv", index=False)

    cobertura_por_req = matriz.sum(axis=0)
    print(f"[TSM] {name}: {tests} tests x {reqs} reqs | "
          f"cobertura por req {cobertura_por_req.min()}..{cobertura_por_req.max()}")


def gen_nrp(name, features, budget, seed, requires=1, excludes=1, out_dir=None):
    rng = np.random.default_rng(seed)
    idx = [f"F{i}" for i in range(features)]
    cost = rng.integers(5, 40, features)
    value = rng.integers(10, 90, features)
    nombres = [NOMBRES_FEATURES[i % len(NOMBRES_FEATURES)] for i in range(features)]

    feats = pd.DataFrame(
        {"Feature": idx, "Nombre": nombres, "Costo": cost, "Valor": value}
    )
    feats.index = idx

    filas = []
    usados = set()
    for _ in range(requires):
        o, d = _par_distinto(rng, features, usados)
        filas.append((idx[o], "REQUIRES", idx[d],
                      f"{idx[o]} requiere {idx[d]}"))
    for _ in range(excludes):
        o, d = _par_distinto(rng, features, usados)
        filas.append((idx[o], "EXCLUDES", idx[d],
                      f"{idx[o]} y {idx[d]} no pueden ir juntos"))
    deps = pd.DataFrame(filas, columns=["Origen", "Relacion", "Destino", "Descripcion"])

    out = Path(out_dir) if out_dir else NRP_DATA
    out.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out / f"nrp_{name}_features.csv")
    deps.to_csv(out / f"nrp_{name}_dependencies.csv", index=False)

    print(f"[NRP] {name}: {features} features, budget {budget} | "
          f"costo total {int(cost.sum())}, valor total {int(value.sum())} | "
          f"deps {len(deps)}")


def _par_distinto(rng, features, usados):
    while True:
        o, d = rng.integers(0, features, 2)
        if o != d and (o, d) not in usados and (d, o) not in usados:
            usados.add((o, d))
            return int(o), int(d)


def main():
    parser = argparse.ArgumentParser(description="Generador de datasets TSM/NRP")
    sub = parser.add_subparsers(dest="modo", required=True)

    p_tsm = sub.add_parser("tsm")
    p_tsm.add_argument("--name", default="demo")
    p_tsm.add_argument("--tests", type=int, default=12)
    p_tsm.add_argument("--reqs", type=int, default=12)
    p_tsm.add_argument("--cover", type=int, default=2,
                       help="Tests que cubren cada requisito (2 y reqs==tests -> ciclo)")
    p_tsm.add_argument("--seed", type=int, default=42)

    p_nrp = sub.add_parser("nrp")
    p_nrp.add_argument("--name", default="demo")
    p_nrp.add_argument("--features", type=int, default=12)
    p_nrp.add_argument("--budget", type=int, default=60)
    p_nrp.add_argument("--requires", type=int, default=1)
    p_nrp.add_argument("--excludes", type=int, default=1)
    p_nrp.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.modo == "tsm":
        gen_tsm(args.name, args.tests, args.reqs, args.cover, args.seed)
    else:
        gen_nrp(args.name, args.features, args.budget, args.seed,
                args.requires, args.excludes)


if __name__ == "__main__":
    main()
