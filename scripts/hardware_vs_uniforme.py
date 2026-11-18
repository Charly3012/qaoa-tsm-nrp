"""Analisis EXPLORATORIO (fuera del protocolo confirmatorio): hardware frente al muestreo uniforme.

Para cada instancia ejecutada en hardware calcula p_feasible y p_opt del muestreo
uniforme sobre las variables de decision (exactos, por enumeracion) y los pone junto
a los valores del simulador y del hardware de hardware_wilson.csv. Sirve para ver si
el hardware queda entre el simulador y el azar, y cuanto de la ventaja del simulador
sobre el azar se conserva:  retenido = (p_hw - p_unif) / (p_sim - p_unif).

No usa IBM ni corre QAOA. La instancia NRP toy2 (30 variables de decision) no se
enumera: su p_feasible uniforme se estima por Monte Carlo (se marca en la tabla).

    python scripts/hardware_vs_uniforme.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("hwrep", ROOT / "scripts" / "hardware_replicas.py")
HW = importlib.util.module_from_spec(spec)
spec.loader.exec_module(HW)

CONF = ROOT / "report" / "confirmatorio"
EST = ROOT / "report" / "estadistica" / "confirmatorio"
MAX_ENUM = 24
MC_SAMPLES = 4_000_000


def mc_feasible_nrp(df_feat, df_deps, budget, rng):
    names = df_feat.index.tolist()
    x = rng.integers(0, 2, size=(MC_SAMPLES, len(names)), dtype=np.uint8)
    cost = x @ df_feat["Costo"].to_numpy(dtype=float)
    feas = cost <= budget
    pos = {f: k for k, f in enumerate(names)}
    for _, row in df_deps.iterrows():
        o, rel, d = row["Origen"], row["Relacion"], row["Destino"]
        if o not in pos or d not in pos:
            continue
        xo, xd = x[:, pos[o]], x[:, pos[d]]
        if rel == "REQUIRES":
            feas &= ~((xo == 1) & (xd == 0))
        elif rel == "EXCLUDES":
            feas &= ~((xo == 1) & (xd == 1))
    return float(feas.mean())


def main():
    jobs = HW.default_jobs(["A", "B"]) + HW.read_jobs_csv(CONF / "bloque_c.csv")
    rows = []
    for job in jobs:
        M = HW.SCALE.load_problem(job["problem"])
        qubo, names, tables = HW.build_job_qubo(job, M)
        row = dict(instancia=HW.job_key(job), bloque=job["block"], n_decision=len(names),
                   qubits=qubo.get_num_vars(), metodo_uniforme="exacto")
        if len(names) <= MAX_ENUM:
            feas, opt = tables()
            row.update(unif_p_feasible=float(feas.mean()), unif_p_opt=float(opt.mean()))
        else:
            df_a, df_b = M["data_io"].load_nrp(job["instance"])
            row.update(unif_p_feasible=mc_feasible_nrp(df_a, df_b, job["budget"], np.random.default_rng(0)),
                       unif_p_opt=np.nan, metodo_uniforme=f"Monte Carlo ({MC_SAMPLES} muestras)")
        rows.append(row)
        print(row, flush=True)
    uni = pd.DataFrame(rows)
    hw = pd.read_csv(EST / "hardware_wilson.csv")
    wide = hw.pivot(index="instancia", columns="medida", values=["p_hw", "wilson_inf", "wilson_sup", "p_sim"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    out = uni.merge(wide.reset_index(), on="instancia", how="left")
    for m in ("p_feasible", "p_opt"):
        u, s, h = out[f"unif_{m}"], out[f"p_sim_{m}"], out[f"p_hw_{m}"]
        out[f"sim_sobre_unif_{m}"] = s / u
        out[f"hw_sobre_unif_{m}"] = h / u
        out[f"retenido_{m}"] = (h - u) / (s - u)
    out.to_csv(EST / "hardware_vs_uniforme.csv", index=False)
    print("\n->", EST / "hardware_vs_uniforme.csv")


if __name__ == "__main__":
    sys.exit(main())
