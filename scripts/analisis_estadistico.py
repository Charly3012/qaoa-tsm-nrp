#!/usr/bin/env python3
"""Analisis estadistico de la experimentacion (cap. 5 de la tesis).

Solo LEE los CSV de scale_n.py (y el de hardware) y escribe tablas. No corre QAOA.

Unidad de analisis: la instancia. Las semillas del optimizador se resumen por
instancia con la mediana (y el rango intercuartilico, RIC) de log10 A, donde
A = p_opt / p_opt_uniforme.

Hipotesis (familia de Holm: H1, H2, H3, H4, H6, H7; H5 aparte, por instancia):
  H1  TSM aleatoria, F+C+I: mediana de A > 1            Wilcoxon 1 muestra + signos
  H2  TSM aleatoria: A(F+C) > A(F)                      Wilcoxon pareado + FS >= 0.64
  H3  TSM aleatoria: RIC(F+C+I) < RIC(F+C)              Wilcoxon pareado
  H4  TSM aleatoria: QAOA ve el optimo donde el azar no McNemar exacto
  H6  NRP, F+C+I: mediana de A > 1                      igual que H1
  H7  TSM aleatoria, N = 8: A(F) > A(Base)              Wilcoxon pareado + FS >= 0.64
  H5  hardware < simulador (p_feasible y p_opt)         binomial exacta + Wilson

Modos:
  exploratorio   datos ya existentes (semilla de datos 42, 3 semillas, 1 instancia
                 por tamano). Resultado PRELIMINAR: no decide las hipotesis.
  confirmatorio  report/confirmatorio/ds*/ (semillas de datos nuevas).

Uso (desde codigo/):
    python scripts/analisis_estadistico.py --modo exploratorio --out report/estadistica/exploratorio
    python scripts/analisis_estadistico.py --modo confirmatorio --out report/estadistica/confirmatorio
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

ALPHA = 0.05
FS_MIN = 0.64            # umbral de efecto "mediano" (Vargha y Delaney, 2000, Tabla 1)
SHOTS = 8000             # mediciones de la extraccion final (scale_n.py --final-shots)
P_FLOOR = 1e-20          # piso de p_opt antes del logaritmo
HOLM_FAMILY = ["H1", "H2", "H3", "H4", "H6", "H7"]
SIZES = {"tsm_aleatoria": (8, 22), "nrp": (8, 20), "tsm_ciclo": (8, 22)}   # tamanos que se analizan
H7_SIZES = (8, 8)         # solo N = 8: unico tamano donde las holguras pesan (24 qubits frente a 8)
H7_DESCRIPTIVE_SIZES = (4, 6)
MIN_EXPECTED_OPT = 10    # regla de potencia de H5 para p_opt


# ---------------------------------------------------------------------------
# Estadisticos basicos (cada uno tiene su prueba unitaria)
# ---------------------------------------------------------------------------
def wilson_interval(k, n, z=1.959963984540054):
    """Intervalo de Wilson para una proporcion k/n. Devuelve (inferior, superior)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    center = (k + z * z / 2.0) / (n + z * z)
    half = z * math.sqrt(k * (n - k) / n + z * z / 4.0) / (n + z * z)
    return (max(0.0, center - half), min(1.0, center + half))


def holm_adjust(pvalues):
    """p-valores ajustados por Holm, en el orden de entrada (NaN se conservan)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    idx = np.where(~np.isnan(p))[0]
    m = len(idx)
    order = idx[np.argsort(p[idx], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def fraction_superior(d):
    """FS: fraccion de diferencias pareadas positivas (los empates cuentan la mitad)."""
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return float("nan")
    return float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / d.size)


def a12(x, y):
    """A12 de Vargha y Delaney: P(X > Y) + 0.5 P(X = Y), muestras independientes."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    greater = np.sum(x[:, None] > y[None, :])
    equal = np.sum(x[:, None] == y[None, :])
    return float((greater + 0.5 * equal) / (x.size * y.size))


def sign_test(values, alternative="greater"):
    """Prueba de signos contra 0. Devuelve (positivos, n sin ceros, p)."""
    v = np.asarray(values, dtype=float)
    v = v[v != 0]
    if v.size == 0:
        return 0, 0, 1.0
    k = int(np.sum(v > 0))
    return k, int(v.size), float(stats.binomtest(k, v.size, 0.5, alternative=alternative).pvalue)


def wilcoxon_signed(d, alternative="greater"):
    """Wilcoxon de rangos con signo sobre las diferencias d. Devuelve (W, p)."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0 or np.all(d == 0):
        return float("nan"), 1.0
    res = stats.wilcoxon(d, alternative=alternative, zero_method="wilcox")
    return float(res.statistic), float(res.pvalue)


def mcnemar_exact(b, c):
    """McNemar exacto unilateral (H1: b > c). Devuelve p; sin discordantes, p = 1."""
    if b + c == 0:
        return 1.0
    return float(stats.binomtest(int(b), int(b + c), 0.5, alternative="greater").pvalue)


def prob_see(p, shots=SHOTS):
    """Probabilidad de ver al menos un optimo en `shots` mediciones."""
    return 1.0 - (1.0 - np.asarray(p, dtype=float)) ** shots


def median_ci(values, n_boot=10000, seed=12345, level=0.95):
    """IC por bootstrap (percentiles) de la mediana. Devuelve (mediana, inf, sup)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    lo, hi = np.percentile(boots, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(np.median(v)), float(lo), float(hi)


def iqr(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan")
    q75, q25 = np.percentile(v, [75, 25])
    return float(q75 - q25)


# ---------------------------------------------------------------------------
# Carga y resumen por instancia
# ---------------------------------------------------------------------------
def config_label(row):
    cfg = row.get("config")
    if pd.isna(cfg) or cfg in ("", "base"):
        return "Base"
    if cfg == "F":
        return "F"
    alpha = row.get("alpha")
    alpha = "" if pd.isna(alpha) or alpha == "" else f"{float(alpha):g}"
    if cfg == "FC":
        return "F+C" if alpha == "0.25" else f"F+C(a={alpha})"
    if cfg == "FCI":
        init = row.get("init")
        base = "F+C+I" if init == "transfer" else f"F+C+I({init})"
        return base if alpha == "0.25" else f"{base}(a={alpha})"
    return str(cfg)


def load_runs(paths):
    """Concatena CSV de scale_n.py y normaliza las columnas que cambiaron con el tiempo."""
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source"] = Path(path).as_posix()
        for col, default in [("config", "base"), ("alpha", np.nan), ("init", "random"),
                             ("data_seed", 42), ("warm", np.nan), ("penalty_logic", np.nan)]:
            if col not in df.columns:
                df[col] = default
        df["data_seed"] = df["data_seed"].fillna(42).astype(int)
        frames.append(df)
    if not frames:
        raise SystemExit("No se encontraron CSV de entrada")
    runs = pd.concat(frames, ignore_index=True)
    runs["cfg"] = runs.apply(config_label, axis=1)
    return runs


def summarize_instances(runs, shots=SHOTS):
    """Una fila por (familia, instancia, configuracion): mediana y RIC de log10 A, etc."""
    unif = (runs[runs.method == "uniforme"]
            .drop_duplicates(["family", "instance"])
            .set_index(["family", "instance"])[["p_opt", "p_feasible"]]
            .rename(columns={"p_opt": "p_opt_unif", "p_feasible": "p_feas_unif"}))
    q = runs[runs.method == "qaoa"].copy()
    q = q.join(unif, on=["family", "instance"])
    q["logA"] = np.log10(np.maximum(q["p_opt"].astype(float), P_FLOOR) / q["p_opt_unif"])
    rows = []
    for (family, instance, cfg), g in q.groupby(["family", "instance", "cfg"], sort=False):
        p_med = float(np.median(g["p_opt"]))
        rows.append(dict(
            family=family, instance=instance, cfg=cfg, n=int(g["n"].iloc[0]),
            data_seed=int(g["data_seed"].iloc[0]), n_seeds=len(g),
            logA_median=float(np.median(g["logA"])), logA_iqr=iqr(g["logA"]),
            logA_max=float(np.max(g["logA"])), A_median=float(10 ** np.median(g["logA"])),
            p_opt_median=p_med, p_opt_unif=float(g["p_opt_unif"].iloc[0]),
            p_feasible_median=float(np.median(g["p_feasible"])),
            p_see_qaoa=float(prob_see(p_med, shots)),
            p_see_unif=float(prob_see(g["p_opt_unif"].iloc[0], shots)),
            success_sampled=float(np.mean(g["optimal"])),
            warm_all=(bool((g["warm"] == 1).all()) if g["warm"].notna().any() else None),
            secs_mean=float(np.mean(g["secs"])),
        ))
    return pd.DataFrame(rows)


def check_integrity(summary, expected_seeds=None):
    """Lista de problemas que invalidarian el analisis confirmatorio."""
    problems = []
    fci = summary[(summary.cfg == "F+C+I") & (summary.n >= 8)]
    cold = fci[fci.warm_all == False]            # noqa: E712  (None = dato exploratorio sin columna)
    for _, r in cold.iterrows():
        problems.append(f"{r.instance}: F+C+I sin transferencia (warm=0) en alguna semilla")
    if expected_seeds:
        short = summary[summary.n_seeds != expected_seeds]
        for _, r in short.iterrows():
            problems.append(f"{r.instance} [{r.cfg}]: {r.n_seeds} semillas (se esperaban {expected_seeds})")
    return problems


def _select(summary, family, cfg, sizes):
    lo, hi = sizes
    s = summary[(summary.family == family) & (summary.cfg == cfg) & (summary.n >= lo) & (summary.n <= hi)]
    return s.set_index("instance").sort_index()


def _paired(summary, family, cfg_new, cfg_old, sizes, column="logA_median"):
    a, b = _select(summary, family, cfg_new, sizes), _select(summary, family, cfg_old, sizes)
    common = a.index.intersection(b.index)
    return a.loc[common, column].to_numpy(), b.loc[common, column].to_numpy()


# ---------------------------------------------------------------------------
# Hipotesis
# ---------------------------------------------------------------------------
def test_one_sample(summary, hid, family, pi):
    s = _select(summary, family, "F+C+I", SIZES[family])
    x = s["logA_median"].to_numpy()
    W, p = wilcoxon_signed(x, "greater")
    k, n_sign, p_sign = sign_test(x, "greater")
    med, lo, hi = median_ci(x)
    return dict(hipotesis=hid, pi=pi, familia=family, n=len(x), prueba="Wilcoxon 1 muestra (>0)",
                estadistico=W, p=p,
                efecto=f"mediana A = {10 ** med:.3g} [IC95 {10 ** lo:.3g}, {10 ** hi:.3g}]; "
                       f"signos {k}/{n_sign} (p = {p_sign:.3g})",
                efecto_valor=10 ** med if len(x) else np.nan, umbral_efecto=np.nan)


def test_paired(summary, hid, pi, family, cfg_new, cfg_old, sizes, column="logA_median",
                alternative="greater", effect_threshold=None, label=None):
    new, old = _paired(summary, family, cfg_new, cfg_old, sizes, column)
    d = new - old
    W, p = wilcoxon_signed(d, alternative)
    fs = fraction_superior(d if alternative == "greater" else -d)
    med = float(np.median(d)) if len(d) else float("nan")
    if column == "logA_median":
        effect = f"FS = {fs:.2f}; mediana de la razon de A = {10 ** med:.3g}"
    else:
        effect = (f"FS = {fs:.2f}; mediana de RIC: {np.median(new) if len(d) else float('nan'):.3g} "
                  f"({cfg_new}) frente a {np.median(old) if len(d) else float('nan'):.3g} ({cfg_old})")
    return dict(hipotesis=hid, pi=pi, familia=family, n=len(d),
                prueba=label or f"Wilcoxon pareado ({cfg_new} vs {cfg_old})",
                estadistico=W, p=p, efecto=effect, efecto_valor=fs,
                umbral_efecto=effect_threshold if effect_threshold is not None else np.nan)


def test_success(summary, hid, pi, family):
    s = _select(summary, family, "F+C+I", SIZES[family])
    q_ok, u_ok = s["p_see_qaoa"] >= 0.5, s["p_see_unif"] >= 0.5
    b, c = int((q_ok & ~u_ok).sum()), int((~q_ok & u_ok).sum())
    return dict(hipotesis=hid, pi=pi, familia=family, n=len(s), prueba="McNemar exacto",
                estadistico=float(b), p=mcnemar_exact(b, c),
                efecto=f"b = {b} (QAOA si, azar no), c = {c}; QAOA {int(q_ok.sum())}/{len(s)}, "
                       f"azar {int(u_ok.sum())}/{len(s)}",
                efecto_valor=float(b - c), umbral_efecto=np.nan)


def run_hypotheses(summary):
    tests = [
        test_one_sample(summary, "H1", "tsm_aleatoria", "PI3"),
        test_paired(summary, "H2", "PI3", "tsm_aleatoria", "F+C", "F", SIZES["tsm_aleatoria"],
                    effect_threshold=FS_MIN),
        test_paired(summary, "H3", "PI3", "tsm_aleatoria", "F+C+I", "F+C", SIZES["tsm_aleatoria"],
                    column="logA_iqr", alternative="less",
                    label="Wilcoxon pareado (RIC F+C+I < RIC F+C)"),
        test_success(summary, "H4", "PI1", "tsm_aleatoria"),
        test_one_sample(summary, "H6", "nrp", "PI3"),
        test_paired(summary, "H7", "PI3", "tsm_aleatoria", "F", "Base", H7_SIZES,
                    effect_threshold=FS_MIN),
    ]
    out = pd.DataFrame(tests)
    out["p_holm"] = holm_adjust(out["p"].to_numpy())
    decisions = []
    for _, r in out.iterrows():
        if r.n == 0:
            decisions.append("sin datos")
            continue
        ok = r.p_holm < ALPHA
        if not np.isnan(r.umbral_efecto):
            ok = ok and r.efecto_valor >= r.umbral_efecto
        decisions.append("Rechaza H0" if ok else "No rechaza H0")
    out["decision"] = decisions
    cols = ["hipotesis", "pi", "familia", "n", "prueba", "estadistico", "p", "p_holm", "efecto",
            "efecto_valor", "umbral_efecto", "decision"]
    return out[cols]


def run_secondary(summary):
    """Analisis secundarios: descriptivos, sin ajuste ni regla de decision."""
    rows = []
    lo, hi = SIZES["nrp"]
    tsm = _select(summary, "tsm_aleatoria", "F+C+I", (lo, hi))["logA_median"].to_numpy()
    nrp = _select(summary, "nrp", "F+C+I", (lo, hi))["logA_median"].to_numpy()
    if len(tsm) and len(nrp):
        u = stats.mannwhitneyu(tsm, nrp, alternative="greater")
        rows.append(dict(analisis="TSM aleatoria vs NRP (log A, F+C+I, N 8-20)", n=f"{len(tsm)} / {len(nrp)}",
                         prueba="Mann-Whitney U (TSM > NRP)", estadistico=float(u.statistic),
                         p=float(u.pvalue), efecto=f"A12 = {a12(tsm, nrp):.2f}"))
    for family in ("nrp", "tsm_ciclo"):
        if len(_select(summary, family, "F+C+I", SIZES[family])):
            t = test_success(summary, "H4", "PI1", family)
            rows.append(dict(analisis=f"Exito frente al azar en {family} (F+C+I)", n=t["n"],
                             prueba=t["prueba"], estadistico=t["estadistico"], p=t["p"], efecto=t["efecto"]))
    new, old = _paired(summary, "tsm_aleatoria", "F", "Base", H7_DESCRIPTIVE_SIZES)
    if len(new):
        W, p = wilcoxon_signed(new - old, "greater")
        rows.append(dict(analisis="Sin holguras vs con holguras en N = 4 y 6 (TSM aleatoria)", n=len(new),
                         prueba="Wilcoxon pareado (F vs Base)", estadistico=W, p=p,
                         efecto=f"FS = {fraction_superior(new - old):.2f}; mediana de la razon de A = "
                                f"{10 ** np.median(new - old):.3g}"))
    new, old = _paired(summary, "tsm_aleatoria", "F+C+I", "F+C", SIZES["tsm_aleatoria"])
    if len(new):
        W, p = wilcoxon_signed(new - old, "greater")
        rows.append(dict(analisis="Nivel de A: F+C+I vs F+C (TSM aleatoria)", n=len(new),
                         prueba="Wilcoxon pareado", estadistico=W, p=p,
                         efecto=f"FS = {fraction_superior(new - old):.2f}; mediana de la razon de A = "
                                f"{10 ** np.median(new - old):.3g}"))
    return pd.DataFrame(rows)


def describe_by_size(summary):
    g = summary.groupby(["family", "cfg", "n"])
    out = g.agg(instancias=("instance", "count"), semillas=("n_seeds", "median"),
                A_mediana=("A_median", "median"), A_min=("A_median", "min"), A_max=("A_median", "max"),
                logA_ric_entre_instancias=("logA_median", iqr),
                ric_entre_semillas=("logA_iqr", "median"),
                p_opt_mediana=("p_opt_median", "median"),
                p_feasible_mediana=("p_feasible_median", "median"),
                ve_optimo_qaoa=("p_see_qaoa", lambda s: float(np.mean(s >= 0.5))),
                ve_optimo_azar=("p_see_unif", lambda s: float(np.mean(s >= 0.5))),
                exito_muestreado=("success_sampled", "mean"),
                segundos=("secs_mean", "mean")).reset_index()
    return out


# ---------------------------------------------------------------------------
# H5: hardware frente a simulador
# ---------------------------------------------------------------------------
def hardware_tests(hw, default_shots=4000):
    """Una fila por (instancia, medida). Junta las replicas de cada instancia."""
    hw = hw.copy()
    if "shots" not in hw.columns:
        hw["shots"] = default_shots
    if "replica" not in hw.columns:
        hw["replica"] = hw.groupby("instance").cumcount() + 1
    rows = []
    for instance, g in hw.groupby("instance", sort=False):
        n = int(g["shots"].sum())
        for measure, col, sim_col in [("p_feasible", "p_feasible", "sim_p_feasible"),
                                      ("p_opt", "p_opt_sampled", "sim_p_opt")]:
            counts = np.rint(g[col].astype(float) * g["shots"]).astype(int).to_numpy()
            k = int(counts.sum())
            p_sim = float(g[sim_col].iloc[0]) if sim_col in g and pd.notna(g[sim_col].iloc[0]) else float("nan")
            lo, hi = wilson_interval(k, n)
            row = dict(instancia=instance, medida=measure, replicas=len(g), mediciones=n, k=k,
                       p_hw=k / n, wilson_inf=lo, wilson_sup=hi, p_sim=p_sim)
            if np.isnan(p_sim) or p_sim <= 0:
                row.update(esperados=np.nan, cociente=np.nan, cociente_inf=np.nan, cociente_sup=np.nan,
                           p=np.nan, conclusion="sin referencia del simulador")
            else:
                expected = n * p_sim
                p = float(stats.binomtest(k, n, min(p_sim, 1.0), alternative="less").pvalue)
                concl = "hardware < simulador" if p < ALPHA else "compatible con el simulador"
                if measure == "p_opt" and expected < MIN_EXPECTED_OPT:
                    concl = f"no concluyente (< {MIN_EXPECTED_OPT} optimos esperados)"
                row.update(esperados=expected, cociente=(k / n) / p_sim, cociente_inf=lo / p_sim,
                           cociente_sup=hi / p_sim, p=p, conclusion=concl)
            if len(g) >= 2 and 0 < k < n:
                table = np.vstack([counts, g["shots"].to_numpy() - counts])
                row["p_homogeneidad"] = float(stats.chi2_contingency(table)[1])
            else:
                row["p_homogeneidad"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entradas por modo y salida
# ---------------------------------------------------------------------------
def exploratory_inputs():
    f4, sn = ROOT / "report" / "fase4", ROOT / "report" / "scale_n"
    names = ["F", "FC_a0.25", "FCI_transfer_a0.25"]
    paths = [f4 / f"scale_tsm_aleatoria_{n}.csv" for n in names]
    paths += [f4 / "nrp" / f"scale_nrp_{n}.csv" for n in names]
    paths += [f4 / f"scale_tsm_ciclo_{n}.csv" for n in ["FC_a0.25", "FCI_transfer_a0.25"]]
    paths += [sn / "scale_tsm_aleatoria.csv"]
    return [p for p in paths if p.exists()], f4 / "hardware_fase4.csv"


def confirmatory_inputs(root):
    root = Path(root)
    return sorted(root.glob("ds*/scale_*.csv")), root / "hardware_replicas.csv"


def to_markdown(df, floatfmt="{:.4g}"):
    if df.empty:
        return "_(sin filas)_"
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modo", choices=["exploratorio", "confirmatorio"], required=True)
    ap.add_argument("--entrada", default="report/confirmatorio",
                    help="carpeta con ds*/scale_*.csv (solo modo confirmatorio)")
    ap.add_argument("--out", required=True, help="carpeta de salida")
    ap.add_argument("--semillas", type=int, default=None,
                    help="semillas del optimizador esperadas por instancia (confirmatorio: 10)")
    ap.add_argument("--hardware", default=None, help="CSV de hardware (por defecto, el del modo)")
    args = ap.parse_args()

    if args.modo == "exploratorio":
        paths, hw_path = exploratory_inputs()
        expected = args.semillas
    else:
        entrada = Path(args.entrada)
        paths, hw_path = confirmatory_inputs(entrada if entrada.is_absolute() else ROOT / entrada)
        expected = args.semillas or 10
    if args.hardware:
        hw_path = Path(args.hardware)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(paths)
    if args.modo == "confirmatorio":
        seen = runs[runs.data_seed == 42]
        if len(seen):
            raise SystemExit("El modo confirmatorio no admite datos de la semilla de datos 42 (exploratoria)")
    summary = summarize_instances(runs)
    problems = check_integrity(summary, expected)
    tests = run_hypotheses(summary)
    secondary = run_secondary(summary)
    descr = describe_by_size(summary)

    summary.to_csv(out / "resumen_por_instancia.csv", index=False)
    descr.to_csv(out / "descriptiva_por_n.csv", index=False)
    tests.to_csv(out / "pruebas.csv", index=False)
    secondary.to_csv(out / "secundarios.csv", index=False)
    hw = pd.DataFrame()
    if hw_path.exists():
        hw = hardware_tests(pd.read_csv(hw_path))
        hw.to_csv(out / "hardware_wilson.csv", index=False)

    label = "PRELIMINAR (exploratorio): no decide las hipotesis" if args.modo == "exploratorio" \
        else "CONFIRMATORIO"
    lines = [f"# Analisis estadistico - {label}", "",
             f"Entradas: {len(paths)} CSV; {summary.instance.nunique()} instancias; "
             f"semillas de datos {sorted(summary.data_seed.unique().tolist())}.", ""]
    if problems:
        lines += ["## Problemas de integridad", ""] + [f"- {p}" for p in problems] + [""]
    else:
        lines += ["Sin problemas de integridad.", ""]
    show = tests[["hipotesis", "familia", "n", "prueba", "estadistico", "p", "p_holm", "efecto", "decision"]]
    lines += ["## Hipotesis (familia de Holm)", "", to_markdown(show), "",
              "## Analisis secundarios (sin ajuste)", "", to_markdown(secondary), ""]
    if not hw.empty:
        hw_show = hw[["instancia", "medida", "replicas", "mediciones", "k", "p_hw", "wilson_inf",
                      "wilson_sup", "p_sim", "esperados", "cociente", "p", "conclusion"]]
        lines += ["## H5: hardware frente a simulador (por instancia)", "", to_markdown(hw_show), ""]
    (out / "resumen.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nTablas en {out}")


if __name__ == "__main__":
    main()
