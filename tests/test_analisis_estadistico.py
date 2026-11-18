"""Pruebas del analisis estadistico (scripts/analisis_estadistico.py) con
valores calculados a mano.

Ejecutar desde codigo/:  python -m pytest tests -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ae():
    path = ROOT / "scripts" / "analisis_estadistico.py"
    spec = importlib.util.spec_from_file_location("analisis_estadistico", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analisis_estadistico"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Estadisticos basicos
# ---------------------------------------------------------------------------
def test_wilson_valores_conocidos(ae):
    z2 = 1.959963984540054 ** 2
    lo, hi = ae.wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(z2 / (10 + z2))            # 0.2775
    lo, hi = ae.wilson_interval(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert lo + hi == pytest.approx(1.0)                   # simetrico en k = n/2
    lo, hi = ae.wilson_interval(10, 10)
    assert hi == 1.0 and lo == pytest.approx(10 / (10 + z2))


def test_wilson_con_muchas_mediciones_se_acerca_a_la_proporcion(ae):
    lo, hi = ae.wilson_interval(40, 12000)
    assert lo < 40 / 12000 < hi
    assert hi - lo < 0.0025


def test_holm_ejemplo_a_mano(ae):
    # ordenados: .01, .03, .04 -> 3*.01, 2*.03, 1*.04 -> .03, .06, max(.06, .04) = .06
    assert ae.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    # seis pruebas: el menor se multiplica por 6 y nunca pasa de 1
    adj = ae.holm_adjust([0.001, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert adj[0] == pytest.approx(0.006)
    assert np.all(adj[1:] == 1.0)


def test_holm_es_monotono_e_ignora_nan(ae):
    adj = ae.holm_adjust([0.02, np.nan, 0.01])
    assert np.isnan(adj[1])
    assert adj[2] == pytest.approx(0.02) and adj[0] == pytest.approx(0.02)


def test_fraccion_de_superioridad(ae):
    assert ae.fraction_superior([1, 2, 0, -1]) == pytest.approx(0.625)   # (2 + 0.5) / 4
    assert ae.fraction_superior([3, 1, 2]) == 1.0
    assert ae.fraction_superior([0, 0]) == 0.5


def test_a12(ae):
    # x=[1,2,3], y=[1,2]: mayores (2>1),(3>1),(3>2) = 3; empates 2 -> (3 + 1) / 6
    assert ae.a12([1, 2, 3], [1, 2]) == pytest.approx(4 / 6)
    assert ae.a12([5, 6], [1, 2]) == 1.0
    assert ae.a12([1, 2], [5, 6]) == 0.0
    assert ae.a12([1, 2], [1, 2]) == 0.5


def test_prueba_de_signos(ae):
    k, n, p = ae.sign_test([0.5, 1.0, 2.0, 0.1, 3.0])
    assert (k, n) == (5, 5) and p == pytest.approx(1 / 32)
    k, n, p = ae.sign_test([1, -1, 0, 2])                  # el cero se descarta
    assert (k, n) == (2, 3) and p == pytest.approx(0.5)    # P(X >= 2 | n = 3) = 4/8


def test_wilcoxon_cinco_diferencias_positivas(ae):
    _, p = ae.wilcoxon_signed([0.1, 0.2, 0.3, 0.4, 0.5], "greater")
    assert p == pytest.approx(1 / 32)                      # el menor p posible con n = 5
    _, p = ae.wilcoxon_signed([0.1, 0.2, 0.3, 0.4, 0.5], "less")
    assert p == pytest.approx(1.0)
    _, p = ae.wilcoxon_signed([0, 0, 0], "greater")
    assert p == 1.0


def test_mcnemar_exacto(ae):
    assert ae.mcnemar_exact(7, 0) == pytest.approx(1 / 128)
    assert ae.mcnemar_exact(0, 0) == 1.0
    assert ae.mcnemar_exact(3, 3) == pytest.approx(42 / 64)   # P(X >= 3 | n = 6)


def test_probabilidad_de_ver_el_optimo(ae):
    assert ae.prob_see(0.0) == 0.0
    assert ae.prob_see(1.0) == 1.0
    umbral = 1 - 0.5 ** (1 / 8000)                         # p con la que P_ver = 0.5
    assert umbral == pytest.approx(8.66e-5, rel=1e-2)
    assert ae.prob_see(umbral) == pytest.approx(0.5)


def test_ic_de_la_mediana_contiene_a_la_mediana(ae):
    med, lo, hi = ae.median_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert med == 3.0 and lo <= med <= hi
    assert ae.median_ci([2.0] * 6) == (2.0, 2.0, 2.0)


# ---------------------------------------------------------------------------
# Flujo completo con datos sinteticos
# ---------------------------------------------------------------------------
def _fake_runs(amp):
    """Corridas sinteticas: amp[(familia, cfg)] -> funcion(n, semilla) = A deseada."""
    cfgs = {"Base": ("base", np.nan, "random"), "F": ("F", np.nan, "random"),
            "F+C": ("FC", 0.25, "random"), "F+C+I": ("FCI", 0.25, "transfer")}
    rows = []
    for (family, label), fn in amp.items():
        config, alpha, init = cfgs[label]
        sizes = (4, 6, 8) if label == "Base" else range(4, 24 if family != "nrp" else 22, 2)
        plan = [(ds, n) for ds in (45, 46, 47, 48, 49) for n in sizes]
        if family == "tsm_aleatoria" and label in ("Base", "F"):      # bloque adicional de H7
            plan += [(ds, 8) for ds in range(50, 60)]
        for ds, n in plan:
            if True:
                inst = f"{family}_n{n:02d}_d{ds}"
                p_unif = 2.0 ** -n
                base = dict(family=family, n=n, instance=inst, config=config, alpha=alpha, init=init,
                            data_seed=ds)
                rows.append(dict(base, method="uniforme", p_opt=p_unif, p_feasible=0.3, n_opt_states=1))
                for seed in range(100, 110):
                    rows.append(dict(base, method="qaoa", seed=seed, p_opt=p_unif * fn(n, seed, ds),
                                     p_feasible=0.4, optimal=1, secs=1.0,
                                     warm=1.0 if (label != "F+C+I" or n > 4) else 0.0))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic(ae):
    amp = {
        ("tsm_aleatoria", "Base"): lambda n, s, d: 1.0 if n < 8 else 0.01,
        ("tsm_aleatoria", "F"): lambda n, s, d: 2.0 + 0.01 * d + 0.1 * (s % 3),
        ("tsm_aleatoria", "F+C"): lambda n, s, d: (20.0 + d + n) * (1.0 + 0.3 * (s % 5)),
        ("tsm_aleatoria", "F+C+I"): lambda n, s, d: (40.0 + d + 3 * n) * (1.0 + 0.01 * (s % 5)),
        ("nrp", "F+C+I"): lambda n, s, d: 1.0 if (n + d) % 4 else 0.8,
    }
    runs = _fake_runs(amp)
    runs["cfg"] = runs.apply(ae.config_label, axis=1)
    return ae.summarize_instances(runs)


def test_resumen_por_instancia(ae, synthetic):
    s = synthetic
    assert set(s.cfg) == {"Base", "F", "F+C", "F+C+I"}
    assert (s.n_seeds == 10).all()
    row = s[(s.instance == "tsm_aleatoria_n06_d45") & (s.cfg == "Base")].iloc[0]
    assert row.A_median == pytest.approx(1.0) and row.logA_median == pytest.approx(0.0)
    row = s[(s.instance == "tsm_aleatoria_n08_d45") & (s.cfg == "Base")].iloc[0]
    assert row.A_median == pytest.approx(0.01)
    assert row.p_see_unif == pytest.approx(1 - (1 - 2.0 ** -8) ** 8000)
    assert not ae.check_integrity(s, expected_seeds=10)     # N >= 8 con warm = 1


def test_hipotesis_en_datos_sinteticos(ae, synthetic):
    t = ae.run_hypotheses(synthetic).set_index("hipotesis")
    assert list(t.index) == ae.HOLM_FAMILY
    assert t.loc["H1", "n"] == 40 and t.loc["H6", "n"] == 35
    assert t.loc["H7", "n"] == 15                           # N = 8: 5 instancias + 10 del bloque adicional
    assert t.loc["H2", "n"] == 40                           # el bloque de H7 no entra en H2
    for h in ("H1", "H2", "H3", "H4", "H7"):
        assert t.loc[h, "decision"] == "Rechaza H0", h
    assert t.loc["H6", "decision"] == "No rechaza H0"       # NRP: A <= 1 por construccion
    assert (t["p_holm"] >= t["p"] - 1e-15).all()
    assert t.loc["H2", "efecto_valor"] == 1.0               # F+C gana en todas las instancias


def test_integridad_detecta_transferencia_rota_y_semillas_faltantes(ae, synthetic):
    broken = synthetic.copy()
    idx = broken[(broken.cfg == "F+C+I") & (broken.n == 22)].index[0]
    broken.loc[idx, "warm_all"] = False
    broken.loc[broken.index[0], "n_seeds"] = 3
    problems = ae.check_integrity(broken, expected_seeds=10)
    assert any("warm=0" in p for p in problems)
    assert any("3 semillas" in p for p in problems)


def test_secundario_holguras_en_tamanos_chicos(ae, synthetic):
    sec = ae.run_secondary(synthetic)
    row = sec[sec.analisis.str.startswith("Sin holguras vs con holguras")].iloc[0]
    assert row.n == 10                                      # N = 4 y 6, 5 semillas de datos


def test_secundario_tsm_frente_a_nrp(ae, synthetic):
    sec = ae.run_secondary(synthetic)
    row = sec[sec.analisis.str.startswith("TSM aleatoria vs NRP")].iloc[0]
    assert "A12 = 1.00" in row.efecto and row.p < 0.001


# ---------------------------------------------------------------------------
# H5
# ---------------------------------------------------------------------------
def test_hardware_junta_replicas_y_aplica_la_regla_de_potencia(ae):
    hw = pd.DataFrame({
        "instance": ["a", "a", "a", "b", "b", "b"],
        "shots": [4000] * 6,
        "p_feasible": [0.20, 0.21, 0.19, 0.30, 0.30, 0.30],
        "p_opt_sampled": [0.0010, 0.0015, 0.0005, 0.0, 0.00025, 0.0],
        "sim_p_feasible": [0.40] * 3 + [0.30] * 3,
        "sim_p_opt": [0.0030] * 3 + [0.0003] * 3,
    })
    out = ae.hardware_tests(hw).set_index(["instancia", "medida"])
    a_feas = out.loc[("a", "p_feasible")]
    assert a_feas.mediciones == 12000 and a_feas.k == 2400
    assert a_feas.p_hw == pytest.approx(0.20) and a_feas.cociente == pytest.approx(0.5)
    assert a_feas.conclusion == "hardware < simulador"
    a_opt = out.loc[("a", "p_opt")]
    assert a_opt.k == 12 and a_opt.esperados == pytest.approx(36.0)
    assert a_opt.conclusion == "hardware < simulador"
    b_feas = out.loc[("b", "p_feasible")]
    assert b_feas.conclusion == "compatible con el simulador"
    b_opt = out.loc[("b", "p_opt")]
    assert b_opt.esperados == pytest.approx(3.6)
    assert b_opt.conclusion.startswith("no concluyente")
    assert a_feas.wilson_inf < 0.20 < a_feas.wilson_sup
