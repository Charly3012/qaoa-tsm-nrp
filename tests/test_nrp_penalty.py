"""Pruebas de la penalizacion de dependencias (REQUIRES / EXCLUDES) del NRP
sin holguras, separada de los lambdas del presupuesto (deuda tecnica #7).

Ejecutar desde codigo/:  python -m pytest tests -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
BENCH = [("toy", 100), ("demo", 60)]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def nrp():
    base = ROOT / "NRP"
    sys.path.insert(0, str(base))
    data_io = _load(base / "data_io.py", "data_io")
    qubo_mod = _load(base / "quantum" / "nrp_qubo.py", "nrp_qubo")
    sys.path.pop(0)
    return data_io, qubo_mod


def _enumerate(qubo_mod, qp, names, qubo):
    """Energia del QUBO y numero de dependencias violadas de los 2**N estados."""
    n = len(names)
    energies = np.empty(1 << n)
    violations = np.zeros(1 << n, dtype=int)
    for key in range(1 << n):
        x = [(key >> k) & 1 for k in range(n)]
        energies[key] = qubo_mod.qubo_value(dict(zip(names, x)), qubo)
        for cons in qp.linear_constraints:
            if cons.name != "Limite_Presupuesto" and cons.linear.evaluate(x) > cons.rhs:
                violations[key] += 1
    return energies, violations


def test_penalizacion_por_defecto_supera_al_objetivo(nrp):
    """P por defecto debe ser mayor que cualquier coeficiente del objetivo
    normalizado; la variante heredada (lambda1 + lambda2 = 0.11) no lo es."""
    data_io, qubo_mod = nrp
    l1, l2 = qubo_mod.DEFAULT_LAMBDA1, qubo_mod.DEFAULT_LAMBDA2
    assert qubo_mod.resolve_penalty_logic(l1, l2) == qubo_mod.DEFAULT_PENALTY_LOGIC
    assert qubo_mod.resolve_penalty_logic(l1, l2, "lambda") == pytest.approx(l1 + l2)
    assert qubo_mod.resolve_penalty_logic(l1, l2, 3.5) == 3.5
    for dataset, budget in BENCH:
        df_feat, df_deps = data_io.load_nrp(dataset)
        qp, _ = qubo_mod.build_qp(df_feat, df_deps, budget)
        coef_max = max(abs(c) for c in qp.objective.linear.to_dict().values())
        assert qubo_mod.DEFAULT_PENALTY_LOGIC > coef_max
        assert qubo_mod.resolve_penalty_logic(l1, l2, "lambda") < coef_max


@pytest.mark.parametrize("dataset,budget", BENCH)
def test_p_solo_cambia_la_energia_de_los_estados_que_violan_dependencias(nrp, dataset, budget):
    """E_P2(x) - E_heredada(x) = (2 - 0.11) * violaciones(x): la separacion de P
    no toca el objetivo ni el termino del presupuesto."""
    data_io, qubo_mod = nrp
    df_feat, df_deps = data_io.load_nrp(dataset)
    qp, names = qubo_mod.build_qp(df_feat, df_deps, budget)
    l1, l2 = qubo_mod.DEFAULT_LAMBDA1, qubo_mod.DEFAULT_LAMBDA2
    new, viol = _enumerate(qubo_mod, qp, names, qubo_mod.build_qubo_unbalanced(qp, l1, l2))
    old, _ = _enumerate(qubo_mod, qp, names,
                        qubo_mod.build_qubo_unbalanced(qp, l1, l2, penalty_logic="lambda"))
    assert viol.max() > 0                                    # la instancia tiene dependencias
    assert np.allclose(new - old, (qubo_mod.DEFAULT_PENALTY_LOGIC - (l1 + l2)) * viol)


@pytest.mark.parametrize("dataset,budget", BENCH)
def test_calibracion_y_qubo_usan_la_misma_energia(nrp, dataset, budget):
    """sweep_lambda.py calcula la energia de forma vectorizada; debe coincidir
    con build_qubo_unbalanced para cualquier P (si no, se calibraria un QUBO
    distinto del que luego se ejecuta)."""
    data_io, qubo_mod = nrp
    sweep = _load(ROOT / "scripts" / "sweep_lambda.py", "sweep_lambda_test")
    problem = sweep.Problem("nrp", None)
    pre = problem.precompute(dataset, budget, None)
    names = pre["names"]
    rng = np.random.default_rng(11)
    keys = rng.integers(0, 1 << len(names), 60)
    for l1, l2, plog in [(0.1, 0.01, None), (0.1, 0.01, "lambda"), (0.5, 0.05, 2.0), (1.0, 0.2, 0.7)]:
        P = problem.penalty_logic(l1, l2, plog)
        vector = pre["obj"] + l1 * pre["A"] + l2 * pre["B"] + P * pre["V"]
        qubo = problem.qm.build_qubo_unbalanced(pre["qp"], l1, l2, penalty_logic=plog)
        for key in keys:
            x = {nm: int((key >> k) & 1) for k, nm in enumerate(names)}
            assert np.isclose(vector[key], problem.qm.qubo_value(x, qubo)), (l1, l2, plog, int(key))
