"""Pruebas de la logica base del motor QAOA (TSM y NRP).

Cubren los puntos de la auditoria que pueden verificarse de forma exacta:
  * orden de asignacion de parametros gamma/beta en el circuito;
  * endianness al decodificar bitstrings de Qiskit;
  * equivalencia entre build_value_lookup y qubo_value;
  * el argmin del QUBO (fuerza bruta) es factible y coincide con el ILP.

Ejecutar desde codigo/:  python -m pytest tests -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _problem(problem):
    base = ROOT / problem.upper()
    sys.path.insert(0, str(base))
    data_io = _load(base / "data_io.py", "data_io")
    engine = _load(base / "quantum" / "qaoa_engine.py", f"{problem}_engine")
    qubo_mod = _load(base / "quantum" / f"{problem}_qubo.py", f"{problem}_qubo")
    sys.path.pop(0)
    return data_io, engine, qubo_mod


def _tsm(dataset):
    data_io, engine, qubo_mod = _problem("tsm")
    df_cov, df_tests = data_io.load_tsm(dataset)
    qp, names = qubo_mod.build_qp(df_cov, df_tests)
    qubo = qubo_mod.build_qubo(qp, penalty=10.0 * (len(names) + 1))
    validate = lambda sel: data_io.is_feasible(sel, df_cov)
    return engine, qubo_mod, qubo, names, validate, df_cov


def _nrp(dataset, budget):
    data_io, engine, qubo_mod = _problem("nrp")
    df_feat, df_deps = data_io.load_nrp(dataset)
    qp, names = qubo_mod.build_qp(df_feat, df_deps, budget)
    qubo = qubo_mod.build_qubo(qp, penalty=float(len(names) + 1))
    validate = lambda sel: data_io.is_feasible(sel, df_feat, df_deps, budget)
    return engine, qubo_mod, qubo, names, validate, (df_feat, df_deps)


# ---------------------------------------------------------------------------
# Orden de parametros
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("problem", ["tsm", "nrp"])
def test_initial_params_land_on_intended_parameters(problem):
    """gamma debe recibir el valor generado para gamma y beta el de beta.

    `initial_params` genera [gamma_0, beta_0, gamma_1, beta_1, ...]. El motor
    asigna con dict(zip(circuit.parameters, params)), y Qiskit ordena
    `circuit.parameters` alfabeticamente, asi que el orden debe coincidir.
    """
    _, engine, _ = _problem(problem)
    from qiskit.quantum_info import SparsePauliOp

    op = SparsePauliOp.from_list([("ZZ", 1.0), ("ZI", 0.5)])
    reps = 2
    ansatz = engine.build_ansatz(op, reps)
    circuit = engine.prepare_circuit(None, ansatz)
    params = engine.initial_params(reps, np.random.default_rng(0))

    names = [p.name for p in circuit.parameters]
    # Lo que hacia el motor antes (zip posicional) cruza gamma y beta:
    assert names == sorted(names)
    # Lo que hace ahora (bind_params): cada valor va a su parametro.
    bound = engine.bind_params(circuit, params)
    assert not bound.parameters
    expected = {}
    for layer in range(reps):
        expected[f"gamma_{layer}"] = params[2 * layer]
        expected[f"beta_{layer}"] = params[2 * layer + 1]
    ref = circuit.assign_parameters({p: expected[p.name] for p in circuit.parameters})
    assert bound == ref


# ---------------------------------------------------------------------------
# Endianness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("problem", ["tsm", "nrp"])
def test_decode_full_endianness(problem):
    """Un estado |x> preparado con X en el qubit i decodifica variable i = 1."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_optimization import QuadraticProgram

    _, engine, _ = _problem(problem)
    qp = QuadraticProgram()
    for k in range(4):
        qp.binary_var(name=f"v{k}")
    qp.minimize(linear={f"v{k}": 1.0 for k in range(4)})

    sim = AerSimulator()
    for i in range(4):
        qc = QuantumCircuit(4)
        qc.x(i)
        qc.measure_all()
        counts = sim.run(qc, shots=10).result().get_counts()
        (bitstring,) = counts.keys()
        full = engine.decode_full(bitstring, qp)
        assert full == {f"v{k}": int(k == i) for k in range(4)}


# ---------------------------------------------------------------------------
# value_lookup == qubo_value
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("build", [lambda: _tsm("toy2"), lambda: _nrp("demo", 60)])
def test_value_lookup_matches_qubo_value(build):
    engine, qubo_mod, qubo, names, _, _ = build()
    values = engine.build_value_lookup(qubo)
    n = qubo.get_num_vars()
    rng = np.random.default_rng(1)
    var_names = [v.name for v in qubo.variables]
    for _ in range(50):
        idx = int(rng.integers(0, 1 << n))
        assignment = {var_names[k]: (idx >> k) & 1 for k in range(n)}
        assert np.isclose(values[idx], qubo_mod.qubo_value(assignment, qubo))


# ---------------------------------------------------------------------------
# Correccion de la formulacion: argmin QUBO factible y == ILP
# ---------------------------------------------------------------------------
def _brute_force_argmin(engine, qubo, names):
    values = engine.build_value_lookup(qubo)
    var_names = [v.name for v in qubo.variables]
    idx = int(np.argmin(values))
    sel = [nm for nm in names if (idx >> var_names.index(nm)) & 1]
    return sel, float(values[idx])


@pytest.mark.parametrize("dataset,expected_tests", [("toy", 2), ("toy2", 3), ("demo", 6)])
def test_tsm_qubo_argmin_is_feasible_and_optimal(dataset, expected_tests):
    engine, _, qubo, names, validate, _ = _tsm(dataset)
    sel, _ = _brute_force_argmin(engine, qubo, names)
    assert validate(sel), sel
    assert len(sel) == expected_tests, sel   # optimo ILP de la instancia


@pytest.mark.parametrize("dataset,budget,expected_value", [("toy", 100, 160), ("demo", 60, 268)])
def test_nrp_qubo_argmin_is_feasible_and_optimal(dataset, budget, expected_value):
    engine, _, qubo, names, validate, (df_feat, _) = _nrp(dataset, budget)
    sel, _ = _brute_force_argmin(engine, qubo, names)
    assert validate(sel), sel
    assert float(df_feat.loc[sel, "Valor"].sum()) == expected_value, sel


# ---------------------------------------------------------------------------
# Fase 4: formulacion desbalanceada (sin holguras)
# ---------------------------------------------------------------------------
def _tsm_from_frame(data_io, qubo_mod, df_cov, df_tests):
    qp, names = qubo_mod.build_qp(df_cov, df_tests)
    validate = lambda sel: data_io.is_feasible(sel, df_cov)
    objective = lambda sel: data_io.evaluate_selection(sel, df_cov, df_tests)["objective"]
    return qp, names, validate, objective


def _ciclo_instance(n, seed=42):
    """Instancia en ciclo (cada requisito cubierto por 2 tests consecutivos)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    tests = [f"T{i}" for i in range(n)]
    reqs = [f"R{j}" for j in range(n)]
    m = np.zeros((n, n), dtype=int)
    for j in range(n):
        m[j, j] = 1
        m[(j + 1) % n, j] = 1
    df_cov = pd.DataFrame(m, index=tests, columns=reqs)
    df_tests = pd.DataFrame({"Test": tests, "time": np.round(rng.uniform(0.5, 2.0, n), 2)})
    df_tests.index = df_cov.index
    return df_cov, df_tests


def _coeffs(qubo):
    names = [v.name for v in qubo.variables]
    lin = {names[i]: round(c, 9) for i, c in qubo.objective.linear.to_dict().items() if c != 0}
    quad = {}
    for (i, j), c in qubo.objective.quadratic.to_dict().items():
        if c != 0:
            key = tuple(sorted((names[i], names[j])))
            quad[key] = round(quad.get(key, 0.0) + c, 9)
    return round(qubo.objective.constant, 9), lin, quad


def test_tsm_unbalanced_equals_exact_penalty_on_cycle():
    """Con 2 tests por requisito y l1 = l2 = P/2 la penalizacion desbalanceada
    coincide coeficiente a coeficiente con la exacta P(1-x_a)(1-x_b) que
    QuadraticProgramToQubo usa sin holguras (familia ciclo)."""
    data_io, _, qubo_mod = _problem("tsm")
    df_cov, df_tests = _ciclo_instance(6)
    qp, names, _, _ = _tsm_from_frame(data_io, qubo_mod, df_cov, df_tests)
    P = 10.0 * (len(names) + 1)
    exact = qubo_mod.build_qubo(qp, penalty=P)
    unb = qubo_mod.build_qubo_unbalanced(qp, P / 2, P / 2)
    assert exact.get_num_vars() == len(names)      # ciclo: sin holguras
    assert _coeffs(exact) == _coeffs(unb)


@pytest.mark.parametrize("dataset", ["toy", "toy2", "demo"])
def test_tsm_unbalanced_has_n_vars_and_exact_value_when_h_is_zero(dataset):
    """El QUBO desbalanceado tiene exactamente N variables (sin holguras) y,
    en una asignacion que cubre cada requisito exactamente una vez (h_i = 0),
    su valor es el objetivo original. Si no existe tal asignacion en el
    dataset, se comprueba la formula zeta por requisito sobre asignaciones
    aleatorias."""
    data_io, engine, qubo_mod = _problem("tsm")
    df_cov, df_tests = data_io.load_tsm(dataset)
    qp, names, _, objective = _tsm_from_frame(data_io, qubo_mod, df_cov, df_tests)
    l1, l2 = 1.5, 0.75
    unb = qubo_mod.build_qubo_unbalanced(qp, l1, l2)
    assert unb.get_num_vars() == len(names)
    assert [v.name for v in unb.variables] == names
    assert not unb.linear_constraints and not unb.quadratic_constraints

    rng = np.random.default_rng(3)
    for _ in range(30):
        bits = rng.integers(0, 2, len(names))
        sel = [nm for nm, b in zip(names, bits) if b]
        expected = objective(sel)
        for req in df_cov.columns:
            h = int(sum(df_cov.loc[t, req] for t in sel)) - 1
            expected += -l1 * h + l2 * h * h
        value = qubo_mod.qubo_value({nm: int(b) for nm, b in zip(names, bits)}, unb)
        assert np.isclose(value, expected), (sel, value, expected)


@pytest.mark.skipif(
    _problem("tsm")[2].DEFAULT_LAMBDA1 is None, reason="lambdas aun no calibrados (Fase 4)")
@pytest.mark.parametrize("dataset,expected_tests", [("toy", 2), ("toy2", 3), ("demo", 6)])
def test_tsm_unbalanced_argmin_is_feasible_and_optimal(dataset, expected_tests):
    """Con los lambdas congelados tras la calibracion, el argmin del QUBO
    desbalanceado debe ser factible y coincidir con el optimo del ILP."""
    data_io, engine, qubo_mod = _problem("tsm")
    df_cov, df_tests = data_io.load_tsm(dataset)
    qp, names, validate, _ = _tsm_from_frame(data_io, qubo_mod, df_cov, df_tests)
    unb = qubo_mod.build_qubo_unbalanced(qp, qubo_mod.DEFAULT_LAMBDA1, qubo_mod.DEFAULT_LAMBDA2)
    sel, _ = _brute_force_argmin(engine, unb, names)
    assert validate(sel), sel
    assert len(sel) == expected_tests, sel


@pytest.mark.parametrize("penalty_logic", [None, "lambda", 3.5])
@pytest.mark.parametrize("dataset,budget", [("toy", 100), ("demo", 60)])
def test_nrp_unbalanced_has_n_vars_and_matches_formula(dataset, budget, penalty_logic):
    """El QUBO desbalanceado de NRP tiene N variables y, en asignaciones
    aleatorias, vale objetivo + zeta(h_presupuesto) + P * (violaciones
    REQUIRES/EXCLUDES), con h y las violaciones leidas de las restricciones
    del QuadraticProgram original. P es independiente de lambda: por defecto
    DEFAULT_PENALTY_LOGIC; "lambda" reproduce el P = l1 + l2 del experimento E3."""
    data_io, engine, qubo_mod = _problem("nrp")
    df_feat, df_deps = data_io.load_nrp(dataset)
    qp, names = qubo_mod.build_qp(df_feat, df_deps, budget)
    l1, l2 = 1.0, 0.1
    unb = qubo_mod.build_qubo_unbalanced(qp, l1, l2, penalty_logic=penalty_logic)
    P = {None: qubo_mod.DEFAULT_PENALTY_LOGIC, "lambda": l1 + l2}.get(penalty_logic, penalty_logic)
    assert qubo_mod.resolve_penalty_logic(l1, l2, penalty_logic) == P
    assert unb.get_num_vars() == len(names)
    assert [v.name for v in unb.variables] == names
    assert not unb.linear_constraints

    rng = np.random.default_rng(7)
    for _ in range(40):
        x = rng.integers(0, 2, len(names)).astype(float)
        expected = qp.objective.evaluate(x)
        for cons in qp.linear_constraints:
            lhs = cons.linear.evaluate(x)
            if cons.name == "Limite_Presupuesto":
                h = cons.rhs - lhs
                expected += -l1 * h + l2 * h * h
            else:
                expected += P * max(0.0, lhs - cons.rhs)
        value = qubo_mod.qubo_value({nm: int(b) for nm, b in zip(names, x)}, unb)
        assert np.isclose(value, expected), (x, value, expected)


@pytest.mark.skipif(
    _problem("nrp")[2].DEFAULT_LAMBDA1 is None, reason="lambdas de NRP aun no calibrados (Fase 4)")
@pytest.mark.parametrize("dataset,budget,expected_value", [("toy", 100, 160), ("demo", 60, 268)])
def test_nrp_unbalanced_optimum_is_among_lowest_states(dataset, budget, expected_value):
    """Para NRP la formulacion desbalanceada NO es exacta (ver nrp_qubo.py):
    la garantia documentada tras la calibracion es que el optimo real del ILP
    queda entre los 30 estados de menor energia del QUBO."""
    data_io, engine, qubo_mod = _problem("nrp")
    df_feat, df_deps = data_io.load_nrp(dataset)
    qp, names = qubo_mod.build_qp(df_feat, df_deps, budget)
    unb = qubo_mod.build_qubo_unbalanced(qp, qubo_mod.DEFAULT_LAMBDA1, qubo_mod.DEFAULT_LAMBDA2)
    values = engine.build_value_lookup(unb)
    n = len(names)
    best_opt = np.inf
    for key in range(1 << n):
        sel = [names[k] for k in range(n) if (key >> k) & 1]
        if float(df_feat.loc[sel, "Valor"].sum()) == expected_value and \
                data_io.is_feasible(sel, df_feat, df_deps, budget):
            best_opt = min(best_opt, values[key])
    assert np.isfinite(best_opt)
    rank = int(np.count_nonzero(values < best_opt - 1e-9))
    assert rank <= 30, rank


# ---------------------------------------------------------------------------
# Fase 4: objetivo CVaR e inicializacion INTERP
# ---------------------------------------------------------------------------
def test_cvar_on_known_distribution():
    """Valores 1..4 equiprobables: CVaR_0.5 = media de {1, 2} = 1.5;
    CVaR_0.25 = 1; CVaR_1 = media = 2.5; alpha que corta un estado a la
    mitad (0.375) pondera: (0.25*1 + 0.125*2) / 0.375 = 4/3."""
    _, engine, _ = _problem("tsm")
    values = np.array([3.0, 1.0, 4.0, 2.0])
    probs = np.full(4, 0.25)
    assert np.isclose(engine.cvar_from_probs(probs, values, 0.5), 1.5)
    assert np.isclose(engine.cvar_from_probs(probs, values, 0.25), 1.0)
    assert np.isclose(engine.cvar_from_probs(probs, values, 1.0), 2.5)
    assert np.isclose(engine.cvar_from_probs(probs, values, 0.375), 4.0 / 3.0)
    order = np.argsort(values)
    assert np.isclose(engine.cvar_from_probs(probs, values, 0.5, order=order), 1.5)


def test_cvar_from_counts_matches_probs():
    """La version por muestreo coincide con la exacta sobre la misma distribucion."""
    _, engine, qubo_mod = _problem("tsm")
    _, _, qubo, names, _, _ = _tsm("toy")
    values = engine.build_value_lookup(qubo)
    n = qubo.get_num_vars()
    rng = np.random.default_rng(5)
    states = rng.choice(1 << n, size=6, replace=False)
    counts = {format(int(s), f"0{n}b"): int(c) for s, c in zip(states, [10, 20, 5, 40, 15, 10])}
    probs = np.zeros(values.size)
    for s, c in zip(states, [10, 20, 5, 40, 15, 10]):
        probs[int(s)] = c / 100.0
    for alpha in (0.1, 0.25, 0.5, 1.0):
        assert np.isclose(engine.cvar_from_counts(counts, qubo, qubo_mod.qubo_value, alpha),
                          engine.cvar_from_probs(probs, values, alpha))


def test_evaluate_energy_cvar_alpha_one_equals_energy():
    """CVaR_1 debe coincidir con la energia esperada (statevector exacto)."""
    _, engine, qubo_mod = _problem("tsm")
    _, _, qubo, names, _, _ = _tsm("toy")
    operator, offset = qubo.to_ising()
    backend = engine.AerBackend(method="statevector", seed=1)
    circuit = engine.prepare_circuit(backend, engine.build_ansatz(operator, 1))
    values = engine.build_value_lookup(qubo)
    p = np.array([0.7, 0.3])
    e = engine.evaluate_energy(backend, circuit, p, qubo, qubo_mod.qubo_value, 100,
                               exact=True, value_lookup=values)
    c1 = engine.evaluate_energy(backend, circuit, p, qubo, qubo_mod.qubo_value, 100,
                                exact=True, value_lookup=values, objective="cvar", alpha=1.0)
    c_small = engine.evaluate_energy(backend, circuit, p, qubo, qubo_mod.qubo_value, 100,
                                     exact=True, value_lookup=values, objective="cvar", alpha=0.1)
    assert np.isclose(e, c1)
    assert c_small <= e + 1e-9      # la cola izquierda nunca supera la media


def test_interp_params_zhou():
    """INTERP de p=1 a 2 duplica el valor; de p=2 a 3 interpola linealmente."""
    _, engine, _ = _problem("tsm")
    p1 = np.array([0.4, 0.9])                     # [gamma_0, beta_0]
    p2 = engine.interp_params(p1)
    assert np.allclose(p2, [0.4, 0.9, 0.4, 0.9])
    p2 = np.array([0.2, 0.8, 0.6, 0.4])           # gammas (0.2, 0.6), betas (0.8, 0.4)
    p3 = engine.interp_params(p2)
    assert np.allclose(p3[0::2], [0.2, 0.4, 0.6])
    assert np.allclose(p3[1::2], [0.8, 0.6, 0.4])
    # initial_params con transfer lleva de 1 a 3 capas; random sigue igual
    t = engine.initial_params(3, None, init="transfer", warm=p1)
    assert len(t) == 6
    r = engine.initial_params(2, np.random.default_rng(0))
    assert len(r) == 4 and np.all(r >= 0)
    with pytest.raises(ValueError):
        engine.initial_params(1, None, init="transfer", warm=p2)


# ---------------------------------------------------------------------------
# extract_best: entre factibles gana el mejor objetivo original, no el QUBO
# ---------------------------------------------------------------------------
class _FakeBackend:
    def __init__(self, counts):
        self.counts = counts

    def sample(self, circuit, shots):
        return self.counts


@pytest.mark.parametrize("problem", ["tsm", "nrp"])
def test_extract_best_ranks_feasible_by_original_objective(problem):
    """Una solucion optima con holguras 'mal puestas' (valor QUBO alto) debe
    ganar a una factible peor con holguras correctas (valor QUBO bajo)."""
    _, engine, _ = _problem(problem)
    _, qubo_mod, qubo, names, validate, _ = _tsm("toy2")
    operator, _ = qubo.to_ising()
    circuit = engine.prepare_circuit(None, engine.build_ansatz(operator, 1))
    n = qubo.get_num_vars()
    var_names = [v.name for v in qubo.variables]

    def bitstring(selected, slack_bits):
        bits = ["0"] * n
        for nm in selected:
            bits[var_names.index(nm)] = "1"
        for k, b in enumerate(slack_bits):
            bits[len(names) + k] = b
        return "".join(reversed(bits))  # little-endian de Qiskit

    opt_bad_slack = bitstring(["T1", "T2", "T3"], "1" * (n - len(names)))   # optimo (3 tests)
    # Solucion peor (7 tests) con las holguras que minimizan su valor QUBO
    values = engine.build_value_lookup(qubo)
    dec_mask = sum(1 << var_names.index(t) for t in ["T0", "T4", "T5", "T6", "T7", "T8", "T9"])
    best_idx = min((i for i in range(len(values)) if (i & ((1 << len(names)) - 1)) == dec_mask), key=lambda i: values[i])
    worse_good = format(best_idx, f"0{n}b")

    counts = {opt_bad_slack: 1, worse_good: 50}
    objective = len
    assignment, _, feasible, _ = engine.extract_best(
        _FakeBackend(counts), circuit, [0.1, 0.2], qubo, names, qubo_mod.qubo_value,
        validate, 51, objective=objective,
    )
    assert feasible
    assert [t for t in names if assignment[t] == 1] == ["T1", "T2", "T3"]


@pytest.mark.parametrize("problem", ["tsm", "nrp"])
def test_extract_best_passes_selected_list_to_validate(problem):
    """`validate` y `objective` deben recibir la lista de seleccionados, no el
    dict {nombre: 0/1} (iterar el dict daba por seleccionadas TODAS las
    variables, de modo que un conjunto vacio podia ser elegido)."""
    _, engine, _ = _problem(problem)
    _, qubo_mod, qubo, names, validate, _ = _tsm("toy")
    operator, _ = qubo.to_ising()
    circuit = engine.prepare_circuit(None, engine.build_ansatz(operator, 1))
    n = qubo.get_num_vars()
    empty = "0" * n                                   # infactible (no cubre nada)
    opt = "".join(reversed(["0", "1", "1", "0", "0"]))  # T1,T2 cubre todo (2 tests)
    received = []

    def spy_validate(sel):
        received.append(sel)
        return validate(sel)

    assignment, _, feasible, _ = engine.extract_best(
        _FakeBackend({empty: 100, opt: 1}), circuit, [0.1, 0.2], qubo, names,
        qubo_mod.qubo_value, spy_validate, 101, objective=len,
    )
    assert all(isinstance(s, list) for s in received)
    assert feasible
    assert [t for t in names if assignment[t] == 1] == ["T1", "T2"]
