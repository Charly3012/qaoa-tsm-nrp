"""Construcción del modelo QuadraticProgram y su QUBO para TSM.

TSM se modela como un problema de cobertura de conjuntos:

    min  sum_t x_t + w * sum_t time_t * x_t          (x_t binario por test)
    s.t. sum_{t: cubre j} x_t >= 1  para cada requisito j

Las restricciones de cobertura (>=) se transforman en penalizaciones de la
función objetivo QUBO mediante `QuadraticProgramToQubo`. qiskit-optimization
>= 0.7 resuelve las desigualdades de forma exacta:
  * Las formas especiales (p. ej. sum x >= n-1) se convierten en una
    penalización cuadrática directa sin variables de holgura.
  * Las desigualdades generales se convierten en igualdades con variables de
    holgura (InequalityToEquality + IntegerToBinary) y se penalizan.

El peso de penalización debe dominar el rango posible del objetivo para que
ninguna solución infactible compita con una factible.
"""
import numpy as np

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from data_io import TIME_WEIGHT


def build_qp(df_coverage, df_tests, time_weight=TIME_WEIGHT):
    """Construye el QuadraticProgram binario del TSM.

    Devuelve (qp, var_names) donde var_names es el orden de las variables
    originales (los tests).
    """
    tests = df_coverage.index.tolist()
    var_names = tests

    qp = QuadraticProgram("tsm")
    x = {t: qp.binary_var(name=t) for t in tests}

    linear = {t: 1.0 + time_weight * float(df_tests.loc[t, "time"]) for t in tests}
    qp.minimize(linear=linear)

    for req in df_coverage.columns:
        covering = [t for t in tests if int(df_coverage.loc[t, req]) == 1]
        if not covering:
            # Requisito sin tests que lo cubran: modelo infactible.
            continue
        qp.linear_constraint(
            linear={t: 1.0 for t in covering},
            sense=">=",
            rhs=1,
            name=f"cov_{req}",
        )
    return qp, var_names


def build_qubo(qp, penalty=None):
    """Convierte el QuadraticProgram a QUBO aplicando penalizaciones.

    penalty=None usa la heurística automática de qiskit-optimization.
    Devuelve el QuadraticProgram QUBO resultante.
    """
    return QuadraticProgramToQubo(penalty=penalty).convert(qp)


# Valores fijados en la calibración de lambda1/lambda2 (scripts/sweep_lambda.py,
# resultados en report/fase4/sweep_lambda*.csv): con l1 = l2 = 1 el argmin es
# factible y óptimo en las 8 instancias de calibración y en 11/12 no vistas
# (falla tsm_aleatoria N=24: el óptimo queda en la posición 63 de 2**24).
# En la diagonal zeta(h) = l*h*(h-1): 0 si h in {0, 1}, 2l si h = -1 o h = 2.
DEFAULT_LAMBDA1 = 1.0
DEFAULT_LAMBDA2 = 1.0


def build_qubo_unbalanced(qp, lambda1, lambda2):
    """QUBO de TSM con penalización desbalanceada (sin variables de holgura).

    Montañez-Barrera et al. (2024): cada restricción de desigualdad
    h_i(x) = sum_{j in C_i} x_j - 1 >= 0 se penaliza con

        zeta_i(x) = -lambda1 * h_i(x) + lambda2 * h_i(x)^2

    que vale lambda1 + lambda2 si el requisito queda sin cubrir (h = -1),
    0 si se cubre exactamente una vez (h = 0) y lambda2 - lambda1 si se
    cubre dos veces (h = 1). Usando x^2 = x, la forma cerrada es

        zeta_i = (l1 + l2) - (l1 + l2) * sum_{j in C_i} x_j
                 + 2 * l2 * sum_{j < k in C_i} x_j x_k

    Con |C_i| = 2 y l1 = l2 = P/2 coincide con la penalización exacta
    P (1 - x_a)(1 - x_b) que usa QuadraticProgramToQubo (familia ciclo).
    Para |C_i| >= 3 NO es exacta: el argmin puede no ser el óptimo del ILP
    y debe verificarse por fuerza bruta.

    Devuelve un QuadraticProgram sin restricciones con exactamente las N
    variables de decisión de `qp` (mismo orden), de modo que to_ising,
    build_value_lookup y decode_full funcionan sin cambios.
    """
    l1, l2 = float(lambda1), float(lambda2)
    names = [v.name for v in qp.variables]
    pos = {name: i for i, name in enumerate(names)}

    linear = np.zeros(len(names))
    for i, coeff in qp.objective.linear.to_dict().items():
        linear[i] += coeff
    quadratic = {}
    for (i, j), coeff in qp.objective.quadratic.to_dict().items():
        quadratic[(i, j)] = quadratic.get((i, j), 0.0) + coeff
    constant = float(qp.objective.constant)

    for cons in qp.linear_constraints:
        if cons.sense.name != "GE" or cons.rhs != 1:
            raise ValueError(
                f"build_qubo_unbalanced solo admite restricciones de cobertura "
                f"sum x >= 1 (restriccion {cons.name}: {cons.sense.name} {cons.rhs})"
            )
        covering = sorted(pos[name] for name, c in cons.linear.to_dict(use_name=True).items()
                          if c != 0)
        constant += l1 + l2
        for j in covering:
            linear[j] -= l1 + l2
        for a in range(len(covering)):
            for b in range(a + 1, len(covering)):
                key = (covering[a], covering[b])
                quadratic[key] = quadratic.get(key, 0.0) + 2.0 * l2

    qubo = QuadraticProgram(qp.name + "_unbalanced")
    for name in names:
        qubo.binary_var(name=name)
    qubo.minimize(
        constant=constant,
        linear={names[i]: float(c) for i, c in enumerate(linear) if c != 0},
        quadratic={(names[i], names[j]): float(c) for (i, j), c in quadratic.items() if c != 0},
    )
    return qubo


def qubo_value(assignment, qubo):
    """Evalúa la función objetivo QUBO dada una asignación completa
    {nombre_variable: 0/1} sobre TODAS las variables del QUBO (incl. holguras)."""
    linear = qubo.objective.linear.to_dict()
    quadratic = qubo.objective.quadratic.to_dict()
    value = qubo.objective.constant
    for i, coeff in linear.items():
        value += coeff * assignment[qubo.variables[i].name]
    for (i, j), coeff in quadratic.items():
        value += coeff * assignment[qubo.variables[i].name] * assignment[qubo.variables[j].name]
    return value


def decode_bitstring(bitstring, qubo, original_names):
    """Decodifica un bitstring de Aer en una asignación de variables originales.

    Qiskit devuelve bitstrings little-endian: el bit i del circuito (variable i
    del QUBO) es el carácter s[n-1-i]. Se devuelve {nombre_original: 0/1}.
    """
    n = qubo.get_num_vars()
    var_names = [v.name for v in qubo.variables]
    mapping = {}
    for idx, name in enumerate(var_names):
        if name in original_names:
            mapping[name] = int(bitstring[n - 1 - idx])
    return mapping