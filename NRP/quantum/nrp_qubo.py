"""Construcción del modelo QuadraticProgram y su QUBO para NRP.

NRP se modela como una mochila con dependencias lógicas:

    max  sum_f Valor_f * x_f
    s.t. sum_f Costo_f * x_f <= budget              (presupuesto)
         x_origen - x_destino <= 0                  (REQUIRES)
         x_origen + x_destino <= 1                  (EXCLUDES)

Las restricciones se convierten en penalizaciones QUBO con
`QuadraticProgramToQubo` (qiskit-optimization >= 0.7):
  * REQUIRES (x <= y) y EXCLUDES (sum x <= 1) son "formas especiales" que se
    penalizan de forma exacta sin variables de holgura.
  * El presupuesto (desigualdad general) se convierte en una igualdad con
    variables de holgura binarias y se penaliza.

El peso de penalización debe dominar el rango posible del valor para que
ninguna solución infactible compita con una factible.
"""
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo


def build_qp(df_features, df_deps, budget, value_scale=None):
    """Construye el QuadraticProgram binario del NRP.

    Devuelve (qp, var_names).

    `value_scale` normaliza la función objetivo (Valor / value_scale) para
    mantenerla en un rango comparable al de la penalización, evitando paisajes
    de energía dominados por el término de penalización (p. ej. ~1e8). Por
    defecto usa el valor máximo, de modo que los coeficientes quedan en [-1, 0).

    Además, costos y presupuesto se dividen por su máximo común divisor para
    reducir la magnitud del término cuadrático de la penalización del
    presupuesto (el conjunto factible es idéntico).
    """
    import math
    from functools import reduce

    features = df_features.index.tolist()
    var_names = features

    if value_scale is None:
        value_scale = float(df_features["Valor"].max())

    costs = [int(df_features.loc[f, "Costo"]) for f in features]
    g = reduce(math.gcd, costs + [int(budget)])
    if g > 1:
        costs = [c // g for c in costs]
        budget = budget // g

    qp = QuadraticProgram("nrp")
    x = {f: qp.binary_var(name=f) for f in features}

    # Maximizar valor == minimizar su negativo (normalizado)
    qp.minimize(
        linear={f: -float(df_features.loc[f, "Valor"]) / value_scale for f in features}
    )

    qp.linear_constraint(
        linear={f: float(costs[i]) for i, f in enumerate(features)},
        sense="<=",
        rhs=float(budget),
        name="Limite_Presupuesto",
    )

    for _, row in df_deps.iterrows():
        origen, relacion, destino = row["Origen"], row["Relacion"], row["Destino"]
        if origen not in features or destino not in features:
            continue
        if relacion == "REQUIRES":
            qp.linear_constraint(
                linear={origen: 1.0, destino: -1.0},
                sense="<=",
                rhs=0.0,
                name=f"Req_{origen}_needs_{destino}",
            )
        elif relacion == "EXCLUDES":
            qp.linear_constraint(
                linear={origen: 1.0, destino: 1.0},
                sense="<=",
                rhs=1.0,
                name=f"Exc_{origen}_or_{destino}",
            )
    return qp, var_names


def build_qubo(qp, penalty=None):
    """Convierte el QuadraticProgram a QUBO aplicando penalizaciones.

    penalty=None usa la heurística automática de qiskit-optimization.
    """
    return QuadraticProgramToQubo(penalty=penalty).convert(qp)


# Valores CONGELADOS el 2026-11-17 para la ronda confirmatoria: lambda1 = 0.1,
# lambda2 = 0.01 y P = DEFAULT_PENALTY_LOGIC = 2 (más abajo).
# Calibración: scripts/sweep_lambda.py --problem nrp (P = 2), resultados en
# report/fase4/nrp/sweep_lambda*_P2*.csv; 7 instancias de calibración y 12 de
# evaluación. A diferencia de TSM, NINGÚN par hace que el argmin coincida con el
# óptimo del ILP en todas las instancias (mochila: la parábola premia dejar
# presupuesto sin usar).
# Decisión (2026-11-17): se CONSERVAN los lambdas de la exploración, (0.1, 0.01),
# para que la única diferencia entre E3 y la ronda confirmatoria en el NRP sea P.
# Con P = 2 ese par da: argmin óptimo en 6/7 de calibración y 9/12 de evaluación,
# argmin factible en 12/12 de evaluación y óptimo entre los 15 estados de menor
# energía en las 19 instancias (rango <= 3 en calibración y <= 14 en evaluación).
# OJO: (0.1, 0.01) NO es el par que elige el criterio de respaldo documentado
# (menor rango máximo del óptimo). Con P = 2 ese criterio prefiere (0.05, 0.005)
# o (0.075, 0.008), ambos con rango máximo 6; (0.1, 0.01) queda quinto. No se
# adoptaron para no cambiar dos elementos de la codificación a la vez.
# Las métricas de QAOA se calculan sobre los óptimos reales (p_opt) y la
# selección final filtra factibilidad, así que la inexactitud se mide, no se oculta.
# Historia: la calibración original (2026-10-31, report/fase4/nrp/sweep_lambda.csv
# y sweep_lambda_fino.csv) usó P = lambda1 + lambda2 y eligió el mismo par, con
# argmin óptimo en 4/7 y 2/12, argmin factible en 2/12 y rango <= 27. Con ese P se
# corrió el experimento exploratorio E3.
DEFAULT_LAMBDA1 = 0.1
DEFAULT_LAMBDA2 = 0.01

# Penalización de las dependencias (REQUIRES / EXCLUDES), independiente de los
# lambdas del presupuesto. Debe superar el mayor coeficiente del objetivo
# normalizado (<= 1) para que violar una dependencia nunca salga más barato que
# renunciar a una característica.
# El experimento exploratorio E3 usó P = lambda1 + lambda2 = 0.11 (< 1): el
# estado de menor energía violaba dependencias en la mayoría de las instancias
# (deuda técnica #7). Ese comportamiento se conserva solo para reproducir E3,
# con penalty_logic=LEGACY_PENALTY_LOGIC.
DEFAULT_PENALTY_LOGIC = 2.0
LEGACY_PENALTY_LOGIC = "lambda"


def resolve_penalty_logic(lambda1, lambda2, penalty_logic=None):
    """Valor numérico de P: None -> DEFAULT_PENALTY_LOGIC; "lambda" -> l1 + l2 (E3)."""
    if penalty_logic is None:
        return float(DEFAULT_PENALTY_LOGIC)
    if penalty_logic == LEGACY_PENALTY_LOGIC:
        return float(lambda1) + float(lambda2)
    return float(penalty_logic)


def build_qubo_unbalanced(qp, lambda1, lambda2, penalty_logic=None):
    """QUBO de NRP sin variables de holgura.

    * Presupuesto (sum c_i x_i <= B, el caso mochila de Montañez-Barrera et
      al. 2024): h(x) = B - sum c_i x_i >= 0 y
          zeta(x) = -lambda1 * h(x) + lambda2 * h(x)^2
      con h en unidades de costo (ya divididas por el mcd en build_qp). Usando
      x^2 = x: zeta = (-l1 B + l2 B^2) + sum_i (l1 c_i - 2 l2 B c_i + l2 c_i^2) x_i
                      + 2 l2 sum_{i<j} c_i c_j x_i x_j.
      NO es exacta: la parábola premia dejar presupuesto sin usar (mínimo en
      h = l1 / (2 l2)); hay que verificar por fuerza bruta que el argmin siga
      siendo el óptimo del ILP.
    * REQUIRES (x_o - x_d <= 0) y EXCLUDES (x_o + x_d <= 1): penalización exacta
      P * x_o (1 - x_d) y P * x_o x_d, la misma forma que usa
      QuadraticProgramToQubo (sin holguras). P = penalty_logic: por defecto
      DEFAULT_PENALTY_LOGIC (2.0, mayor que cualquier coeficiente del objetivo
      normalizado); "lambda" usa lambda1 + lambda2, como en el experimento E3.

    Devuelve un QuadraticProgram sin restricciones con exactamente las N
    variables de decisión de `qp`, en el mismo orden.
    """
    l1, l2 = float(lambda1), float(lambda2)
    P = resolve_penalty_logic(l1, l2, penalty_logic)
    names = [v.name for v in qp.variables]
    pos = {name: i for i, name in enumerate(names)}

    linear = [0.0] * len(names)
    for i, coeff in qp.objective.linear.to_dict().items():
        linear[i] += coeff
    quadratic = {}
    for (i, j), coeff in qp.objective.quadratic.to_dict().items():
        quadratic[(i, j)] = quadratic.get((i, j), 0.0) + coeff
    constant = float(qp.objective.constant)

    def add_quad(i, j, c):
        key = (min(i, j), max(i, j))
        quadratic[key] = quadratic.get(key, 0.0) + c

    for cons in qp.linear_constraints:
        if cons.sense.name != "LE":
            raise ValueError(f"build_qubo_unbalanced: restriccion {cons.name} con sentido {cons.sense.name}")
        terms = {pos[n]: float(c) for n, c in cons.linear.to_dict(use_name=True).items() if c != 0}
        rhs = float(cons.rhs)
        coeffs = sorted(terms.values())
        if len(terms) == 2 and coeffs == [-1.0, 1.0] and rhs == 0.0:          # REQUIRES
            o = next(i for i, c in terms.items() if c == 1.0)
            d = next(i for i, c in terms.items() if c == -1.0)
            linear[o] += P
            add_quad(o, d, -P)
        elif len(terms) == 2 and coeffs == [1.0, 1.0] and rhs == 1.0:         # EXCLUDES
            o, d = terms.keys()
            add_quad(o, d, P)
        else:                                                                 # presupuesto
            B = rhs
            constant += -l1 * B + l2 * B * B
            items = sorted(terms.items())
            for i, c in items:
                linear[i] += l1 * c - 2.0 * l2 * B * c + l2 * c * c
            for a in range(len(items)):
                for b in range(a + 1, len(items)):
                    (i, ci), (j, cj) = items[a], items[b]
                    add_quad(i, j, 2.0 * l2 * ci * cj)

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
    """Evalúa la función objetivo QUBO con una asignación completa."""
    linear = qubo.objective.linear.to_dict()
    quadratic = qubo.objective.quadratic.to_dict()
    value = qubo.objective.constant
    for i, coeff in linear.items():
        value += coeff * assignment[qubo.variables[i].name]
    for (i, j), coeff in quadratic.items():
        value += coeff * assignment[qubo.variables[i].name] * assignment[qubo.variables[j].name]
    return value


def decode_bitstring(bitstring, qubo, original_names):
    """Decodifica un bitstring en asignación de variables originales (F*)."""
    n = qubo.get_num_vars()
    var_names = [v.name for v in qubo.variables]
    mapping = {}
    for idx, name in enumerate(var_names):
        if name in original_names:
            mapping[name] = int(bitstring[n - 1 - idx])
    return mapping