"""Motor QAOA reutilizable para simulador y hardware.

Construye el ansatz QAOA (capa de evolución del Hamiltoniano de coste con
PauliEvolutionGate + mezclador RX), evalúa la energía esperada sobre el QUBO
mediante muestreo de counts y extrae la mejor solución factible.

El backend se abstrae con un objeto que implementa:

    sample(circuit: QuantumCircuit, shots: int) -> dict[str, int]

(su método devuelve counts con el mismo formato que AerSimulator).
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate


def build_ansatz(operator, reps):
    """Construye el circuito QAOA para el operador Ising (diagonal).

    Parámetros en el orden [gamma_0, beta_0, gamma_1, beta_1, ...].
    No incluye medición: esta se añade solo al muestrear (extract_best).
    """
    n = operator.num_qubits
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(reps):
        qc.append(PauliEvolutionGate(operator, time=Parameter(f"gamma_{layer}")), range(n))
        qc.rx(2.0 * Parameter(f"beta_{layer}"), range(n))
    return qc


def initial_params(reps, rng, init="random", warm=None, objective_fn=None, grid=(36, 18)):
    """Punto inicial QAOA [gamma_0, beta_0, gamma_1, beta_1, ...].

    init="random"  : gamma ~ U[0, 2pi), beta ~ U[0, pi) (configuración
                     Base). Evita iniciar en un entorno minúsculo que
                     atrapa a COBYLA en un mínimo local.
    init="grid"    : mejor punto de una malla gamma x beta para p = 1 según
                     `objective_fn`; si reps > 1 se
                     extiende con INTERP.
    init="transfer": parte de `warm` (parámetros optimizados de otra instancia
                     o de menos capas) y lo lleva a `reps` capas con INTERP.
    """
    if init == "random":
        params = []
        for _ in range(reps):
            params.append(float(rng.uniform(0.0, 2.0 * np.pi)))
            params.append(float(rng.uniform(0.0, np.pi)))
        return np.asarray(params)
    if init == "grid":
        if objective_fn is None:
            raise ValueError("init='grid' requiere objective_fn")
        params, _ = grid_search_params(objective_fn, *grid)
    elif init == "transfer":
        if warm is None:
            raise ValueError("init='transfer' requiere warm")
        params = np.asarray(warm, dtype=np.float64)
    else:
        raise ValueError(f"init desconocido: {init}")
    while len(params) // 2 < reps:
        params = interp_params(params)
    if len(params) // 2 > reps:
        raise ValueError(f"warm tiene {len(params) // 2} capas y se piden {reps}")
    return params


def grid_search_params(objective_fn, n_gamma=36, n_beta=18):
    """Mejor (gamma, beta) de una malla regular para p = 1.

    Devuelve (params, valor). gamma en [0, 2pi), beta en [0, pi), sin el
    extremo (dominio periódico).
    """
    best_val, best = np.inf, None
    for g in np.linspace(0.0, 2.0 * np.pi, n_gamma, endpoint=False):
        for b in np.linspace(0.0, np.pi, n_beta, endpoint=False):
            p = np.array([g, b])
            v = float(objective_fn(p))
            if v < best_val:
                best_val, best = v, p
    return best, best_val


def interp_params(params):
    """INTERP (Zhou et al. 2020, ap. B): de p capas optimizadas a p+1.

    Para cada vector v (gammas o betas) de longitud p, el nuevo vector de
    longitud p+1 es, con v[0] = v[p+1] = 0 (índices 1..p):
        v'[i] = (i-1)/p * v[i-1] + (p-i+1)/p * v[i],   i = 1..p+1
    """
    params = np.asarray(params, dtype=np.float64)
    p = len(params) // 2
    gammas, betas = params[0::2], params[1::2]

    def _interp(v):
        out = np.zeros(p + 1)
        for i in range(1, p + 2):
            prev = v[i - 2] if i >= 2 else 0.0        # v[i-1] (1-indexado)
            cur = v[i - 1] if i <= p else 0.0         # v[i]
            out[i - 1] = (i - 1) / p * prev + (p - i + 1) / p * cur
        return out

    new = np.empty(2 * (p + 1))
    new[0::2], new[1::2] = _interp(gammas), _interp(betas)
    return new


def bind_params(circuit, params):
    """Asigna params=[gamma_0, beta_0, gamma_1, beta_1, ...] por NOMBRE.

    `circuit.parameters` está ordenado alfabéticamente (beta_* antes que
    gamma_*), por lo que un zip posicional cruzaba gamma y beta. Se asigna
    explícitamente cada valor al parámetro que le corresponde.
    """
    by_name = {p.name: p for p in circuit.parameters}
    reps = len(params) // 2
    mapping = {}
    for layer in range(reps):
        mapping[by_name[f"gamma_{layer}"]] = float(params[2 * layer])
        mapping[by_name[f"beta_{layer}"]] = float(params[2 * layer + 1])
    return circuit.assign_parameters(mapping)


def prepare_circuit(backend, ansatz):
    """Transpila el ansatz (estructura independiente de los parámetros).

    Se transpila una sola vez para no repetir el coste en cada evaluación.
    Se usan basis gates explícitos (sin coupling map) para no limitar el
    número de qubits simulables.
    """
    return transpile(
        ansatz,
        basis_gates=["rx", "ry", "rz", "h", "cx", "x", "sx", "measure", "id"],
        optimization_level=0,
    )


def build_value_lookup(qubo):
    """Precalcula el valor QUBO de cada bitstring (2**n) con numpy.

    values[i] = valor QUBO del estado cuyo índice (little-endian) es i.
    Los bits se extraen al vuelo por término (sin materializar la matriz
    2**n x n), así la memoria es ~2 vectores de 2**n en vez de ~n+2.
    Solo viable para pocas decenas de qubits (memoria ~ 2**n).
    """
    n = qubo.get_num_vars()
    idx = np.arange(1 << n, dtype=np.uint32 if n <= 32 else np.uint64)
    values = np.full(idx.size, qubo.objective.constant, dtype=np.float64)
    for i, coeff in qubo.objective.linear.to_dict().items():
        values += coeff * ((idx >> i) & 1)
    for (i, j), coeff in qubo.objective.quadratic.to_dict().items():
        values += coeff * ((idx >> i) & (idx >> j) & 1)
    return values


def cvar_from_sorted(sorted_values, sorted_probs, alpha):
    """CVaR_alpha de una distribución discreta ya ordenada por valor creciente.

    Barkoutsos et al. (2020): media de los valores en la fracción alpha de
    menor energía (la cola izquierda). El último estado que entra se recorta
    para que la masa sume exactamente alpha. alpha = 1 devuelve la media.
    """
    if alpha >= 1.0:
        return float(np.dot(sorted_probs, sorted_values))
    cum = np.cumsum(sorted_probs)
    k = min(int(np.searchsorted(cum, alpha, side="left")), cum.size - 1)
    w = np.array(sorted_probs[:k + 1], dtype=np.float64)
    w[-1] = max(w[-1] - (cum[k] - alpha), 0.0)
    return float(np.dot(w, sorted_values[:k + 1]) / w.sum())


def cvar_from_probs(probs, values, alpha, order=None):
    """CVaR_alpha a partir del vector de probabilidades del statevector.

    `order` = np.argsort(values) precalculado (una sola vez por QUBO) para
    no reordenar los 2**n valores en cada evaluación.
    """
    if alpha >= 1.0:
        return float(np.dot(probs, values))
    if order is None:
        order = np.argsort(values, kind="stable")
    return cvar_from_sorted(values[order], probs[order], alpha)


def cvar_from_counts(counts, qubo, qubo_value_fn, alpha):
    """CVaR_alpha estimada por muestreo (simulador con shots o hardware)."""
    n_shots = sum(counts.values())
    pairs = sorted(
        (qubo_value_fn(decode_full(b, qubo), qubo), c / n_shots) for b, c in counts.items()
    )
    vals = np.array([p[0] for p in pairs])
    probs = np.array([p[1] for p in pairs])
    return cvar_from_sorted(vals, probs, alpha)


def evaluate_energy(backend, circuit, params, qubo, qubo_value_fn, shots,
                    operator=None, offset=None, exact=False, value_lookup=None,
                    objective="energy", alpha=0.25, value_order=None):
    """Objetivo variacional del QUBO para unos parámetros.

    objective="energy": energía esperada (configuración Base).
    objective="cvar"  : CVaR_alpha (media de la fracción alpha de muestras de
                        menor energía); con exact=True requiere value_lookup.
    Si exact=True usa el statevector (AerSimulator); en caso contrario estima
    por muestreo de counts.
    """
    if objective not in ("energy", "cvar"):
        raise ValueError(f"objective desconocido: {objective}")
    bound = bind_params(circuit, params)
    if exact:
        probe = bound.copy()
        probe.save_statevector()
        sv = backend.backend.run(probe).result().get_statevector()
        if value_lookup is not None:
            probs = np.abs(np.asarray(sv.data)) ** 2
            if objective == "cvar":
                return cvar_from_probs(probs, value_lookup, alpha, order=value_order)
            return float(np.dot(probs, value_lookup))
        if objective == "cvar":
            raise ValueError("CVaR exacta requiere value_lookup")
        return float(np.real(offset + sv.expectation_value(operator)))
    sampled = bound.copy()
    sampled.measure_all()
    counts = backend.sample(sampled, shots)
    if objective == "cvar":
        return cvar_from_counts(counts, qubo, qubo_value_fn, alpha)
    n_shots = sum(counts.values())
    energy = 0.0
    for bitstring, count in counts.items():
        assignment = decode_full(bitstring, qubo)
        energy += (count / n_shots) * qubo_value_fn(assignment, qubo)
    return energy


def decode_full(bitstring, qubo):
    """Asignación completa {variable: 0/1} sobre todas las variables del QUBO."""
    n = qubo.get_num_vars()
    var_names = [v.name for v in qubo.variables]
    return {
        name: int(bitstring[n - 1 - idx]) for idx, name in enumerate(var_names)
    }


def extract_best(backend, circuit, params, qubo, original_names, qubo_value_fn,
                 validate, shots, objective=None):
    """Muestrea el circuito optimizado y devuelve la mejor solución.

    Devuelve (asignacion_original, qubo_val, factible, counts).
    Selecciona la de menor valor QUBO entre las factibles; si no hay ninguna,
    devuelve la de menor valor QUBO global (y factible=False).
    """
    measured = circuit.copy()
    measured.measure_all()
    bound = bind_params(measured, params)
    counts = backend.sample(bound, shots)

    # `validate` y `objective` reciben la LISTA de variables seleccionadas
    # (antes se pasaba el dict {nombre: 0/1}; is_feasible iteraba sus claves
    # y daba por seleccionadas todas las variables).
    candidates = []
    for bitstring, count in counts.items():
        full = decode_full(bitstring, qubo)
        orig = {name: full[name] for name in original_names}
        sel = [name for name in original_names if orig[name] == 1]
        qv = qubo_value_fn(full, qubo)
        candidates.append((qv, orig, count, sel))

    feasible = [c for c in candidates if validate(c[3])]
    if feasible:
        # Entre las factibles se ordena por el objetivo ORIGINAL (si se da),
        # no por el valor QUBO: ese incluye la penalización de las variables
        # de holgura, y una solución óptima muestreada con holguras "mal
        # puestas" tiene valor QUBO alto y se descartaba frente a otra peor.
        if objective is not None:
            best = min(feasible, key=lambda c: (objective(c[3]), c[0], -c[2]))
        else:
            best = min(feasible, key=lambda c: (c[0], -c[2]))
        return best[1], best[0], True, counts
    best = min(candidates, key=lambda c: (c[0], -c[2]))
    return best[1], best[0], False, counts


class AerBackend:
    """Adaptador de AerSimulator a la interfaz sample() -> counts."""

    def __init__(self, method="statevector", seed=None):
        from qiskit_aer import AerSimulator

        self.seed = seed
        self.backend = AerSimulator(method=method, seed_simulator=seed)

    def sample(self, circuit, shots):
        return self.backend.run(circuit, shots=shots).result().get_counts()