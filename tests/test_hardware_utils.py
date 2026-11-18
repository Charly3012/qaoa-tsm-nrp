"""Pruebas de las utilidades de hardware que no requieren IBM Quantum.

* _job_quantum_seconds reintenta hasta que IBM llena el uso del job (deuda
  tecnica #8: se leia 0.0 justo despues de job.result()).
* hardware_replicas distingue la misma instancia con dos configuraciones.

Ejecutar desde codigo/:  python -m pytest tests -q
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeJob:
    """Job cuyo uso aparece como pendiente las primeras `delay` consultas."""

    def __init__(self, delay, seconds=3.0, status_when_ready="complete"):
        self.calls = 0
        self.delay = delay
        self.seconds = seconds
        self.status = status_when_ready

    def metrics(self):
        self.calls += 1
        if self.calls <= self.delay:
            return {"usage": {"quantum_seconds": 0, "status": "pending"}}
        return {"usage": {"quantum_seconds": self.seconds, "status": self.status}}

    def usage(self):
        return 0.0


@pytest.fixture(scope="module")
def hw():
    return _load(ROOT / "scripts" / "hardware_replicas.py", "hardware_replicas_t")


@pytest.fixture(params=["tsm", "nrp"])
def quantum_seconds(request, hw):
    return hw.activate(request.param)._job_quantum_seconds


def test_activate_carga_los_modulos_del_problema_correcto(hw):
    """TSM y NRP comparten nombres de modulo; tras activar NRP, data_io debe ser el de NRP."""
    hw.activate("tsm")
    assert hasattr(sys.modules["data_io"], "load_tsm")
    nrp = hw.activate("nrp")
    assert hasattr(sys.modules["data_io"], "load_nrp")
    assert nrp.load_nrp is sys.modules["data_io"].load_nrp
    hw.activate("tsm")
    assert hasattr(sys.modules["data_io"], "load_tsm")


def test_espera_a_que_ibm_llene_el_uso(quantum_seconds):
    waits = []
    job = FakeJob(delay=3, seconds=3.0)
    assert quantum_seconds(job, retries=10, wait=5.0, sleep=waits.append) == 3.0
    assert job.calls == 4 and waits == [5.0, 5.0, 5.0]


def test_devuelve_lo_ultimo_si_se_agotan_los_reintentos(quantum_seconds):
    job = FakeJob(delay=100)
    assert quantum_seconds(job, retries=3, wait=0.0, sleep=lambda s: None) == 0.0
    assert job.calls == 3


def test_disponible_de_inmediato_no_espera(quantum_seconds):
    waits = []
    assert quantum_seconds(FakeJob(delay=0, seconds=2.5), sleep=waits.append) == 2.5
    assert waits == []


def test_job_key_separa_configuraciones(hw):
    base = dict(problem="tsm", instance="toy2", label="Base")
    fci = dict(problem="tsm", instance="toy2", label="F+C+I")
    assert hw.job_key(base) != hw.job_key(fci)
    assert len(hw.BLOCK_A) == 7 and len(hw.BLOCK_B) == 6
    assert all(j[0] == "A" and j[5] == "unbalanced" for j in hw.BLOCK_A)
    assert all(j[0] == "B" and j[5] == "slack" for j in hw.BLOCK_B)


def test_referencia_no_construye_tablas_si_no_cabe(hw, monkeypatch):
    """Deuda tecnica #10: sobre MAX_SIM_QUBITS no se arman las tablas de 2^N estados."""
    job = hw.default_jobs(["B"])[0]                      # tsm toy, con holguras
    monkeypatch.setattr(hw, "MAX_SIM_QUBITS", 1)
    real = hw.build_job_qubo
    called = []

    def spy(job, M):
        qubo, names, tables = real(job, M)
        return qubo, names, lambda: called.append(1) or tables()

    monkeypatch.setattr(hw, "build_job_qubo", spy)
    p_opt, p_feas, n_q = hw.sim_reference(job, [0.1, 0.2])
    assert p_opt is None and p_feas is None and n_q > 1 and called == []


def test_referencia_coincide_con_la_fase4(hw):
    """La referencia exacta de tsm_ciclo_n16 debe ser la registrada en hardware_fase4.csv."""
    import pandas as pd
    job = next(j for j in hw.default_jobs(["A"]) if j["instance"] == "tsm_ciclo_n16")
    hw.activate("tsm")
    params, _ = hw.resolve_params(job, hw.SCALE.load_problem("tsm")["engine"])
    p_opt, p_feas, n_q = hw.sim_reference(job, params)
    ref = pd.read_csv(ROOT / "report" / "fase4" / "hardware_fase4.csv").set_index("instance").loc["tsm_ciclo_n16"]
    assert n_q == 16
    assert p_opt == pytest.approx(ref.sim_p_opt, rel=1e-6)
    assert p_feas == pytest.approx(ref.sim_p_feasible, rel=1e-6)
