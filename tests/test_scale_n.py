"""Pruebas de la generacion de instancias de scripts/scale_n.py.

Verifican que las instancias de la ronda confirmatoria (semilla de datos
distinta de 42) no sobrescriben las exploratorias y que la semilla 42 sigue
produciendo exactamente los archivos versionados en report/scale_n/data/.

Ejecutar desde codigo/:  python -m pytest tests -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
REPO_DATA = ROOT / "report" / "scale_n" / "data"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def scale(tmp_path, monkeypatch):
    """scale_n con DATA redirigido a una carpeta temporal (no toca el repo)."""
    mod = _load(ROOT / "scripts" / "scale_n.py", "scale_n_test")
    monkeypatch.setattr(mod, "DATA", tmp_path / "data")
    return mod


def _files(folder):
    return sorted(p.name for p in Path(folder).glob("*.csv"))


def _same_csv(a, b):
    """Mismo contenido, sin depender de los finales de linea."""
    return Path(a).read_bytes().replace(b"\r\n", b"\n") == Path(b).read_bytes().replace(b"\r\n", b"\n")


@pytest.mark.parametrize("family, prefix, n", [
    ("tsm_aleatoria", "tsm", 8),
    ("tsm_ciclo", "tsm", 8),
    ("nrp", "nrp", 8),
])
def test_semilla_42_reproduce_las_instancias_versionadas(scale, family, prefix, n):
    problem, name, _ = scale.make_instance(family, n, 42)
    assert name == f"{family}_n{n:02d}"
    assert scale.instance_dir(42) == scale.DATA
    generated = [p for p in scale.DATA.glob(f"{prefix}_{name}_*.csv")]
    assert len(generated) == 2
    for path in generated:
        assert _same_csv(path, REPO_DATA / path.name), path.name


@pytest.mark.parametrize("family", ["tsm_aleatoria", "tsm_ciclo", "nrp"])
def test_semilla_nueva_no_toca_las_instancias_de_la_42(scale, family):
    scale.make_instance(family, 6, 42)
    before = {p.name: p.read_bytes() for p in scale.DATA.glob("*.csv")}

    _, name, _ = scale.make_instance(family, 6, 45)

    assert name == f"{family}_n06_d45"
    assert scale.instance_dir(45) == scale.DATA / "ds45"
    after = {p.name: p.read_bytes() for p in scale.DATA.glob("*.csv")}
    assert after == before                                  # nada cambio ni se agrego en data/
    assert len(_files(scale.DATA / "ds45")) == 2
    assert all("_d45_" in f for f in _files(scale.DATA / "ds45"))


def test_semillas_de_datos_distintas_no_se_pisan(scale):
    scale.make_instance("tsm_aleatoria", 6, 45)
    first = {p.name: p.read_bytes() for p in (scale.DATA / "ds45").glob("*.csv")}
    scale.make_instance("tsm_aleatoria", 6, 46)
    assert {p.name: p.read_bytes() for p in (scale.DATA / "ds45").glob("*.csv")} == first
    assert len(_files(scale.DATA / "ds46")) == 2


def test_generacion_determinista(scale):
    scale.make_instance("nrp", 8, 45)
    first = {p.name: p.read_bytes() for p in (scale.DATA / "ds45").glob("*.csv")}
    scale.make_instance("nrp", 8, 45)
    assert {p.name: p.read_bytes() for p in (scale.DATA / "ds45").glob("*.csv")} == first


def test_semilla_derivada_del_tamano(scale):
    assert scale.instance_seed(42, 8) == 42
    assert scale.instance_seed(45, 8) == [45, 8]
    assert scale.instance_seed(45, 8) != scale.instance_seed(45, 10)


def _nrp_costs(folder, name):
    return pd.read_csv(Path(folder) / f"nrp_{name}_features.csv")["Costo"].to_numpy()


def test_tamanos_distintos_no_comparten_la_secuencia_aleatoria(scale):
    # Con la semilla 42 (heredada) los costos de N=8 son el prefijo de los de N=10.
    _, n8, _ = scale.make_instance("nrp", 8, 42)
    _, n10, _ = scale.make_instance("nrp", 10, 42)
    assert np.array_equal(_nrp_costs(scale.DATA, n8), _nrp_costs(scale.DATA, n10)[:8])
    # Con una semilla nueva ya no: cada tamano es una instancia independiente.
    _, m8, _ = scale.make_instance("nrp", 8, 45)
    _, m10, _ = scale.make_instance("nrp", 10, 45)
    ds = scale.DATA / "ds45"
    assert not np.array_equal(_nrp_costs(ds, m8), _nrp_costs(ds, m10)[:8])


@pytest.mark.parametrize("seed", [42, 45])
def test_presupuesto_nrp_coincide_con_los_costos_generados(scale, seed):
    _, name, budget = scale.make_instance("nrp", 10, seed)
    costs = _nrp_costs(scale.instance_dir(seed), name)
    assert budget == int(round(0.4 * costs.sum()))
