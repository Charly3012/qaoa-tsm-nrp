"""Motor QAOA de TSM: alias del motor compartido `codigo/common/qaoa_engine.py`.

Desde 2026-10-30 (Fase 4, deuda tecnica #1) el motor vive en un solo lugar;
este archivo se conserva para que los runners, scripts y pruebas que lo cargan
por ruta (TSM/quantum/qaoa_engine.py) sigan funcionando sin cambios.
"""
import sys
from pathlib import Path

_CODIGO = str(Path(__file__).resolve().parents[2])
if _CODIGO not in sys.path:
    sys.path.insert(0, _CODIGO)

from common.qaoa_engine import *  # noqa: E402,F401,F403
from common.qaoa_engine import (  # noqa: E402,F401  (nombres explicitos para los IDE)
    AerBackend, bind_params, build_ansatz, build_value_lookup, cvar_from_counts,
    cvar_from_probs, cvar_from_sorted, decode_full, evaluate_energy, extract_best,
    grid_search_params, initial_params, interp_params, prepare_circuit,
)
