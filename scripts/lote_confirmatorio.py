#!/usr/bin/env python3
"""Lote de la ronda confirmatoria en simulador (cap. 5 de la tesis).

Recorre semilla de datos x familia x configuracion y llama a scale_n.py una
vez por combinacion. Cada llamada recorre la cadena completa de tamanos con
todas las semillas del optimizador (lo exige la inicializacion por
transferencia).

  familia         tamanos        configuraciones
  tsm_aleatoria   4..22          F, FC (alpha 0.25), FCI (transferencia)
  tsm_aleatoria   4, 6, 8        base (con holguras; contraste de H7)
  nrp             4..20          F, FC, FCI
  tsm_ciclo       4..22          FC, FCI   (control, secundario)

Bloque adicional de H7 (semillas de datos 50-59): tsm_aleatoria en N = 8, solo
F y base. Con las 5 instancias de N = 8 del bloque principal suman las 15
instancias de H7 (el contraste se prueba donde las holguras pesan: 24 qubits
frente a 8).

La linea base aleatorio + filtro se calcula una sola vez por instancia: en la
primera configuracion de cada familia; las demas usan --no-random.

Salida: <out>/ds<semilla>/scale_<familia>_<config>.csv, un log por corrida en
<out>/logs/ y <out>/lote_resumen.csv con duracion y estado. Una corrida que ya
termino bien (archivo .ok) se salta, asi que el lote se puede reanudar; si una
falla, se registra y se sigue con la siguiente.

Uso (desde codigo/):
    python scripts/lote_confirmatorio.py --dry-run                       # solo lista el plan
    python scripts/lote_confirmatorio.py --data-seeds 45 --only tsm_aleatoria:FCI   # piloto
    python scripts/lote_confirmatorio.py --only h7                       # solo el bloque de H7
    python scripts/lote_confirmatorio.py                                 # lote completo
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_SEEDS = "45,46,47,48,49"
OPT_SEEDS = "100,101,102,103,104,105,106,107,108,109"
ALPHA = "0.25"

TSM_SIZES = "4,6,8,10,12,14,16,18,20,22"
NRP_SIZES = "4,6,8,10,12,14,16,18,20"
BASE_SIZES = "4,6,8"
H7_SEEDS = "50,51,52,53,54,55,56,57,58,59"
H7_SIZE = "8"

# (familia, etiqueta, tamanos, argumentos de configuracion, calcula aleatorio + filtro)
PLAN = [
    ("tsm_aleatoria", "F", TSM_SIZES, ["--config", "F"], True),
    ("tsm_aleatoria", "FC", TSM_SIZES, ["--config", "FC", "--alpha", ALPHA], False),
    ("tsm_aleatoria", "FCI", TSM_SIZES, ["--config", "FCI", "--init", "transfer", "--alpha", ALPHA], False),
    ("tsm_aleatoria", "base", BASE_SIZES, ["--config", "base"], False),
    ("nrp", "F", NRP_SIZES, ["--config", "F"], True),
    ("nrp", "FC", NRP_SIZES, ["--config", "FC", "--alpha", ALPHA], False),
    ("nrp", "FCI", NRP_SIZES, ["--config", "FCI", "--init", "transfer", "--alpha", ALPHA], False),
    ("tsm_ciclo", "FC", TSM_SIZES, ["--config", "FC", "--alpha", ALPHA], True),
    ("tsm_ciclo", "FCI", TSM_SIZES, ["--config", "FCI", "--init", "transfer", "--alpha", ALPHA], False),
]

# Bloque adicional de H7: mismas configuraciones F y base, un solo tamano, semillas de datos propias.
PLAN_H7 = [
    ("tsm_aleatoria", "F", H7_SIZE, ["--config", "F"], False),
    ("tsm_aleatoria", "base", H7_SIZE, ["--config", "base"], False),
]


def build_jobs(data_seeds, only, opt_seeds, sizes_override, h7_seeds=()):
    jobs = []
    blocks = [(ds, PLAN, "") for ds in data_seeds] + [(ds, PLAN_H7, "_h7") for ds in h7_seeds]
    for ds, plan, suffix in blocks:
        for family, label, sizes, cfg, with_random in plan:
            if only and f"{family}:{label}" not in only and family not in only \
                    and not (suffix and "h7" in only):
                continue
            if only and not suffix and only == {"h7"}:
                continue
            sizes = sizes_override or sizes
            args = ["--family", family, "--sizes", sizes, *cfg, "--seeds", opt_seeds,
                    "--data-seed", str(ds)]
            if not with_random:
                args.append("--no-random")
            jobs.append(dict(data_seed=ds, family=family, config=label, sizes=sizes, args=args,
                             tag=f"ds{ds}_{family}_{label}{suffix}"))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-seeds", default=DATA_SEEDS)
    ap.add_argument("--seeds", default=OPT_SEEDS, help="semillas del optimizador")
    ap.add_argument("--h7-seeds", default=H7_SEEDS,
                    help="semillas de datos del bloque adicional de H7 (vacio para omitirlo)")
    ap.add_argument("--only", default=None,
                    help="filtro separado por comas: familia, familia:config o h7 "
                         "(p. ej. tsm_aleatoria:FCI,nrp; 'h7' = solo el bloque adicional)")
    ap.add_argument("--sizes", default=None, help="sustituye los tamanos de todas las corridas (solo pruebas)")
    ap.add_argument("--out", default="report/confirmatorio", help="carpeta de salida")
    ap.add_argument("--dry-run", action="store_true", help="lista el plan sin ejecutar")
    ap.add_argument("--force", action="store_true", help="repite tambien las corridas ya terminadas")
    args = ap.parse_args()

    data_seeds = [int(s) for s in args.data_seeds.split(",")]
    only = set(args.only.split(",")) if args.only else None
    out_root = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    h7_seeds = [int(s) for s in args.h7_seeds.split(",") if s]
    jobs = build_jobs(data_seeds, only, args.seeds, args.sizes, h7_seeds)

    print(f"{len(jobs)} corridas; salida en {out_root}")
    for j in jobs:
        done = (out_root / "logs" / f"{j['tag']}.ok").exists()
        print(f"  {'[hecha] ' if done and not args.force else '        '}{j['tag']:32s} "
              f"N={j['sizes']}  {' '.join(j['args'][4:])}")
    if args.dry_run or not jobs:
        return

    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "lote_resumen.csv"
    new_file = not summary_path.exists()
    failed = 0
    with open(summary_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["inicio", "tag", "data_seed", "family", "config", "sizes", "estado", "segundos", "avisos"])
        for k, j in enumerate(jobs, 1):
            ok_marker = logs / f"{j['tag']}.ok"
            if ok_marker.exists() and not args.force:
                continue
            ok_marker.unlink(missing_ok=True)
            out_dir = out_root / f"ds{j['data_seed']}"
            cmd = [sys.executable, "-u", str(ROOT / "scripts" / "scale_n.py"), *j["args"],
                   "--out-dir", str(out_dir)]
            start = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{k}/{len(jobs)}] {start} {j['tag']} ...", flush=True)
            t0 = time.perf_counter()
            log_path = logs / f"{j['tag']}.log"
            with open(log_path, "w", encoding="utf-8", errors="replace") as log:
                log.write("# " + " ".join(cmd) + "\n")
                log.flush()
                proc = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT)
            secs = time.perf_counter() - t0
            text = log_path.read_text(encoding="utf-8", errors="replace")
            avisos = sum(1 for ln in text.splitlines() if ln.startswith("AVISO"))
            state = "ok" if proc.returncode == 0 else f"error({proc.returncode})"
            if proc.returncode == 0:
                ok_marker.write_text(start + "\n")
            else:
                failed += 1
            w.writerow([start, j["tag"], j["data_seed"], j["family"], j["config"], j["sizes"], state,
                        f"{secs:.1f}", avisos])
            fh.flush()
            print(f"        {state} en {secs / 60:.1f} min" + (f", {avisos} AVISO(s)" if avisos else ""),
                  flush=True)
    print(f"\nLote terminado: {failed} corridas con error. Resumen en {summary_path}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
