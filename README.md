# qaoa-tsm-nrp

Código, instancias y resultados de la tesis de licenciatura **«Análisis de algoritmos cuánticos
aplicado a problemas de optimización en la ingeniería de software»** (Carlos Aldair Coronado Silva,
Licenciatura en Ingeniería de Software, Facultad de Matemáticas, Universidad Autónoma de Yucatán,
2026).

La tesis analiza los límites de viabilidad del Algoritmo Cuántico de Optimización Aproximada (QAOA)
en la era de los procesadores cuánticos ruidosos (NISQ), aplicado a dos problemas de ingeniería de
*software*:

- **TSM** (*Test Suite Minimization*): elegir el menor número de casos de prueba que cubran todos
  los requisitos (cobertura de conjuntos).
- **NRP** (*Next Release Problem*): elegir los requerimientos de mayor valor que quepan en el
  presupuesto y respeten sus dependencias (mochila con dependencias).

Ambos problemas se formulan como modelos QUBO, con y sin variables de holgura, y QAOA se ejecuta con
Qiskit en simulador y en procesadores de IBM Quantum. Se compara con programación lineal entera,
un algoritmo voraz, recocido simulado y muestreo aleatorio uniforme.

El Apéndice A de la tesis reproduce el código del algoritmo cuántico y el Apéndice B, el de los
métodos clásicos.

## Estructura

```text
.
├── common/qaoa_engine.py      Motor de QAOA compartido (circuito, energía o CVaR, COBYLA, inicializaciones)
├── TSM/                       Test Suite Minimization
│   ├── data/                  Instancias de referencia (toy, toy2, demo)
│   ├── data_io.py             Carga de instancias, métricas y validación de factibilidad
│   ├── classic/               ILP (PuLP/CBC), voraz y recocido simulado
│   ├── quantum/               Modelo QUBO y ejecución de QAOA en simulador y en IBM Quantum
│   ├── benchmark_tsm.py       Ejecuta varios métodos sobre varias instancias
│   └── results/               Resultados de los experimentos E1 y E3 por método e instancia
├── NRP/                       Next Release Problem (misma organización que TSM/)
├── scripts/                   Generación de instancias, barridos, lotes, análisis y figuras
├── tests/                     Pruebas automatizadas (pytest)
└── report/                    Instancias sintéticas, resultados, análisis estadístico y figuras
    ├── scale_n/               E2: escalamiento con la codificación con holguras
    ├── fase4/                 E3: codificación sin holguras, CVaR e inicialización; calibración de λ
    ├── confirmatorio/         E4: lote del simulador y réplicas en hardware
    ├── estadistica/           Pruebas de hipótesis (exploratorio y confirmatorio)
    ├── audit/                 Verificación del QUBO y de QAOA (E1)
    └── figures/               Figuras generadas por los scripts
```

### Experimentos

| Tesis | Nombre en el código | Qué hace | Resultados |
|---|---|---|---|
| E1. Verificación y referencia | Fase 1 | Instancias de referencia con la codificación con holguras, en simulador y en `ibm_fez` | `TSM/results/`, `NRP/results/`, `report/audit/` |
| E2. Escalamiento | Fase 2 | Familias sintéticas de tamaño creciente, codificación con holguras | `report/scale_n/` |
| E3. Codificación y optimización | Fase 4 | Codificación sin holguras, CVaR e inicialización por malla y por transferencia; 7 trabajos en `ibm_fez` | `report/fase4/`, `TSM/results/qaoa_*_fase4/` |
| E4. Confirmación | Ronda confirmatoria | Instancias nuevas (semillas de datos 45–59), 10 semillas del optimizador, configuraciones Base, F, F+C y F+C+I; réplicas en `ibm_marrakesh` | `report/confirmatorio/`, `report/estadistica/confirmatorio/` |

Los comentarios del código conservan la numeración interna por fases y algunas notas del proceso de
desarrollo (fechas y referencias a una lista interna de deuda técnica que no forma parte de este
repositorio).

### Instancias de referencia

La tesis usa nombres distintos a los del código para las instancias de referencia:

| Código | Tesis | Tamaño |
|---|---|---|
| TSM `toy` | T5 | 5 casos de prueba, 6 requisitos |
| TSM `toy2` | T10 | 10 casos de prueba, 12 requisitos |
| TSM `demo` | T12 | 12 casos de prueba, 12 requisitos |
| NRP `toy` | R10 | 10 requerimientos, presupuesto 100 |
| NRP `demo` | R12 | 12 requerimientos, presupuesto 60 |
| NRP `toy2` | R30 | 30 requerimientos, presupuesto 500 |

Las familias sintéticas (`tsm_aleatoria`, `tsm_ciclo`, `nrp`) se generan con
`scripts/generate_dataset.py` a partir de una semilla de datos y del tamaño N; las de la ronda
confirmatoria están en `report/scale_n/data/ds45` a `ds59`.

## Instalación

Se usó Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Versiones con las que se obtuvieron los resultados de la tesis: Qiskit 1.4.6, qiskit-aer 0.17.2,
qiskit-optimization 0.7.0, qiskit-ibm-runtime 0.39.0, SciPy 1.18.1, PuLP 3.3.2, NumPy 2.5.3 y
pandas 2.3.3.

## Uso

Todos los comandos se ejecutan desde la raíz del repositorio, salvo donde se indica.

### Métodos clásicos y QAOA sobre una instancia

```bash
cd TSM
python classic/ilp_tsm.py toy
python classic/greedy_tsm.py toy2
python classic/sa_tsm.py toy --seed 42
python quantum/qaoa_tsm_simulator.py toy --reps 1 --shots 4000 --maxiter 100

cd ../NRP
python classic/ilp_nrp.py toy --budget 100
python quantum/qaoa_nrp_simulator.py toy --budget 100 --reps 1 --maxiter 100
```

Cada script acepta `--help` con todas sus opciones; la equivalencia entre las opciones y los
términos de la tesis (codificación, función de costo, inicialización) está en la Tabla A.1 de la
tesis. La simulación por vector de estado se limita a unos 26 qubits.

### Escalamiento y ronda confirmatoria

```bash
# Una familia, varios tamaños y semillas (E2/E3)
python scripts/scale_n.py --family tsm_ciclo --sizes 4,6,8 --seeds 42,43,44

# Lote completo del simulador de la ronda confirmatoria (E4; varias horas de CPU)
python scripts/lote_confirmatorio.py --dry-run     # solo muestra el plan
python scripts/lote_confirmatorio.py

# Pruebas de hipótesis y figuras
python scripts/analisis_estadistico.py --modo confirmatorio --out report/estadistica/confirmatorio
python scripts/make_confirmatorio_figures.py
```

### Hardware de IBM Quantum

Las credenciales se leen de un archivo `.env` en la raíz del repositorio, que no se versiona.
Copie la plantilla y escriba su clave de API (se obtiene en <https://quantum.cloud.ibm.com>):

```bash
cp .env.example .env
```

```bash
cd TSM
python quantum/qaoa_tsm_ibm.py toy --shots 4000

# Réplicas de la ronda confirmatoria; --dry-run calcula las referencias sin enviar trabajos
python scripts/hardware_replicas.py --blocks A,B --dry-run
```

Ejecutar en *hardware* consume tiempo de QPU de la cuenta de IBM Quantum.

## Pruebas

```bash
python -m pytest
```

## Formulación QUBO

- **Con holguras:** las restricciones de desigualdad (cobertura en el TSM, presupuesto en el NRP) se
  convierten en igualdades con variables de holgura binarias y se penalizan
  (`QuadraticProgramToQubo`).
- **Sin holguras:** la desigualdad se penaliza con una penalización desbalanceada de coeficientes
  λ₁ y λ₂, calibrados por fuerza bruta (`scripts/sweep_lambda.py`), sin agregar qubits. En el NRP,
  las dependencias se penalizan con un peso P independiente (P = 2 en la ronda confirmatoria).

En ambos casos, la factibilidad de la solución final se valida contra el modelo original
(cobertura, presupuesto y dependencias), no contra la energía del QUBO.

## Licencia

Distribuido bajo la licencia MIT. Consulte el archivo [LICENSE](LICENSE).
