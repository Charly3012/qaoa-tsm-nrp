# Analisis estadistico - PRELIMINAR (exploratorio): no decide las hipotesis

Entradas: 9 CSV; 33 instancias; semillas de datos [42].

Sin problemas de integridad.

## Hipotesis (familia de Holm)

| hipotesis | familia | n | prueba | estadistico | p | p_holm | efecto | decision |
|---|---|---|---|---|---|---|---|---|
| H1 | tsm_aleatoria | 8 | Wilcoxon 1 muestra (>0) | 35 | 0.007812 | 0.04688 | mediana A = 38.7 [IC95 11.5, 91.9]; signos 7/8 (p = 0.0352) | Rechaza H0 |
| H2 | tsm_aleatoria | 8 | Wilcoxon pareado (F+C vs F) | 30 | 0.05469 | 0.2188 | FS = 0.88; mediana de la razon de A = 2.29 | No rechaza H0 |
| H3 | tsm_aleatoria | 8 | Wilcoxon pareado (RIC F+C+I < RIC F+C) | 4 | 0.02734 | 0.1367 | FS = 0.88; mediana de RIC: 0.00176 (F+C+I) frente a 0.453 (F+C) | No rechaza H0 |
| H4 | tsm_aleatoria | 8 | McNemar exacto | 4 | 0.0625 | 0.2188 | b = 4 (QAOA si, azar no), c = 0; QAOA 8/8, azar 4/8 | No rechaza H0 |
| H6 | nrp | 7 | Wilcoxon 1 muestra (>0) | 14 | 0.5312 | 1 | mediana A = 1.07 [IC95 0.462, 1.48]; signos 4/7 (p = 0.5) | No rechaza H0 |
| H7 | tsm_aleatoria | 1 | Wilcoxon pareado (F vs Base) | 1 | 0.5 | 1 | FS = 1.00; mediana de la razon de A = 696 | No rechaza H0 |

## Analisis secundarios (sin ajuste)

| analisis | n | prueba | estadistico | p | efecto |
|---|---|---|---|---|---|
| TSM aleatoria vs NRP (log A, F+C+I, N 8-20) | 7 / 7 | Mann-Whitney U (TSM > NRP) | 45 | 0.003497 | A12 = 0.92 |
| Exito frente al azar en nrp (F+C+I) | 7 | McNemar exacto | 0 | 1 | b = 0 (QAOA si, azar no), c = 0; QAOA 3/7, azar 3/7 |
| Exito frente al azar en tsm_ciclo (F+C+I) | 8 | McNemar exacto | 4 | 0.0625 | b = 4 (QAOA si, azar no), c = 0; QAOA 8/8, azar 4/8 |
| Sin holguras vs con holguras en N = 4 y 6 (TSM aleatoria) | 2 | Wilcoxon pareado (F vs Base) | 1 | 0.75 | FS = 0.50; mediana de la razon de A = 0.948 |
| Nivel de A: F+C+I vs F+C (TSM aleatoria) | 8 | Wilcoxon pareado | 20 | 0.4219 | FS = 0.38; mediana de la razon de A = 0.996 |

## H5: hardware frente a simulador (por instancia)

| instancia | medida | replicas | mediciones | k | p_hw | wilson_inf | wilson_sup | p_sim | esperados | cociente | p | conclusion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| toy2 | p_feasible | 1 | 4000 | 1657 | 0.4143 | 0.3991 | 0.4296 |  |  |  |  | sin referencia del simulador |
| toy2 | p_opt | 1 | 4000 | 33 | 0.00825 | 0.005881 | 0.01156 |  |  |  |  | sin referencia del simulador |
| tsm_ciclo_n16 | p_feasible | 1 | 4000 | 1161 | 0.2903 | 0.2764 | 0.3045 | 0.5339 | 2136 | 0.5437 | 1.565e-213 | hardware < simulador |
| tsm_ciclo_n16 | p_opt | 1 | 4000 | 7 | 0.00175 | 0.000848 | 0.003608 | 0.00159 | 6.36 | 1.101 | 0.6932 | no concluyente (< 10 optimos esperados) |
| tsm_ciclo_n20 | p_feasible | 1 | 4000 | 767 | 0.1918 | 0.1798 | 0.2042 | 0.4502 | 1801 | 0.4259 | 4.969e-259 | hardware < simulador |
| tsm_ciclo_n20 | p_opt | 1 | 4000 | 1 | 0.00025 | 4.413e-05 | 0.001415 | 0.0002687 | 1.075 | 0.9304 | 0.7083 | no concluyente (< 10 optimos esperados) |
| tsm_ciclo_n24 | p_feasible | 1 | 4000 | 569 | 0.1422 | 0.1318 | 0.1534 | 0.3894 | 1558 | 0.3653 | 6.175e-260 | hardware < simulador |
| tsm_ciclo_n24 | p_opt | 1 | 4000 | 0 | 0 | 0 | 0.0009594 | 4.547e-05 | 0.1819 | 0 | 0.8337 | no concluyente (< 10 optimos esperados) |
| tsm_aleatoria_n16 | p_feasible | 1 | 4000 | 1198 | 0.2995 | 0.2855 | 0.3139 | 0.3355 | 1342 | 0.8928 | 6.294e-07 | hardware < simulador |
| tsm_aleatoria_n16 | p_opt | 1 | 4000 | 5 | 0.00125 | 0.000534 | 0.002923 | 0.00334 | 13.36 | 0.3743 | 0.008403 | hardware < simulador |
| tsm_aleatoria_n18 | p_feasible | 1 | 4000 | 789 | 0.1973 | 0.1852 | 0.2099 | 0.265 | 1060 | 0.7443 | 7.747e-24 | hardware < simulador |
| tsm_aleatoria_n18 | p_opt | 1 | 4000 | 0 | 0 | 0 | 0.0009594 | 0.0003507 | 1.403 | 0 | 0.2458 | no concluyente (< 10 optimos esperados) |
| tsm_aleatoria_n22 | p_feasible | 1 | 4000 | 537 | 0.1343 | 0.124 | 0.1452 | 0.1788 | 715.3 | 0.7507 | 1.47e-14 | hardware < simulador |
| tsm_aleatoria_n22 | p_opt | 1 | 4000 | 2 | 0.0005 | 0.0001371 | 0.001821 | 0.0009217 | 3.687 | 0.5425 | 0.2876 | no concluyente (< 10 optimos esperados) |
