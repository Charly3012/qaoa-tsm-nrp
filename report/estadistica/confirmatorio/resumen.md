# Analisis estadistico - CONFIRMATORIO

Entradas: 65 CSV; 155 instancias; semillas de datos [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59].

Sin problemas de integridad.

## Hipotesis (familia de Holm)

| hipotesis | familia | n | prueba | estadistico | p | p_holm | efecto | decision |
|---|---|---|---|---|---|---|---|---|
| H1 | tsm_aleatoria | 40 | Wilcoxon 1 muestra (>0) | 820 | 9.095e-13 | 5.457e-12 | mediana A = 54.5 [IC95 15.1, 82.8]; signos 40/40 (p = 9.09e-13) | Rechaza H0 |
| H2 | tsm_aleatoria | 40 | Wilcoxon pareado (F+C vs F) | 746 | 3.94e-07 | 1.576e-06 | FS = 0.93; mediana de la razon de A = 2.1 | Rechaza H0 |
| H3 | tsm_aleatoria | 40 | Wilcoxon pareado (RIC F+C+I < RIC F+C) | 11 | 5.002e-11 | 2.501e-10 | FS = 0.93; mediana de RIC: 0.00563 (F+C+I) frente a 1.03 (F+C) | Rechaza H0 |
| H4 | tsm_aleatoria | 40 | McNemar exacto | 21 | 4.768e-07 | 1.576e-06 | b = 21 (QAOA si, azar no), c = 0; QAOA 40/40, azar 19/40 | Rechaza H0 |
| H6 | nrp | 35 | Wilcoxon 1 muestra (>0) | 172 | 0.9913 | 0.9913 | mediana A = 0.828 [IC95 0.642, 1.03]; signos 13/35 (p = 0.955) | No rechaza H0 |
| H7 | tsm_aleatoria | 15 | Wilcoxon pareado (F vs Base) | 105 | 0.004181 | 0.008362 | FS = 0.93; mediana de la razon de A = 3.45 | Rechaza H0 |

## Analisis secundarios (sin ajuste)

| analisis | n | prueba | estadistico | p | efecto |
|---|---|---|---|---|---|
| TSM aleatoria vs NRP (log A, F+C+I, N 8-20) | 35 / 35 | Mann-Whitney U (TSM > NRP) | 1225 | 3.273e-13 | A12 = 1.00 |
| Exito frente al azar en nrp (F+C+I) | 35 | McNemar exacto | 0 | 1 | b = 0 (QAOA si, azar no), c = 0; QAOA 15/35, azar 15/35 |
| Exito frente al azar en tsm_ciclo (F+C+I) | 40 | McNemar exacto | 15 | 3.052e-05 | b = 15 (QAOA si, azar no), c = 0; QAOA 35/40, azar 20/40 |
| Sin holguras vs con holguras en N = 4 y 6 (TSM aleatoria) | 10 | Wilcoxon pareado (F vs Base) | 41 | 0.09668 | FS = 0.60; mediana de la razon de A = 1.64 |
| Nivel de A: F+C+I vs F+C (TSM aleatoria) | 40 | Wilcoxon pareado | 799 | 4.065e-10 | FS = 0.95; mediana de la razon de A = 1.16 |

## H5: hardware frente a simulador (por instancia)

| instancia | medida | replicas | mediciones | k | p_hw | wilson_inf | wilson_sup | p_sim | esperados | cociente | p | conclusion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsm:toy2[F+C+I] | p_feasible | 3 | 12000 | 4964 | 0.4137 | 0.4049 | 0.4225 | 0.4719 | 5663 | 0.8766 | 7.368e-38 | hardware < simulador |
| tsm:toy2[F+C+I] | p_opt | 3 | 12000 | 128 | 0.01067 | 0.008979 | 0.01267 | 0.01431 | 171.7 | 0.7456 | 0.0002695 | hardware < simulador |
| tsm:tsm_aleatoria_n16[F+C+I] | p_feasible | 3 | 12000 | 3411 | 0.2843 | 0.2762 | 0.2924 | 0.3355 | 4026 | 0.8473 | 1.078e-33 | hardware < simulador |
| tsm:tsm_aleatoria_n16[F+C+I] | p_opt | 3 | 12000 | 6 | 0.0005 | 0.0002292 | 0.001091 | 0.00334 | 40.08 | 0.1497 | 2.522e-11 | hardware < simulador |
| tsm:tsm_ciclo_n16[F+C+I] | p_feasible | 3 | 12000 | 3541 | 0.2951 | 0.287 | 0.3033 | 0.5339 | 6407 | 0.5527 | 0 | hardware < simulador |
| tsm:tsm_ciclo_n16[F+C+I] | p_opt | 3 | 12000 | 5 | 0.0004167 | 0.000178 | 0.0009751 | 0.00159 | 19.08 | 0.2621 | 0.0001435 | hardware < simulador |
| tsm:tsm_ciclo_n20[F+C+I] | p_feasible | 3 | 12000 | 2340 | 0.195 | 0.188 | 0.2022 | 0.4502 | 5402 | 0.4332 | 0 | hardware < simulador |
| tsm:tsm_ciclo_n20[F+C+I] | p_opt | 3 | 12000 | 2 | 0.0001667 | 4.571e-05 | 0.0006075 | 0.0002687 | 3.224 | 0.6203 | 0.3748 | no concluyente (< 10 optimos esperados) |
| tsm:tsm_ciclo_n24[F+C+I] | p_feasible | 3 | 12000 | 1961 | 0.1634 | 0.1569 | 0.1701 | 0.3894 | 4673 | 0.4197 | 0 | hardware < simulador |
| tsm:tsm_ciclo_n24[F+C+I] | p_opt | 3 | 12000 | 1 | 8.333e-05 | 1.471e-05 | 0.0004719 | 4.547e-05 | 0.5456 | 1.833 | 0.8957 | no concluyente (< 10 optimos esperados) |
| tsm:tsm_aleatoria_n18[F+C+I] | p_feasible | 3 | 12000 | 2510 | 0.2092 | 0.202 | 0.2165 | 0.265 | 3180 | 0.7892 | 4.275e-46 | hardware < simulador |
| tsm:tsm_aleatoria_n18[F+C+I] | p_opt | 3 | 12000 | 3 | 0.00025 | 8.503e-05 | 0.0007348 | 0.0003507 | 4.208 | 0.7128 | 0.3938 | no concluyente (< 10 optimos esperados) |
| tsm:tsm_aleatoria_n22[F+C+I] | p_feasible | 3 | 12000 | 1570 | 0.1308 | 0.1249 | 0.137 | 0.1788 | 2146 | 0.7316 | 2.131e-46 | hardware < simulador |
| tsm:tsm_aleatoria_n22[F+C+I] | p_opt | 3 | 12000 | 2 | 0.0001667 | 4.571e-05 | 0.0006075 | 0.0009217 | 11.06 | 0.1808 | 0.001148 | hardware < simulador |
| tsm:toy[Base] | p_feasible | 3 | 12000 | 10438 | 0.8698 | 0.8637 | 0.8757 | 0.9037 | 1.084e+04 | 0.9625 | 1.84e-33 | hardware < simulador |
| tsm:toy[Base] | p_opt | 3 | 12000 | 87 | 0.00725 | 0.005882 | 0.008934 | 0.004917 | 59 | 1.475 | 0.9998 | compatible con el simulador |
| tsm:toy2[Base] | p_feasible | 3 | 12000 | 3358 | 0.2798 | 0.2719 | 0.2879 | 0.2574 | 3088 | 1.087 | 1 | compatible con el simulador |
| tsm:toy2[Base] | p_opt | 3 | 12000 | 10 | 0.0008333 | 0.0004527 | 0.001533 | 0.000963 | 11.56 | 0.8654 | 0.3954 | compatible con el simulador |
| tsm:demo[Base] | p_feasible | 3 | 12000 | 4017 | 0.3347 | 0.3264 | 0.3432 | 0.424 | 5089 | 0.7894 | 1.557e-89 | hardware < simulador |
| tsm:demo[Base] | p_opt | 3 | 12000 | 28 | 0.002333 | 0.001615 | 0.00337 | 0.001886 | 22.63 | 1.237 | 0.889 | compatible con el simulador |
| nrp:toy[Base] | p_feasible | 3 | 12000 | 1920 | 0.16 | 0.1535 | 0.1667 | 0.1884 | 2261 | 0.8494 | 2.725e-16 | hardware < simulador |
| nrp:toy[Base] | p_opt | 3 | 12000 | 10 | 0.0008333 | 0.0004527 | 0.001533 | 0.000315 | 3.779 | 2.646 | 0.9982 | no concluyente (< 10 optimos esperados) |
| nrp:demo[Base] | p_feasible | 3 | 12000 | 408 | 0.034 | 0.0309 | 0.03739 | 0.04457 | 534.8 | 0.7629 | 3.032e-09 | hardware < simulador |
| nrp:demo[Base] | p_opt | 3 | 12000 | 0 | 0 | 0 | 0.00032 | 0.0001698 | 2.038 | 0 | 0.1303 | no concluyente (< 10 optimos esperados) |
| tsm:tsm_aleatoria_n16_d49[F+C+I] | p_feasible | 3 | 12000 | 2545 | 0.2121 | 0.2049 | 0.2195 | 0.2513 | 3016 | 0.8438 | 3.189e-24 | hardware < simulador |
| tsm:tsm_aleatoria_n16_d49[F+C+I] | p_opt | 3 | 12000 | 20 | 0.001667 | 0.001079 | 0.002573 | 0.005503 | 66.04 | 0.3029 | 2.777e-11 | hardware < simulador |
| tsm:tsm_aleatoria_n16_d46[F+C+I] | p_feasible | 3 | 12000 | 2756 | 0.2297 | 0.2222 | 0.2373 | 0.2756 | 3308 | 0.8332 | 8.87e-31 | hardware < simulador |
| tsm:tsm_aleatoria_n16_d46[F+C+I] | p_opt | 3 | 12000 | 11 | 0.0009167 | 0.0005119 | 0.001641 | 0.002579 | 30.94 | 0.3555 | 3.387e-05 | hardware < simulador |
| tsm:tsm_aleatoria_n18_d45[F+C+I] | p_feasible | 3 | 12000 | 2300 | 0.1917 | 0.1847 | 0.1988 | 0.222 | 2664 | 0.8632 | 2.141e-16 | hardware < simulador |
| tsm:tsm_aleatoria_n18_d45[F+C+I] | p_opt | 3 | 12000 | 3 | 0.00025 | 8.503e-05 | 0.0007348 | 0.001579 | 18.95 | 0.1583 | 7.802e-06 | hardware < simulador |
| tsm:tsm_aleatoria_n18_d49[F+C+I] | p_feasible | 3 | 12000 | 2311 | 0.1926 | 0.1856 | 0.1997 | 0.2421 | 2905 | 0.7955 | 8.66e-39 | hardware < simulador |
| tsm:tsm_aleatoria_n18_d49[F+C+I] | p_opt | 3 | 12000 | 4 | 0.0003333 | 0.0001296 | 0.0008568 | 0.001169 | 14.03 | 0.2851 | 0.001758 | hardware < simulador |
| tsm:tsm_aleatoria_n20_d48[F+C+I] | p_feasible | 3 | 12000 | 2365 | 0.1971 | 0.1901 | 0.2043 | 0.2346 | 2815 | 0.8401 | 2.596e-23 | hardware < simulador |
| tsm:tsm_aleatoria_n20_d48[F+C+I] | p_opt | 3 | 12000 | 8 | 0.0006667 | 0.0003379 | 0.001315 | 0.001894 | 22.72 | 0.352 | 0.0003535 | hardware < simulador |
| tsm:tsm_aleatoria_n20_d46[F+C+I] | p_feasible | 3 | 12000 | 1889 | 0.1574 | 0.151 | 0.164 | 0.2199 | 2638 | 0.716 | 2.666e-66 | hardware < simulador |
| tsm:tsm_aleatoria_n20_d46[F+C+I] | p_opt | 3 | 12000 | 4 | 0.0003333 | 0.0001296 | 0.0008568 | 0.001775 | 21.29 | 0.1878 | 5.817e-06 | hardware < simulador |
| tsm:tsm_aleatoria_n22_d49[F+C+I] | p_feasible | 3 | 12000 | 1361 | 0.1134 | 0.1079 | 0.1192 | 0.1624 | 1948 | 0.6986 | 1.578e-52 | hardware < simulador |
| tsm:tsm_aleatoria_n22_d49[F+C+I] | p_opt | 3 | 12000 | 13 | 0.001083 | 0.0006332 | 0.001853 | 0.002921 | 35.05 | 0.3709 | 1.764e-05 | hardware < simulador |
| tsm:tsm_aleatoria_n22_d48[F+C+I] | p_feasible | 3 | 12000 | 1192 | 0.09933 | 0.09411 | 0.1048 | 0.1603 | 1923 | 0.6198 | 2.054e-83 | hardware < simulador |
| tsm:tsm_aleatoria_n22_d48[F+C+I] | p_opt | 3 | 12000 | 0 | 0 | 0 | 0.00032 | 0.001446 | 17.35 | 0 | 2.871e-08 | hardware < simulador |
| nrp:toy2[Base] | p_feasible | 3 | 12000 | 82 | 0.006833 | 0.005509 | 0.008473 |  |  |  |  | sin referencia del simulador |
| nrp:toy2[Base] | p_opt | 3 | 12000 | 0 | 0 | 0 | 0.00032 |  |  |  |  | sin referencia del simulador |
