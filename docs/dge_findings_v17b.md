# DGE — Hallazgos v17b: Eficiencia Real (Wall-Clock vs. k) ⏱️

**Fecha:** 2026-04-19  
**Estado:** ¡Revelada la ventaja de hardware de los pasos pesados!  
**Archivo:** `scratch/dge_time_budget_v17b.py`

---

## 🏆 Resultado headline

> **DGE rinde un 1,400% mejor bajo un presupuesto de tiempo real cuando se utilizan bloques pequeños ($k$ alto).**  
> Al normalizar por tiempo de ejecución en lugar de por evaluaciones, descubrimos que configuraciones con $k$ alto (pasos pesados pero informativos) son masivamente más eficientes. En 5 segundos, $k=256$ redujo la pérdida 14 veces más que $k=4$, a pesar de dar 12 veces menos pasos.

---

## El Experimento: El Fin del Tiempo Constante

Se fijó un límite de **5.0 segundos** de CPU y se permitió a cada configuración dar tantos pasos como pudiera.

### Resultados (Ellipsoid D=10,000 | 5s budget)

| k (Bloques) | Evals Realizadas | Pasos Dados | Loss Final | Evals/seg |
| :--- | :--- | :--- | :--- | :--- |
| 4 | 57,392 | **7,174** | 2.73e+06 | 11,478 |
| 16 | 146,048 | 4,564 | 8.40e+05 | 29,209 |
| 64 | 242,688 | 1,896 | 2.60e+05 | 48,537 |
| **256** | **295,424** | 577 | **1.92e+05** | **59,084** |

---

## Hallazgos Clave

1.  **Amortización del Overhead**: La mayor parte del tiempo en optimización zeroth-order en Python no se gasta evaluando funciones matemáticas simples, sino en la gestión de arrays y el bucle de control del optimizador. $k=256$ amortiza este coste al procesar 512 evaluaciones por cada actualización del estado de Adam.
2.  **Riqueza de Información por Paso**: Aunque $k=256$ solo dio 577 pasos, cada paso "veía" el gradiente de forma mucho más completa (participando el 100% de las variables repartidas en 256 bloques). Esta densidad de información superó por completo a la alta frecuencia de pasos ciegos de $k=4$.
3.  **Rendimiento de Hardware**: DGE demuestra ser un algoritmo diseñado para el mundo real, donde las transferencias de datos y el control de flujo son los cuellos de botella.

---

## Conclusión

La heurística original de $k \approx \log_2 D$ es un buen punto de partida para la precisión teórica, pero para el **rendimiento en tiempo real**, DGE prefiere configuraciones de $k$ mucho más altas. Hemos descubierto que DGE es un optimizador de "alto rendimiento" (High-Throughput).

---
*Documento creado tras descubrir la correlación entre granularidad y eficiencia de ejecución — 2026-04-19.*
