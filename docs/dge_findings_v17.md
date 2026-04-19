# DGE — Hallazgos v17: Mal Acondicionamiento (Ellipsoid) 🌌

**Fecha:** 2026-04-19  
**Estado:** ¡Punto óptimo de granularidad identificado!  
**Archivo:** `scratch/dge_ellipsoid_test_v17.py`

---

## 🏆 Resultado headline

> **En paisajes mal acondicionados, $k \approx \log_2 D$ emerge como el equilibrio perfecto entre resolución de señal y frecuencia de paso.**  
> Al probar la función Ellipsoid (donde las dimensiones varían 1000x en escala), la configuración $k=16$ superó tanto a bloques más grandes como más pequeños, validando la heurística original del algoritmo para entornos heterogéneos.

---

## El Experimento: El "Dulce Punto" del SNR

Se evaluó el rendimiento en $D=10,000$ con un presupuesto fijo de 50,000 evaluaciones, variando $k$.

### Resultados en Ellipsoid (50,000 Evals)

| k | Resolución de Bloque | Pasos | Loss Final | Wall-Clock (s) |
| :--- | :--- | :--- | :--- | :--- |
| 4 | Baja (2,500 vars) | 6250 | 1.94e+06 | 4.87s |
| **16** | **Media (625 vars)** | **1562** | **1.50e+06** | **1.82s** |
| 64 | Alta (156 vars) | 390 | 4.00e+06 | 1.05s |

---

## Hallazgos Clave

1.  **Aislamiento de Señal**: En la Esfera (isótropa), $k$ bajo siempre ganaba por frecuencia. En el Ellipsoid, bloques demasiado grandes ($k=4$) mezclan dimensiones muy sensibles con insensibles, "ahogando" la señal de descenso. $k=16$ proporciona la resolución necesaria para que Adam distinga las escalas.
2.  **Paradoja de la Eficiencia Temporal**: Existe una correlación inversa masiva entre $k$ y el tiempo de ejecución. Configurar $k=64$ fue **4.6 veces más rápido** que $k=4$ para el mismo número de evaluaciones. Esto se debe a que el overhead de gestión de arrays en Python/NumPy se amortiza mejor al dar menos pasos pero más densos en evaluaciones.
3.  **Hacia la v17b (Time-Budgeting)**: Dado que $k$ alto es mucho más rápido por evaluación, una comparativa justa "Wall-Clock to Wall-Clock" podría cambiar las conclusiones sobre el valor óptimo de $k$.

---

## Conclusión

El valor de $k$ no es solo un parámetro de precisión, es un parámetro de **eficiencia de hardware**. En entornos donde el coste de evaluar la función es bajo comparado con el overhead del optimizador, $k$ altos podrían ser superiores por pura fuerza bruta temporal.

---
*Documento creado tras analizar el impacto del acondicionamiento en la granularidad del bloque — 2026-04-19.*
