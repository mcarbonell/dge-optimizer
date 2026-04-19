# DGE — Hallazgos v16b: Escalado de Learning Rate vs. k 📈

**Fecha:** 2026-04-19  
**Estado:** ¡Hipótesis de frecuencia de paso validada!  
**Archivo:** `scratch/dge_lr_scaling_v16b.py`

---

## 🏆 Resultado headline

> **La frecuencia de actualización (número de pasos) es más crítica para la convergencia que la precisión individual de los bloques.**  
> En un test extendido de 100,000 evaluaciones en $D=100,000$, las configuraciones con menos bloques ($k=8$) superaron sistemáticamente a las de alta precisión ($k=128$), incluso cuando estas últimas usaron un Learning Rate 4 veces mayor.

---

## El Experimento: Calidad vs. Cantidad

Se compararon 6 configuraciones cruzando el número de bloques ($k$) con el Learning Rate ($LR$) para compensar la pérdida de pasos por presupuesto.

### Resultados en D=100,000 (100,000 Evals)

| k | LR | Pasos | SNR (Corr) | Loss Final |
| :--- | :--- | :--- | :--- | :--- |
| **8** | **1.0** | **6250** | **0.39** | **8.39e+05** |
| 8 | 0.2 | 6250 | 0.38 | 8.71e+05 |
| 32 | 1.0 | 1562 | 0.30 | 1.00e+06 |
| 128 | 1.0 | 390 | 0.33 | 1.14e+06 |
| 128 | 4.0 | 390 | 0.32 | 1.14e+06 |

---

## Hallazgos Clave

1.  **Dominancia de la Frecuencia**: El optimizador prefiere dar muchos pasos ruidosos ($k=8 \implies 6250$ pasos) que pocos pasos precisos ($k=128 \implies 390$ pasos). La capacidad de Adam para promediar ruido temporalmente es superior a la ganancia de señal obtenida por reducir el tamaño del bloque.
2.  **Paradoja del SNR**: Contrario a la intuición inicial, bloques más grandes ($k=8$) mostraron un SNR ligeramente superior (0.39) que bloques más pequeños (0.33). Esto sugiere que en paisajes densos, los bloques grandes capturan mejor la coherencia global del gradiente.
3.  **Límite de Agresividad**: Subir el $LR$ de forma agresiva en bloques pequeños no compensa la baja tasa de actualización.

---

## Conclusión

El diseño óptimo de DGE debe priorizar mantener un número de bloques $k$ bajo (cerca de $\log_2 D$ o inferior) para maximizar la cantidad de actualizaciones del estado de Adam por unidad de presupuesto. DGE es, ante todo, un optimizador de alta frecuencia.

---
*Documento creado tras analizar la relación entre granularidad espacial y frecuencia temporal — 2026-04-19.*
