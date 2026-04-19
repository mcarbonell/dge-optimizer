# DGE — Hallazgos v17c: Límites del Throughput Extremo 🏎️

**Fecha:** 2026-04-19  
**Estado:** ¡Saturación de hardware mapeada!  
**Archivo:** `scratch/dge_extreme_throughput_v17c.py`

---

## 🏆 Resultado headline

> **El punto dulce de eficiencia real de DGE en $D=10,000$ se encuentra en $k=256$, logrando procesar ~70,000 evaluaciones por segundo.**  
> Al forzar el algoritmo al límite bajo un presupuesto de tiempo estricto (5.0 segundos), descubrimos que seguir aumentando el tamaño del bloque ($k > 256$) aporta rendimientos decrecientes en evaluaciones por segundo y destruye la convergencia por falta de pasos de actualización.

---

## El Experimento: Búsqueda del Techo de Cristal

Para comparar justamente con la v17b, se fijó la dimensión en $D=10,000$ y el presupuesto de tiempo en $5.0$ segundos. Se exploraron configuraciones extremas de bloques pequeños ($k$ alto).

### Resultados (Ellipsoid D=10,000 | 5.0s budget)

| k (Bloques) | Evals Realizadas | Pasos Dados | Evals/seg | Loss Final |
| :--- | :--- | :--- | :--- | :--- |
| 64 | 308,608 | 2,411 | 61,709 | 2.37e+05 |
| **256** | **347,648** | **679** | **69,468** | **1.26e+05** |
| 1024 | 354,304 | 173 | 70,715 | 2.88e+06 |
| 4096 | 360,448 | 44 | 71,634 | 1.26e+07 |

---

## Hallazgos Clave

1.  **Saturación del Hardware**: Pasar de $k=64$ a $k=256$ otorgó un aumento del 12% en evaluaciones por segundo. Sin embargo, subir a $k=4096$ apenas aportó un 3% extra de rendimiento bruto. Hemos llegado al techo computacional del particionado en NumPy.
2.  **Colapso por Falta de Frecuencia**: A partir de $k=1024$, el algoritmo solo fue capaz de dar 173 pasos. Aunque la señal de cada paso era extremadamente pura (al evaluar 2048 perturbaciones por paso), Adam no tuvo suficientes iteraciones para descender el valle del Ellipsoid.
3.  **El "Golden Ratio" Temporal**: Para $D=10,000$, la configuración **$k=256$** (donde los bloques tienen apenas ~39 variables) es el ganador absoluto si nos importa el tiempo de reloj. Representa el punto exacto donde la curva de eficiencia de hardware se cruza con la necesidad mínima de frecuencia de actualización de pesos.

---

## Conclusión

DGE tiene dos regímenes de operación:
1. **Budget-constrained (Evaluaciones limitadas):** Usar $k$ bajo (cerca de $\log_2 D$) para maximizar los pasos dados.
2. **Time-constrained (Tiempo de pared limitado):** Usar $k$ muy alto (ej. $D/40$) para maximizar el throughput de hardware, amortizando el overhead interno de Python/NumPy.

---
*Documento creado tras estresar el motor de procesamiento por bloques hasta la saturación — 2026-04-19.*
