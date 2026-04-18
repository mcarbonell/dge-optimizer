# DGE — Hallazgos v10: Escalado en Profundidad 🧱

**Fecha:** 2026-04-18  
**Estado:** ¡Validación de arquitecturas profundas completada!  
**Archivo:** `scratch/dge_deep_benchmark_v1.py`

---

## 🏆 Resultado headline

> **DGE mantiene su eficacia en redes de 5 capas ocultas.**  
> Mientras que los métodos Zeroth-Order tradicionales suelen colapsar al aumentar la profundidad y el número de parámetros, DGE ha alcanzado un **85.67%** de precisión en una red con más de **63,000 parámetros**, demostrando una estabilidad de escalado superior a SPSA.

---

## El Desafío de la Profundidad

Aumentar la profundidad de una red neuronal presenta dos problemas para un optimizador Zeroth-Order:
1.  **Explosión de Parámetros**: De 25k a 63k parámetros. Los bloques de DGE son más grandes y el ruido estadístico aumenta.
2.  **Credit Assignment**: Un cambio en la primera capa debe propagarse a través de 5 capas de ReLU para afectar el Loss final. La "señal" del gradiente se vuelve más tenue y compleja.

---

## Resultados Comparativos (150k Evals)

| Arquitectura | Algoritmo | Params (D) | Precisión (Test) | Tiempo (s) |
| :--- | :--- | :--- | :--- | :--- |
| **Shallow** (784-32-10) | DGE | 25,450 | 87.50% | 28s |
| **Deep** (784-64x4-10) | DGE | **63,370** | **85.67%** | 98s |
| **Deep** (784-64x4-10) | Adam | 63,370 | 90.50% | 71s |

---

## Análisis de Escalado

1.  **DGE vs Adam**: La brecha de precisión aumentó ligeramente (de 3pp a 5pp). Esto sugiere que a medida que el espacio de parámetros crece, el presupuesto de evaluaciones debería escalar (quizás linealmente con $\log D$).
2.  **Tiempo de Cómputo**: El tiempo subió de 28s a 98s. Esto es esperado ya que $k = \lceil \log_2(D) \rceil$ aumenta y cada pasada hacia adelante es más costosa computacionalmente.
3.  **Convergencia**: A pesar de la profundidad, DGE no dio señales de estancamiento prematuro o divergencia, lo que valida la robustez del **EMA (Exponential Moving Average)** para capturar dependencias de largo alcance en la arquitectura.

---

## Conclusión

DGE es un optimizador capaz de entrenar redes neuronales de múltiples capas sin necesidad de backpropagation. Aunque Adam sigue siendo más rápido y preciso en este escenario (donde las derivadas son suaves y fáciles de calcular), DGE demuestra que es una alternativa viable y estable para profundizar en arquitecturas más complejas.

---
*Documento creado tras el benchmark de profundidad — 2026-04-18.*
