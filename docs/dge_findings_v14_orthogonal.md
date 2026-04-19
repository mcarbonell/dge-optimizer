# DGE — Hallazgos v14: Particionado Ortogonal y Sinceridad Estadística ⚖️

**Fecha:** 2026-04-19  
**Estado:** ¡Base matemática saneada!  
**Archivo:** `scratch/dge_orthogonal_v14.py`

---

## 🏆 Resultado headline

> **DGE v14 elimina el ruido cruzado mediante particiones ortogonales, manteniendo un 86% en MNIST con una base teórica defendible.**  
> Tras recibir feedback técnico externo, se identificó que el uso de bloques aleatorios solapados introducía inconsistencias en el EMA. La v14 implementa una partición estricta del espacio de parámetros, asegurando que cada variable se evalúe exactamente una vez por paso.

---

## El Experimento: Limpieza de la Señal

Se comparó la eficiencia de la estimación del gradiente analizando la correlación entre el gradiente ruidoso del paso (`g_step`) y la acumulación histórica del EMA (`mh`).

### Resultados v14 (100,000 Evals)

| Métrica | Valor | Observación |
| :--- | :--- | :--- |
| **Test Accuracy** | **85.83%** | Estable y rápido (25.6s en CPU). |
| **SNR Correlation** | **0.34 - 0.46** | La señal es real pero ruidosa. |
| **Coste por paso** | $2 \log_2 D$ | Matemáticamente exacto y sin solapes. |

---

## Hallazgos Clave

1.  **Validación de la Ortogonalidad**: Usar `np.array_split` sobre una permutación aleatoria del espacio de parámetros limpia el proceso de actualización. Se elimina el problema de variables evaluadas múltiples veces o ninguna en un mismo paso.
2.  **Límite del SNR**: La correlación (~0.4) confirma que el "Denoising temporal" funciona pero tiene límites físicos. Un bloque de 1,600 variables compartiendo una sola métrica escalar siempre tendrá un componente de ruido intrínseco masivo.
3.  **Hacia la v15**: Se observa que el algoritmo es muy robusto, pero el "paso Greedy" (que se desactivó en algunas pruebas de la v13) podría re-introducirse con un esquema de decaimiento (decay) para ayudar a escapar de mesetas sin arruinar la convergencia final.

---

## Conclusión

La v14 transforma DGE de una "heurística inspirada" en un **Block-SPSA con Memoria por Coordenada** riguroso. Los resultados en MNIST son consistentes con las versiones anteriores, pero ahora el algoritmo es más rápido y su comportamiento es predecible y analizable.

---
*Documento creado tras sanear la arquitectura del optimizador — 2026-04-19.*
