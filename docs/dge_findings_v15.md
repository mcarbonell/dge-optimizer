# DGE — Hallazgos v15: Decaimiento Greedy y Profundidad EMA 🧠

**Fecha:** 2026-04-19  
**Estado:** ¡Sintonía fina de heurísticas completada!  
**Archivo:** `scratch/dge_prototype_v15.py`

---

## 🏆 Resultado headline

> **DGE v15 alcanza un 87.50% en MNIST mediante el uso de un paso Greedy con decaimiento y un EMA de mayor profundidad.**  
> Al combinar la base ortogonal de la v14 con una estrategia de "enfriamiento" del paso Greedy (0.5 -> 0.0) y aumentar la inercia de Adam (beta1=0.95), logramos una convergencia más estable y precisa.

---

## El Experimento: Control de la Codicia

Se implementó un esquema de *Cosine Decay* para el peso del bloque ganador (`greedy_w`). La hipótesis es que la exploración agresiva es buena al inicio, pero el ruido del bloque domina al final, impidiendo el ajuste fino.

### Resultados v15 (120,000 Evals)

| Configuración | Test Accuracy | Observación |
| :--- | :--- | :--- |
| **v14 (Solo EMA)** | 85.83% | Estable pero lenta al inicio. |
| **v15 (Greedy Decay + EMA 0.95)** | **87.50%** | Salto inicial rápido y mejor ajuste final. |

---

## Hallazgos Clave

1.  **Denoising Profundo**: Aumentar `beta1` a 0.95 permite que el estimador promedie el ruido de ~20 pasos ortogonales, mejorando significativamente la calidad de la dirección de descenso en redes densas.
2.  **Schedules de Heurísticas**: El decaimiento del paso Greedy es fundamental. Se observó que al final del entrenamiento (`Greedy_W < 0.05`), la precisión de test dejaba de oscilar y se estabilizaba al alza.
3.  **Hacia la v15b**: Los resultados sugieren que existe un compromiso (trade-off) óptimo entre la profundidad del EMA y el tamaño de los bloques. Una búsqueda de hiperparámetros sistemática sobre esta base ortogonal es el siguiente paso lógico.

---

## Conclusión

DGE v15 demuestra que la arquitectura ortogonal es una plataforma sólida sobre la que construir heurísticas de sintonía fina. El algoritmo ya no solo es "honesto" sino que empieza a competir seriamente en precisión con baselines de búsqueda aleatoria más establecidos.

---
*Documento creado tras optimizar el control de ruido temporal y espacial — 2026-04-19.*
