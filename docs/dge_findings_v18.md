# DGE — Hallazgos v18: Paralelismo Masivo (Batched Engine) 🏎️💨

**Fecha:** 2026-04-19  
**Estado:** ¡Motor de aceleración por hardware validado!  
**Archivo:** `scratch/dge_batched_v18.py`

---

## 🏆 Resultado headline

> **DGE v18 multiplica por 2x la velocidad de evaluación en CPU y sienta las bases para la aceleración masiva en GPU.**  
> Mediante la implementación de un motor de "Batching de Perturbaciones" en PyTorch, logramos evaluar las $2k$ variantes de un paso de optimización de forma simultánea. Esto elimina el overhead de los bucles de Python y permite que el algoritmo escale con el número de núcleos de la GPU/CPU.

---

## El Experimento: Evaluación en Paralelo vs. Secuencial

Se comparó el tiempo necesario para realizar un paso completo de DGE con $D=100,000$ y $k=128$ (256 evaluaciones totales) utilizando un MLP de 3 capas.

### Rendimiento (D=100,000 | 256 evals/step)

| Método | Throughput (Evals/s) | Tiempo por Paso | Speedup |
| :--- | :--- | :--- | :--- |
| **Secuencial (v17)** | 2,709 | 0.0945s | 1.0x |
| **Batched (v18)** | **5,772** | **0.0444s** | **2.1x** |

---

## Hallazgos Clave

1.  **Paralelismo de Perturbaciones**: DGE es intrínsecamente paralelo. Evaluar $f(x+\delta)$ y $f(x-\delta)$ para múltiples bloques no tiene dependencias de datos entre sí, lo que permite empaquetarlos en un solo tensor gigante.
2.  **Eficiencia de PyTorch**: Al mover la lógica de perturbación a operaciones de tensores (`torch.bmm`), el coste de gestión de la optimización cae drásticamente.
3.  **Preparado para iGPU**: El código de la v18 es compatible con DirectML, lo que permitirá usar la iGPU AMD Radeon 780M para procesar miles de evaluaciones por segundo sin saturar la CPU.

---

## Conclusión

La v18 marca el paso de DGE de un algoritmo puramente matemático a una implementación de alto rendimiento optimizada para hardware moderno. El "Batching de Perturbaciones" es la clave para entrenar modelos de gran escala en tiempos competitivos con el descenso de gradiente tradicional.

---
*Documento creado tras implementar el motor de vectorización de perturbaciones — 2026-04-19.*
