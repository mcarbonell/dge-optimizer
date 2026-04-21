# Hallazgos de Investigación - DGE v25 (Scaling Head-to-Head: Hybrid vs Pure)

## 1. Objetivo de la Iteración
El objetivo de la iteración `v25` era resolver la pregunta central planteada en el `research_shortlist.md`: encontrar el punto de cruce en la escalabilidad donde el modelo Híbrido SFWHT+DGE supere en precisión al DGE puro bajo el mismo presupuesto de evaluaciones.

La hipótesis establecía que, debido a que el coste del escáner SFWHT es logarítmico/constante $O(B)$ y el de DGE es lineal $O(D/k)$, el modelo híbrido debería ser más eficiente en redes de mayor tamaño (p. ej., $\ge 100K$ parámetros).

Para probar esto, se escaló el modelo de la red neuronal a una arquitectura Medium de aproximadamente $\sim 110K$ parámetros (`784 -> 128 -> 64 -> 10`), incrementando el presupuesto a 800,000 evaluaciones.

## 2. Configuración del Experimento
- **Arquitectura:** `BatchedMLP([784, 128, 64, 10])` (~109,386 parámetros)
- **Presupuesto Total:** 800,000 evaluaciones de la función objetivo.
- **Pure DGE Config:**
  - `DGE_BLOCKS = [1024, 128, 16]`
  - $\sim 2336$ evaluaciones por paso (sin escáner).
- **Hybrid SFWHT+DGE Config:**
  - SFWHT Buckets: `B_LIST = [1024, 256, 32]`
  - Top-K Buckets: `TOP_K = [200, 50, 8]`
  - DGE Blocks: `DGE_BLOCKS = [256, 32, 8]`
  - Scan Interval: 4
  - $\sim 1248$ evaluaciones medias por paso.

## 3. Resultados Cuantitativos

| Optimizador | Evals Gastadas | Steps (Adam) | Test Accuracy (Peak) | Loss Final |
|-------------|----------------|--------------|----------------------|------------|
| **Pure DGE** | ~801,248 | 343 | **81.17%** | 0.1711 |
| **Hybrid v25** | ~803,156 | 581 | **65.50%** | 0.8500 |

## 4. Análisis y Conclusiones
1. **Ausencia del Crossover y Fracaso del Híbrido:** La hipótesis resultó ser **FALSA**. El modelo híbrido SFWHT+DGE no solo no superó a Pure DGE, sino que colapsó severamente. El híbrido alcanzó un pico temprano de $\sim 65\%$ de precisión y luego la pérdida de entrenamiento se desestabilizó, logrando apenas un 60.5% hacia el final de la ejecución. 
2. **Robustez de Pure DGE:** A pesar del gran incremento en la dimensionalidad (de 25K a 110K parámetros), el DGE puro con bloques escalados ($k=1024$ en la primera capa) convergió de manera estable a **81.17%**. Esto confirma que el mecanismo base de bloques DGE aleatorios superpuestos + EMA es extraordinariamente robusto al escalado.
3. **El Límite Físico de SFWHT:** Como se vio en la iteración `v20`, a medida que las redes densas crecen, las colisiones en los buckets de SFWHT introducen un sesgo insalvable. Aunque logramos mitigar esto en redes de 25K parámetros (`v24`), en una escala de 100K parámetros el escáner SFWHT devuelve ruido estructural que corrompe la señal del refinamiento DGE, provocando divergencia.

## 5. Siguientes Pasos
Siguiendo las instrucciones de la `research_shortlist.md`:
Dado que el híbrido fracasó al escalar, **se da por cerrada y finalizada la línea de investigación SFWHT**.
Se debe hacer un pivote hacia el siguiente punto de la lista: **Vector Group DGE**.
La atención debe enfocarse enteramente en mejorar y potenciar las capacidades nativas del Pure DGE, el cual ha probado ser la aproximación óptima en dimensionalidad densa.