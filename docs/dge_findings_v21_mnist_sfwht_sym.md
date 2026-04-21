# Hallazgos de Investigación - DGE v21 (SFWHT Simétrico)

## 1. Objetivo de la Iteración
El experimento `v21` aplicó el primer bloque de correcciones matemáticas al enfoque de Layer-wise SFWHT implementado en la `v20`. El objetivo era mitigar el colapso ("spike and crash") del modelo una vez que los gradientes perdían su esparsidad inicial.
Las mejoras introducidas (según el análisis `analysis_v20_sfwht.md`) fueron:
1. **Evaluaciones Simétricas:** Calcular `(f(x+e) - f(x-e)) / 2e` para eliminar el sesgo "DC" que contamina todos los buckets de la transformada.
2. **Tamaños de Bucket ($B$) Dinámicos:** Asignar $B=512$ para la Capa 1 y $B=128$ para la Capa 2 para mantener una densidad de variables controlada.
3. **Cosine Annealing (Decay):** Reducción progresiva de `lr` y `epsilon`.

## 2. Configuración del Experimento
- **Archivo:** `scratch/dge_sfwht_mnist_v21.py`
- **Evaluaciones por Paso:** $\approx 7,936$ (debido al doble coste de la simetría y los shifts requeridos).
- **Pasos Totales:** 25 pasos reales (Adam updates) en 200,000 evaluaciones.

## 3. Resultados Cuantitativos (Métricas)
Los resultados detallados se encuentran en `results/raw/sfwht_mnist_v21.json`.
- **Estabilidad Total:** El loss disminuyó monótonamente de `2.639` a `1.696`.
- **Accuracy Final (Techo):** **42.67%** en Entrenamiento y Test.
- **Rendimiento:** 13.0 segundos (extremadamente rápido debido a la baja cantidad de pasos Adam totales).

## 4. Conclusiones y Análisis (Diagnostics)
1. **Fin del Colapso (Crash Evitado):** Como teorizó el análisis previo, el "spike and crash" fue eliminado con éxito. El aprendizaje fue 100% estable gracias a la simetría de la transformada (que eliminó falsos positivos en el Peeling) y el decaimiento de hiperparámetros.
2. **El Techo de Colisiones (Precision Ceiling):** Aunque es estable, el optimizador no logró pasar del $\approx 43\%$. Esto confirma el problema fundamental de SFWHT como estimador directo bajo ratios altos de Variables/Buckets (D/B alto): **Las colisiones no se pueden resolver**. Cuando los gradientes dejan de ser ruidosos y esparsos (early training) y se vuelven una "sopa densa", el Peeling no puede separar el ruido de fondo, provocando que las estimaciones se vuelvan "borrosas" impidiendo alcanzar el 80%+ que lograba el algoritmo DGE puro.

## 5. Próximos Pasos (El Enfoque Híbrido)
El techo del 43% demuestra que SFWHT no debe usarse como *estimador final* en redes densas. La `v22` debe implementar el **Enfoque Híbrido**:
1. Hacer un SFWHT Scan ultrabarato (solo la matriz base, sin shifts).
2. Quedarse solo con las variables que cayeron en el $10\%$ de buckets con mayor magnitud.
3. Aplicar Diferencias Finitas aisladas o DGE aleatorio **solamente** a ese $10\%$ de variables.
Esto debería lograr eficiencias 10x respecto a DGE puro, pero sin perder la precisión necesaria para llegar al $85\%+$ de Accuracy en MNIST.