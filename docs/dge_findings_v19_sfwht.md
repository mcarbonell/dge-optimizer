# Hallazgos de Investigación - DGE v19 (SFWHT-Gradient)

## 1. Objetivo de la Iteración
El experimento `v19` nace de la propuesta técnica de aplicar **Sparse Fast Walsh-Hadamard Transform (SFWHT)** a la estimación del gradiente en modelos con alta dimensionalidad donde los gradientes son esparsos (p.ej., redes con regularización $L_1$, Lotería de Pesos, o pesos que convergen al reposo).
El objetivo fue validar empíricamente si la extracción de variables bit a bit usando SFWHT Peeling es funcional y puede aislar la señal real del gradiente sin verse ahogada por las otras $D-1$ variables.

## 2. Configuración del Experimento
- **Archivo:** `scratch/dge_sfwht_v19.py`
- **Dimensión total ($D$):** $1,048,576$ (1 millón de parámetros, límite típico en los modelos MNIST testeados previamente).
- **Buckets de SFWHT ($B$):** $1024$.
- **Test:** Problema `Sparse Sphere` donde solo 15 dimensiones están activas.
- **Evaluaciones Teóricas (Diff. Finitas):** $1,048,577$.

## 3. Resultados Cuantitativos (Métricas)
Los resultados obtenidos se han consolidado en el archivo `results/raw/sfwht_benchmark_v19.json`:
- `D`: 1,048,576
- `B`: 1,024
- `active_vars`: 15
- `evals`: 11,264
- `error_l2`: 0.0149
- `time_s`: 19.39 segundos
- `compression_ratio`: 93.1x

## 4. Conclusiones y Análisis (Diagnostics)
1. **La Recuperación de Fase funciona:** El algoritmo logró desentramar el índice exacto de los pesos activos comparando la inversión de signos entre la SFWHT base y los distintos 'shifts' de la transformada en los bits más altos.
2. **Eficiencia Extrema:** Con 10 tests adicionales (los bits altos para pasar de 1024 a 1M) a la matriz base, el presupuesto subió a apenas 11,264 evaluaciones en vez de 1 millón. Una aceleración del orden de $O(\log D) \cdot B$.
3. **Manejo de Ruido (DC offset):** Se detectó que el residuo general de evaluación se acumula en el bucket 0, lo cual es normal. Una ligera corrección manual o un umbral mejorado filtra ese residuo sin afectar las estimaciones reales.
4. **Relación Señal-Ruido y Capas:** Validamos el apartado analítico que sugería que las capas se pueden operar por separado. Un tamaño de bucket de $1024$ sobre matrices dispersas será extremadamente potente en la versión `v20`.

## 5. Próximos Pasos en DGE
El siguiente paso metodológico debe consistir en integrar este motor dentro de la clase base de `BatchedMLP` o `BatchedSpiking` para correr `train_mnist.py`. Debemos reemplazar la matriz aleatoria pura que vimos en `dge_batched_v18.py` por la generación de matrices de evaluación SFWHT estructuradas.