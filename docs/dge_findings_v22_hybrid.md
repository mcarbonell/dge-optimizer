# Hallazgos de Investigación - DGE v22 (SFWHT + DGE Híbrido)

## 1. Objetivo de la Iteración
Siguiendo las recomendaciones teóricas de la versión `v21`, el experimento `v22` implementó el **Enfoque Híbrido**.
En lugar de depender exclusivamente de SFWHT *Peeling* para estimar el gradiente exacto (lo cual falla bajo una alta densidad de variables activas debido a las inevitables colisiones en los *buckets*), el algoritmo utiliza SFWHT únicamente como un "radar de bajo coste". Tras identificar qué regiones del espacio de parámetros tienen más señal, lanza el optimizador original de DGE (Denoised Gradient Estimation) de forma quirúrgica *solo* sobre esas regiones.

## 2. Arquitectura del Híbrido (3 Fases por Paso)
1. **SFWHT Scan (El Radar):** Se calcula una sola transformada simétrica base (sin *shifts* ni decodificación bit a bit) usando *buckets* de tamaño $B$. Se identifican los *buckets* con mayor energía.
2. **DGE Refinement:** Se extraen las variables subyacentes que mapean a los *buckets* activos. Sobre este subconjunto reducido, se aplican particiones aleatorias de DGE para obtener una estimación de gradiente de alta fidelidad.
3. **Adam Update:** El gradiente refinado se pasa a Adam. Las variables que no fueron escaneadas tienen un gradiente de cero en este paso (y su *Momentum* decae orgánicamente).

## 3. Configuración del Experimento
- **Archivo:** `scratch/dge_sfwht_hybrid_v22.py`
- **Evaluaciones por Paso:** $\approx 1,304$ evaluaciones (un ahorro inmenso comparado con la `v21`).
- **Pasos Totales:** 154 pasos de actualización (Adam).
- **Capa 1 (25,000 params):** $B=512$, top $50$ buckets, DGE $k=8$.
- **Capa 2 (330 params):** $B=128$, top $12$ buckets, DGE $k=4$.

## 4. Resultados Cuantitativos
Resultados crudos guardados en `results/raw/sfwht_hybrid_v22.json`.
- **Accuracy Final (Test):** **63.33%**
- **Loss Final:** $1.4514$
- **Estabilidad:** Convergencia suave, monótona, sin ningún pico de divergencia.
- **Rendimiento Computacional:** $13.0$ segundos reales.

## 5. Conclusiones y Análisis
1. **Ruptura del Techo de Cristal:** La arquitectura híbrida no solo resolvió el problema del "crash" de la `v20`, sino que superó dramáticamente el techo del 43% de la `v21`, llegando a más del **63%**. Al descargar la responsabilidad de la "estimación exacta" hacia DGE, sorteamos el límite matemático de las colisiones de Hadamard.
2. **Eficiencia de Evaluaciones:** Conseguimos subir el número de pasos útiles (actualizaciones de pesos) de 25 a más de 150 usando el mismo presupuesto de evaluaciones ($200,000$), al eliminar los costosos *shifts* y dejar el SFWHT base en estado de "solo escáner".
3. **Sinergia:** Esta prueba demuestra que el paradigma DGE (rutado de impulso en bloques ruidosos) y SFWHT (identificación de espectro estructurado) son altamente complementarios. SFWHT encuentra la aguja en el pajar, y DGE mide la aguja de forma precisa y barata.

## 6. Próximos Pasos (Hacia la v23)
Para alcanzar de nuevo la marca superior al **85%** vista en el baseline `v9` de DGE puro:
- **Afinar la Tasa de Escaneo:** Un $10\%$ de buckets retenidos tal vez sea un corte muy agresivo o muy bajo. Podemos permitir que el "top K" sea dinámico basado en un umbral estadístico estricto sobre el fondo de ruido.
- **Sincronización del *Learning Rate*:** Incrementar significativamente los bloques DGE de refinamiento en las últimas etapas del entrenamiento, cuando todas las variables necesitan ser micro-afinadas para que el modelo converja en el valle óptimo.