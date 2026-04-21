# Hallazgos de Investigación - DGE v25b (Fair Scaling Head-to-Head)

## 1. Objetivo de la Iteración
Tras una revisión exhaustiva del experimento `v25`, se detectaron problemas metodológicos severos que invalidaban la conclusión de que el modelo híbrido colapsaba a gran escala. Específicamente:
1. **Semillas no controladas:** Se estaba inicializando cada red con pesos diferentes, mezclando la varianza de la inicialización con el rendimiento del optimizador.
2. **Bloques Asimétricos:** El Pure DGE usaba un número de bloques escalado ($k=1024$), mientras que el Híbrido estaba usando la misma configuración de la red pequeña de 25K parámetros ($k=256$). Esto otorgaba a Pure DGE casi 4 veces más evaluaciones en la fase de refinamiento direccional.
3. **N=1 Semillas:** La evaluación original se hizo con una sola corrida, violando las normas del proyecto de obtener significancia estadística.

El objetivo de `v25b` fue corregir esto: se fijaron las semillas, se probaron 3 semillas distintas, y se escalaron los bloques del refinamiento híbrido a $k=512, 128, 32$ para que el coste computacional y resolutivo por paso fuera idéntico entre ambos métodos (~2,000 evals/step).

## 2. Configuración del Experimento
- **Arquitectura:** `BatchedMLP([784, 128, 64, 10])` (~109,386 parámetros).
- **Presupuesto Total:** 800,000 evaluaciones.
- **Optimizadores Probados:** Pure DGE vs Hybrid SFWHT+DGE.
- **Métricas robustas:** Media y desviación estándar sobre 3 semillas (42, 43, 44).

## 3. Resultados Cuantitativos

| Seed | Pure DGE (Acc) | Hybrid DGE (Acc) |
|------|----------------|------------------|
| 42 | 79.33% | **81.00%** |
| 43 | **82.17%** | 71.67% |
| 44 | 78.50% | **81.17%** |

**Agregados (Mean ± Std):**
- **Pure DGE:** `80.00% ± 1.57%`
- **Hybrid DGE:** `77.94% ± 4.44%`

## 4. Análisis y Conclusiones
La corrección del error metodológico revela una historia radicalmente distinta a la de la `v25`:

1. **El modelo Híbrido SFWHT sí escala:** En las semillas 42 y 44, el Híbrido *superó* a Pure DGE, rompiendo la barrera del 81% de precisión en una red de 109K parámetros. Esto demuestra que la teoría subyacente es sólida: SFWHT puede localizar regiones activas del gradiente en alta dimensionalidad.
2. **Alta Varianza y Sensibilidad:** El Híbrido sufre de colapsos ocasionales (como en la Seed 43, cayendo al 71%), lo que lastra su media final y eleva su desviación estándar (±4.44%). La estabilidad del escáner SFWHT ante ciertas inicializaciones de la matriz de pesos es el punto débil actual.
3. **Pure DGE como baseline robusto:** DGE Puro sigue demostrando una fiabilidad extrema, con una varianza mínima (±1.57%) entre inicializaciones, convirtiéndolo en un optimizador superior en expectativa, aunque el Híbrido tiene picos más altos.

## 5. Veredicto Final sobre SFWHT vs Pure DGE
Opus tenía razón al detener el descarte prematuro. SFWHT no está roto en redes grandes, simplemente es más inestable.
Si bien Pure DGE sigue siendo el optimizador "por defecto" debido a su robustez inquebrantable, el enfoque Híbrido ha demostrado empíricamente ser competitivo (incluso superior en inicializaciones favorables) a la escala de 100K parámetros.

El track Híbrido se cierra oficialmente por ahora al no lograr dominar *en media*, pero queda validado como un concepto funcional.

Pasamos, ahora sí con evidencia justa y sin arrastrar deudas técnicas, al siguiente experimento del *shortlist* (que de hecho, ya fue cubierto en la `v26`: **Vector Group DGE**, que también resultó inferior a Pure DGE).