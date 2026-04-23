# DGE Findings v41 & v42: La Neurona como Unidad de Bloque

## Objetivo
Explorar la agrupación topológica pura. En lugar de bloques aleatorios, usamos el "Fan-In" de cada neurona como frontera del bloque.

## Configuración (v41)
- **Bloques:** Pesos de cada neurona y su bias en bloques SEPARADOS.
- **Resultado:** 87.90% (Fracaso relativo).
- **Razón:** Demasiados bloques (404) y mucho ruido en los bloques de pesos de tamaño 784.

## Configuración (v42) - Neuron Aggregated
- **Bloques:** Pesos + Bias de la misma neurona en el MISMO bloque.
- **Resultado:** **90.25%** (Éxito).
- **K Total:** 202 bloques (uno por neurona).

## Hallazgos
1. **Coherencia Topológica:** Agrupar bias y pesos de la misma neurona es superior a separarlos. Al afectar a la misma pre-activación, el optimizador "entiende" mejor el impacto de la neurona como un todo.
2. **Eficiencia de Paso:** Con solo 404 evals/paso, v42 es el método más ligero probado hasta la fecha que alcanza el 90%. 
3. **Limitación de Ruido:** El bloque de tamaño 784 (capa 1) es el cuello de botella. Aunque la estructura es buena, el ruido estadístico de 784 variables es demasiado alto para superar al baseline aleatorio.
