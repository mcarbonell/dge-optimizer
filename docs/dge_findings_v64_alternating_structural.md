# DGE Findings v64: Alternating Node Perturbation (Additive + Multiplicative)

## Objetivo del Experimento
Tras validar en el experimento `v63` que la perturbación estructural *multiplicativa* (78.63%) resolvía en gran medida el problema del "Lavado de Gradiente" visto en la variante *aditiva* `v36` (40.16%), propusimos una estrategia híbrida novedosa.

La **hipótesis** era alternar la estrategia en cada paso:
- **Paso Par (Aditivo):** Rompe la simetría y "despierta" a los pesos muertos (valor $0.0$) que la multiplicación no puede alterar.
- **Paso Impar (Multiplicativo):** Realiza un ajuste fino y estable preservando la magnitud relativa de las conexiones importantes.

## Resultados Empíricos (2.5M Evals)
- **v36 (Puro Aditivo):** 40.16%
- **v63 (Puro Multiplicativo):** 78.63%
- **v64 (Alternante):** **67.49%**

## Análisis de la Regresión (67.49%)
La idea, aunque elegante en teoría, resultó ser contraproducente en la práctica, situando el rendimiento exactamente a medio camino entre las dos estrategias puras. 

¿Por qué falló el modo alternante?
1. **Incompatibilidad del EMA (Adam):** El optimizador acumula el gradiente en las variables de momento ($m_t, v_t$). El gradiente aditivo es matemáticamente de una naturaleza distinta al gradiente multiplicativo (este último está escalado por el peso $x$). Al inyectar gradientes de ambas naturalezas de forma intercalada en el mismo acumulador temporal, estamos sumando "peras con manzanas", lo que destruye la señal coherente de la dirección de descenso.
2. **Lavado intermitente:** En el 50% de los pasos (los aditivos), la red sigue sufriendo el destructivo "Gradient Washing". Aunque intente arreglarlo en el paso siguiente, el daño ya se ha inyectado en el momento de Adam.
3. **El problema de los pesos muertos está sobreestimado:** En una red inicializada con distribución normal (Gaussian), es estadísticamente casi imposible que un peso caiga exactamente a $0.0$ absoluto en un entorno de precisión de 32-bits durante un proceso estocástico. Por tanto, el beneficio teórico de la adición para "resucitar" pesos muertos no compensa la inyección de ruido masivo que conlleva.

## Conclusión
La mezcla ingenua de señales de diferente naturaleza matemática en un estimador de momento de primer orden (EMA) degrada el aprendizaje. La perturbación multiplicativa pura (`v63`) sigue siendo la SOTA indiscutible para la optimización estructural (Node Perturbation) sin Backpropagation. 
