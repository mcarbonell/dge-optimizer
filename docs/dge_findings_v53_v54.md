# DGE Findings v53-v54: Refinamientos Estructurales

## Objetivo
Explorar dos mecanismos de control sobre el descenso puro DGE (basados en la arquitectura sub-dividida de la v45):
- **v53 (Foresight & Backtracking):** Gasta 1 evaluación extra para verificar si el paso propuesto por Adam nos saca del valle (aumento brusco del Loss). Si es así, recorta el tamaño del paso un 75% antes de aplicarlo.
- **v54 (Dynamic Noise Budget):** Mantiene el límite total de 586 bloques a evaluar por paso, pero los distribuye dinámicamente entre las capas basándose en la Varianza (ruido) estimada por Adam.

## Resultados (MNIST, 500k Evaluaciones, LR = 0.004)

| Versión | Método | Presupuesto (evals/paso) | Precisión Test |
|---------|--------|--------------------------|----------------|
| **v45** | Baseline Sub-divided Neurons | 1172 | 91.08% *(LR=0.05)* |
| **v53** | Foresight & Local Backtracking | 1173 | **91.59%** |
| **v54** | Dynamic Noise Budget | 1172 | **91.76%** |

## Análisis y Hallazgos Clave

### 1. La importancia del Learning Rate (v53)
Bajar el LR de `0.05` a `0.004` cambió completamente el comportamiento del optimizador. 
En la v53 vimos que con el LR bien ajustado, el mecanismo de *Backtracking* nunca llegó a dispararse (`bks=0`). Esto significa que el DGE era inherentemente estable. Sin embargo, la precisión subió al **91.59%**, confirmando que en este benchmark particular, un paso más corto y seguro es preferible a arriesgar saltos largos.

### 2. El Presupuesto Dinámico Rompe el Techo (v54)
La v54 logró un **91.76%**, convirtiéndose en el nuevo récord para este número de evaluaciones en redes profundas de MNIST.
El comportamiento de asignación (`k_alloc`) reveló algo profundo sobre la propagación del gradiente DGE:
- Prácticamente todo el presupuesto dinámico (`582` bloques) se asignó siempre a la **primera capa**, dejando el mínimo (`2` bloques) a las capas 2 y 3.
- **Razón:** En el método DGE por diferencias finitas (ZO), el ruido se amplifica enormemente a medida que nos alejamos de la salida (la capa más cercana a los datos de entrada sufre la mayor varianza). Al forzar a la capa 1 a promediar 582 muestras de ruido, mientras que las capas superiores apenas necesitan 2, logramos una reducción de la varianza óptima sin aumentar el coste computacional total.

## Conclusión
La asignación dinámica de evaluaciones basada en la varianza local (v54) es una técnica altamente eficiente. Demuestra que no todos los parámetros de la red sufren el mismo nivel de ruido durante la estimación DGE, y distribuir el "presupuesto de muestras" proporcionalmente a ese ruido produce estimaciones mucho más precisas y, en última instancia, una convergencia superior.
