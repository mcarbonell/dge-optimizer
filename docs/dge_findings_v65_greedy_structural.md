# DGE Findings v65: Greedy Alternating Node Perturbation

## Objetivo del Experimento
Tras descubrir en la versión `v64` que el algoritmo Adam (EMA) se confundía al mezclar gradientes de naturaleza aditiva y multiplicativa, planteamos una alternativa radical: **Eliminar el algoritmo de momentum y usar un enfoque puramente "Greedy" (Búsqueda Local).**

En este experimento no hay memoria temporal (EMA) ni tasa de aprendizaje (Learning Rate). La red se actualiza de forma **inmediata** si (y solo si) una perturbación específica en un bloque estructural mejora la pérdida del batch actual.

Se alternan *sweeps* (barridos completos de toda la red) aditivos y multiplicativos.

## Resultados Empíricos (2.5M Evals)
- **v65 (Greedy Alternating):** 64.77% 
- **Tiempo de ejecución:** ~72 minutos (Extremadamente lento).

## Análisis de la Regresión y Rendimiento
El experimento logró aprender de manera constante y estable sin atascarse, validando que el enfoque *Greedy* no sufre de las incompatibilidades matemáticas del EMA vistas en el `v64`. Sin embargo, el desempeño final (64.77%) y el tiempo de ejecución (1 hora y 12 minutos) son notablemente peores que la versión multiplicativa pura (78.63% en 12 minutos).

### 1. Ineficiencia Computacional (Lentitud)
La principal causa de la lentitud es que el enfoque *Greedy* requiere una evaluación secuencial bloque por bloque.
Para evaluar si la perturbación del Bloque $i$ mejora la red, debemos recalcular el Loss. Al actualizar la red inmediatamente, el cálculo del Loss del Bloque $i+1$ depende estrictamente del resultado del Bloque $i$.
Esto **destruye por completo la capacidad de paralelización y vectorización en la GPU**. En lugar de evaluar 2000 bloques a la vez en una mega-matriz (como hace DGE), estamos haciendo 2000 llamadas individuales a PyTorch para *forward passes* minúsculos en bucle, lo cual asfixia a la CPU enviando micro-comandos a DirectML.

### 2. El Problema de "No ser lo suficientemente Greedy"
Aunque el algoritmo acepta inmediatamente cualquier perturbación que reduzca el error, solo estamos probando dos direcciones ($+\Delta$ y $-\Delta$) fijas y minúsculas por bloque en cada *sweep*. 
Si la dirección óptima de un cable es aumentar su valor en $0.5$, el algoritmo tardará cientos de *sweeps* en llegar ahí avanzando de $0.0001$ en $0.0001$, desperdiciando millones de evaluaciones validando pasos ridículamente pequeños. 
El EMA de Adam resolvía esto acelerando (momentum) en direcciones consistentemente buenas, pero al quitar Adam, nos hemos quedado sin mecanismo de aceleración.

## Conclusión
La Búsqueda Local (Greedy) estructural funciona teóricamente y evita problemas de mezcla de señales, pero es **computacionalmente inviable en hardware moderno (GPUs)** debido a su dependencia secuencial obligatoria. DGE es poderoso precisamente porque evalúa cientos de perturbaciones en paralelo y las agrega; el enfoque *Greedy* puro rompe esa ventaja fundamental.
