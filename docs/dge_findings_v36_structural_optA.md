# DGE Findings v36: Structural DGE (Opción A - Escalar Fijo)

## Objetivo del Experimento
La "Node Perturbation" (Perturbación de Nodos) es un concepto clásico en redes neuronales de hardware (VLSI) donde se perturba la activación completa de una neurona en lugar de sus pesos individuales. 
En `v36` intentamos emular esto agrupando topológicamente los pesos de la red en **Bloques Fan-In** (todas las conexiones que llegan a una neurona) y **Bloques Fan-Out** (todas las que salen de ella). 

La **Opción A** consistía en aplicar una perturbación escalar fija (ej. sumar `+delta` a *todos* los parámetros de un bloque a la vez) y usar la intersección topológica (un peso recibe el gradiente de su Fan-In + su Fan-Out) para actualizar la red de forma similar a un algoritmo "Low-Rank".

## Resultados Empíricos
El experimento fue un fracaso crítico en términos de convergencia de aprendizaje:
- **Test Acc (500k evals):** ~16% (Prácticamente conjetura aleatoria en MNIST).
- La red quedó totalmente "asfixiada" y fue incapaz de aprender características de los datos.

## Análisis del Fallo: "Gradient Washing" (Lavado de Gradiente)
El fracaso se debe a una incompatibilidad matemática fundamental con la regla de aprendizaje de redes neuronales:
1. Al sumar exactamente el mismo `+delta` a todas las entradas de una neurona, la pre-activación de la neurona se perturba en proporción a la **suma de todas sus entradas** ($\Delta \sum x_i$).
2. El gradiente estimado para el bloque entero termina siendo proporcional al error multiplicado por esa suma global ($\frac{\partial L}{\partial z} \sum x_i$).
3. Al asignar este mismo escalar de vuelta a todos los pesos del bloque, forzamos a que el peso $W_i$ reciba un gradiente basado en la suma de todos los píxeles, **destruyendo por completo la información individual del píxel $x_i$**. 
4. La regla matemática real (Backpropagation) exige que la actualización sea individual: $\Delta W_i \propto \text{Error} \times x_i$. Al promediar todo, destruimos la capacidad de la red para distinguir entre píxeles importantes y ruido.

## Conclusión
Aplicar un escalar uniforme fijo a un bloque estructural es matemáticamente inviable para la optimización directa de pesos porque promedia las contribuciones individuales de las entradas. La estructura topológica es una idea excelente, pero requiere una inyección de ruido que preserve la ortogonalidad de las entradas (como signos aleatorios).
