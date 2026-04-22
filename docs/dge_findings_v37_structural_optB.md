# DGE Findings v37: Structural DGE (Opción B - Random Signs Fan-In)

## Objetivo del Experimento
Tras el fallo de la Opción A (Escalar Fijo) por promediar excesivamente los gradientes, la Opción B corrige el defecto matemático utilizando **Signos Aleatorios** dentro de cada bloque.
Además, se optimiza la topología utilizando **exclusivamente bloques Fan-In**. Puesto que cada peso en la red llega exactamente a una neurona de destino, agrupar los pesos por neurona destino cubre el 100% de la red sin solapamientos. 

Esto reduce drásticamente el número de bloques: en una arquitectura `(784, 128, 64, 10)`, el número total de bloques Fan-In es exactamente el número de neuronas ocultas y de salida ($128 + 64 + 10 = 202$ bloques). Esto requiere tan solo **404 evaluaciones por paso** (el doble de rápido que la técnica `O(sqrt_D)` en cantidad de evaluaciones).

## El Triunfo Teórico
Al usar signos aleatorios $S_i$ en un bloque Fan-In, la esperanza matemática del gradiente estimado para el peso $i$ se convierte en la derivada de la pre-activación multiplicada por la entrada individual:
$$ E[\nabla W_{i}] = \frac{\partial L}{\partial z} \times x_i $$
¡Esto es **matemáticamente idéntico al gradiente analítico de Backpropagation**! A diferencia de los bloques aleatorios tradicionales de DGE, esta topología aisla el error a nivel de nodo (Node Perturbation) y extrae de ahí los gradientes de todos los pesos que conectan a él.

## Resultados Empíricos
A pesar de la elegancia matemática, el rendimiento práctico sufre problemas de lentitud severa en la convergencia temprana:
- **Test Acc:** Converge lenta y firmemente (el usuario reporta llegar a un ~40% en pruebas posteriores), pero muy por detrás del ~90.49% de la Baseline de bloques aleatorios estandarizados (`O(sqrt_D)`).

## Análisis de la Lentitud: El Monstruo del Fan-In
El problema radica en la distribución de la varianza debido a la arquitectura de la red:
1. **El Cuello de Botella de la Capa 1:** En la primera capa, cada neurona recibe **784 conexiones** de entrada. En nuestro diseño Fan-In puro, esto significa que el algoritmo debe estimar el gradiente de 784 variables simultáneas utilizando solo 2 evaluaciones perturbadas.
2. **Varianza vs Precisión:** La teoría de perturbación estocástica estipula que la varianza del estimador de gradiente es proporcional al número de variables perturbadas ($D$). Un bloque de tamaño 784 genera un "ruido cruzado" inmenso. La esperanza matemática converge al gradiente correcto de Backpropagation a largo plazo, pero la alta varianza en cada paso requiere miles de iteraciones extra para estabilizarse.
3. **El Equilibrio del $O(\sqrt{D})$:** La técnica de la raíz cuadrada (v35) partía la primera capa en 316 bloques de ~318 pesos. Reducía el tamaño del bloque a la mitad (318 vs 784), lo que disminuía el ruido dramáticamente y permitía pasos más estables a expensas de requerir el doble de evaluaciones (862 vs 404).

## Conclusión
La Node Perturbation a nivel de Fan-In es teóricamente la formulación de orden cero más pura y cercana a Backpropagation posible. Sin embargo, su rendimiento práctico se ve obstaculizado en la primera capa debido al tamaño masivo de las imágenes (784 píxeles), lo que dispara el ruido de estimación en esos bloques iniciales. La idea no se debe abandonar, pero requeriría técnicas avanzadas de reducción de varianza (control variates, ortogonalización) para que los bloques de tamaño 784 converjan rápido.
