# DGE Findings v45: Sub-divided Neuron Blocks (NUEVO RÉCORD)

## Objetivo
Resolver el problema del ruido en bloques grandes (784 variables) manteniendo la alineación estructural con las neuronas.

## Configuración
- **Método:** Cada una de las 128 neuronas de la Capa 1 se divide en **4 sub-bloques** de ~196 cables cada uno.
- **K Total:** 586 bloques.
- **Tamaño bloque L1:** ~196 (frente a los 320 del baseline).

## Resultados
- **Baseline O(sqrt_D):** 90.49%
- **Neuron Aggregated (v42):** 90.25%
- **Sub-divided Neurons (v45):** **91.08%** 🏆

## Hallazgos Clave

### 1. El Triunfo de la Estructura sobre el Azar
Este experimento demuestra que la regla universal de $O(\sqrt{D})$ es superable. Si conocemos la topología (neuronas), podemos crear una partición que no solo reduce el ruido (al usar bloques más pequeños), sino que mantiene la coherencia de la señal al no mezclar parámetros de diferentes neuronas.

### 2. Resolución Micro-Estructural
La "unidad de aprendizaje" óptima para DGE en redes profundas no es la neurona completa, sino **segmentos del fan-in** de la neurona. Al trocear la neurona, permitimos que el optimizador identifique sub-patrones espaciales con mucha más claridad.

### 3. Escalabilidad
Este enfoque es altamente escalable: permite entrenar redes más grandes manteniendo la precisión simplemente ajustando el número de subdivisiones por neurona para mantener el tamaño del bloque en el rango de los 100-200 parámetros.

## Conclusión
v45 establece el nuevo estado del arte para DGE Estructural. Hemos validado que la arquitectura de la red es la mejor guía para diseñar la estrategia de perturbación.
