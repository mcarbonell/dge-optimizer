# DGE Findings v39: Inicialización Constante por Neurona y Normalización

## Objetivo del Experimento
Investigar la capacidad de la red para recuperarse de una inicialización extremadamente restringida: **todos los cables (pesos) que entran a una misma neurona comienzan con el mismo valor**. 

Este experimento está directamente relacionado con la **Optimización Estructural (v37)**, ya que busca entender si forzar una estructura de bloque (todos los pesos de la neurona moviéndose o empezando igual) es una limitación insuperable o simplemente un punto de partida lento.

## Resultados Empíricos (5 Semillas, 5 Épocas)

| Inicialización Capa 1 | Normalización Entrada | Test Accuracy (Media) | Tiempo Total |
| :--- | :--- | :--- | :--- |
| **Constante Global (v36)** | [0, 1] | ~88.69% | ~80s |
| **Constante por Neurona** | [0, 1] | ~91.08% | ~110s |
| **Constante por Neurona** | **Media-Cero (Std)** | **93.69%** | **~188s** |

*Nota: La inicialización aleatoria estándar alcanza ~97.4% en las mismas condiciones.*

## Hallazgos Clave

### 1. Ruptura de la Simetría (Inter-Neuron vs Intra-Neuron)
- **Constante Global:** Si todas las neuronas de la capa son idénticas, la red colapsa a la potencia de una sola neurona (~88%). No hay diversidad.
- **Constante por Neurona:** Al dar a cada neurona un valor inicial distinto, recuperamos la diversidad entre neuronas. Aunque internamente cada neurona sea "ciega" espacialmente al principio, la red puede aprender (~91-93%).

### 2. El Efecto "Desbloqueador" de la Media-Cero
La normalización a media cero (con valores negativos) actúa como un catalizador. Permite que incluso los pesos negativos sean útiles desde el primer milisegundo (al multiplicarse por píxeles de fondo negativos). Esto redujo el Loss de la primera época de **1.25 a 0.75** y subió la precisión final en un **2.5%**.

### 3. El Coste del Pipeline (CPU Tax)
Se confirma de forma robusta que la normalización en CPU (`transforms.Normalize`) introduce un overhead de tiempo de casi el **70%** en este hardware. Para experimentos rápidos de DGE, este es un factor crítico a tener en cuenta.

## Relación con DGE Estructural (Node Perturbation)
Estos experimentos demuestran que, aunque forcemos a los pesos de una neurona a empezar (o incluso a actualizarse) como un bloque, la red conserva una gran capacidad de aprendizaje si el optimizador puede eventualmente "romper" esa estructura. La **Node Perturbation** es un excelente estimador de gradiente macro, pero este estudio sugiere que permitir que cada cable "afine" su valor individualmente después del paso estructural (como hace Adam en este test) es lo que lleva a la red al máximo rendimiento.
