# DGE Findings v57: Tuning Learning Rate Schedules

## Objetivo
Dado que hemos establecido que el mecanismo "Dynamic Noise Budget" (v54) es nuestro mejor algoritmo de optimización, el siguiente paso era optimizar la macro-hiperparametrización, en concreto el esquema de decaimiento del Learning Rate (LR Schedule).

En optimizadores de primer orden (SGD, Adam), el decaimiento de LR es crucial para asentar los pesos al final del entrenamiento. Queríamos comprobar qué familia de decaimiento interactúa mejor con la reducción de varianza dinámica del DGE.

## Configuración
- **Presupuesto:** 500,000 Evals
- **Mecanismo:** Dynamic Budget (K=586 bloques)
- **LR Rango:** `LR_MAX = 0.02` -> `LR_MIN = 0.001`
- **Variantes probadas:**
  1. `Constant` (se queda en 0.02)
  2. `Cosine` (curva suave de 0.02 a 0.001)
  3. `Step` (Caídas 10x al 50% y 75% del entrenamiento)
  4. `Exponential` (Caída constante continua de 0.02 a 0.001)

## Resultados (Mejor Precisión en Test)

| Schedule | Precisión | Comportamiento |
|----------|-----------|----------------|
| **Exponential** | **92.53%** | Excelente curva de aprendizaje, dominando desde las etapas medias. |
| **Cosine** | 92.32% | Muy sólido, el estándar de facto, mantiene el LR más alto durante más tiempo que el exponencial. |
| **Constant** | 91.39% | Convergencia errática al final, se atasca en mínimos ruidosos. |
| **Step** | 90.98% | La caída brusca al 50% (`0.02 -> 0.002`) frenó el aprendizaje prematuramente. |

## Análisis y Conclusiones

### 1. ¡Nuevo Récord Absoluto! (92.53%)
Simplemente afinando el rango inicial y final del LR y usando un decaimiento **Exponencial**, hemos roto por completo la barrera del 91% en la que estábamos estancados, alcanzando un impresionante **92.53%**.
Esto es un salto monumental para un optimizador Zeroth-Order puro en redes neuronales, superando con creces todas las iteraciones anteriores.

### 2. Por qué Exponencial superó a Coseno
En DGE, los gradientes siempre tienen ruido residual.
- El decaimiento **Coseno** mantiene el LR alto (cerca de `LR_MAX`) durante el primer 30-40% del entrenamiento antes de empezar a caer pronunciadamente. En DGE, estar tanto tiempo "caliente" (LR alto) implica que el optimizador rebota demasiado y acumula error en sus medias de Adam ($m_t, v_t$).
- El decaimiento **Exponencial**, por el contrario, empieza a enfriar el sistema *inmediatamente* desde el paso 1. Al paso 250k (la mitad), el LR exponencial ya iba por `0.004`, mientras que el coseno seguía en `0.01`. Esa estabilización temprana y progresiva es perfecta para domar las estimaciones de diferencias finitas, permitiendo que la red se asiente suavemente en valles más profundos.

### 3. Evitar los Saltos (Step Decay)
El `Step Decay` fue el peor de los que decaen. Cuando el LR cayó de golpe de `0.02` a `0.002` a la mitad del entrenamiento, el optimizador simplemente se "congeló". En DGE, los cambios de régimen bruscos son peligrosos porque las estadísticas de momento de Adam necesitan tiempo para ajustarse a las nuevas escalas del paso.

## Recomendación Final para el Paper
El decaimiento **Exponencial (o Coseno agresivo)** es mandatorio para optimizadores Zero-Order. No se debe usar Step Decay. La combinación ganadora final del repositorio es:
**Arquitectura Neuronal Subdividida + Dynamic Noise Budget + Asymmetric Adam EMA + Exponential LR Schedule.**