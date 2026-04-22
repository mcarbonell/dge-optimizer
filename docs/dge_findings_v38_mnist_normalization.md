# DGE Findings v38: Impacto de la Normalización de Entrada (MNIST)

## Objetivo del Experimento
Validar la intuición de si la normalización estándar de MNIST (Media 0, Varianza 1) es realmente necesaria para la convergencia o si una escala simple de `[0, 1]` es suficiente. Se comparó el rendimiento utilizando el optimizador Adam sobre un MLP estándar `(784, 128, 64, 10)`.

## Configuración
- **Datasets:** MNIST (60k entrenamiento, 10k test).
- **Optimizador:** Adam (LR=0.001).
- **Épocas:** 5.
- **Semillas:** 5 (para significancia estadística).
- **Hardware:** AMD Ryzen 7 8845HS + Radeon 780M.

## Resultados Empíricos (Media ± Std)
| Configuración | Test Accuracy | Tiempo Total (5 épocas) | Acc Época 1 |
| :--- | :--- | :--- | :--- |
| **Media 0 (Std)** | 97.43% ± 0.16 | 33.64s | ~95.0% |
| **Solo [0, 1]** | 97.25% ± 0.11 | **20.14s** | ~93.5% |

## Hallazgos Clave

### 1. Robustez de Adam
El experimento demuestra que los optimizadores modernos como Adam son extremadamente robustos a la escala de la entrada. Aunque la normalización a media cero es la "buena práctica" teórica, la diferencia final en precisión es de apenas un **0.18%**, lo cual es despreciable para la mayoría de aplicaciones.

### 2. Velocidad de Convergencia Matemática
La normalización a media cero cumple su promesa teórica: **arranca más rápido**. En la primera época, la red con media cero ya ha alcanzado el 95%, mientras que la versión `[0, 1]` se queda en el 93.5%. Los datos centrados facilitan que los gradientes iniciales encuentren direcciones de descenso productivas de inmediato.

### 3. El "Impuesto" de la CPU (Overhead de Transformación)
El hallazgo más sorprendente es la diferencia en **tiempo real (Wall-clock time)**. La versión `[0, 1]` es un **40% más rápida** en completarse. 
- **Causa:** La función `transforms.Normalize` de PyTorch se ejecuta en la CPU. Aplicar una resta y una división a cada uno de los 784 píxeles de cada imagen en cada época genera un cuello de botella que "estrangula" el flujo de datos hacia la GPU.
- **Lección:** En entornos de alto rendimiento, un pre-procesado matemático "perfecto" puede no compensar si introduce latencia en el pipeline de datos.

## Conclusión para DGE
Para nuestros experimentos de optimización de Orden Cero (DGE), seguiremos usando **Media 0**. Dado que DGE es inherentemente más ruidoso que Adam, necesitamos darle a la matemática todas las facilidades posibles (datos centrados) para que las perturbaciones aleatorias encuentren señal útil lo antes posible, incluso a costa de un ligero overhead en la CPU.
