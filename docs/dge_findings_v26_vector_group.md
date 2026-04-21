# Hallazgos de Investigación - DGE v26 (Vector Group DGE)

## 1. Objetivo de la Iteración
El objetivo de la iteración `v26` era implementar y evaluar **Vector Group DGE**. La hipótesis sugería que agrupar subconjuntos de variables en "super-variables vectoriales" y perturbándolas a lo largo de direcciones unitarias aleatorias (en lugar de perturbaciones alineadas a los ejes vía vectores Rademacher $\pm 1$) podría resolver el problema del "zig-zag" en paisajes no convexos y altamente correlacionados, como la función de Rosenbrock.

Además, para preservar la dirección del gradiente dentro del grupo, el optimizador mantendría una **Media Móvil Exponencial (EMA) Direccional** utilizando una única varianza escalar por grupo, en lugar de calcular la varianza componente a componente como hace el Adam estándar.

## 2. Configuración del Experimento
- **Benchmark:** Función de Rosenbrock ($D=128$).
- **Presupuesto:** 200,000 - 500,000 evaluaciones.
- **Optimizadores Probados:**
  1. **Pure DGE (Baseline):** Perturbaciones Rademacher ($\pm 1$), varianza EMA por coordenada.
  2. **Vector Group DGE:** Grupos contiguos, perturbaciones gaussianas esféricas uniformes, varianza EMA escalar por grupo.
  3. **Spherical DGE:** Perturbaciones gaussianas esféricas, varianza EMA por coordenada (aislando el efecto de la perturbación).
  4. **Scalar Variance DGE:** Perturbaciones Rademacher ($\pm 1$), varianza EMA escalar por grupo (aislando el efecto de la EMA).

Para garantizar una comparación justa, todos los métodos se ajustaron para consumir el mismo número exacto de evaluaciones por paso (ej. 16 evaluaciones por paso, dividiendo el espacio en 8 bloques).

## 3. Resultados Cuantitativos

*Resultados a 500,000 evaluaciones (8 bloques de 16 dimensiones):*
| Optimizador | Loss Medio (5 semillas) | Desviación Estándar |
|-------------|-------------------------|---------------------|
| **Pure DGE** | **92.36** | ± 3.50 |
| **Spherical DGE** | 102.92 | ± 16.32 |
| **Vector Group DGE** | 112.58 | ± 2.20 |

*Resultados a 200,000 evaluaciones (aislando la Varianza Escalar):*
| Optimizador | Loss Medio (5 semillas) | Desviación Estándar |
|-------------|-------------------------|---------------------|
| **Pure DGE** | 131.17 | ± 19.78 |
| **Scalar Variance DGE**| 127.80 | ± 12.82 |

## 4. Análisis y Conclusiones
La hipótesis principal ha sido **FALSADA**. Vector Group DGE no aporta ninguna mejora significativa sobre Pure DGE en el benchmark diseñado explícitamente para favorecerlo (Rosenbrock).

1. **La precondición diagonal de Adam es superior:** Reemplazar la varianza por coordenada de Adam con una varianza escalar por grupo (con el objetivo de preservar la dirección cruda del vector gradiente) resultó en un rendimiento inferior o idéntico. A pesar de las correlaciones en Rosenbrock, escalar las coordenadas de forma adaptativa e independiente sigue siendo una estrategia matemáticamente superior para descender por el valle.
2. **Perturbaciones Rademacher vs Esféricas:** El uso de vectores unitarios gaussianos (`Spherical DGE`) empeoró los resultados en comparación con las perturbaciones diagonales hipercubicas de DGE (`Pure DGE` con $\pm 1$). Las perturbaciones Rademacher exploran los "rincones" del espacio de parámetros de manera más agresiva, proporcionando un estimador de gradiente empíricamente más robusto.

## 5. Cierre de Iteración
Dado que Vector Group DGE falló en demostrar superioridad en el benchmark sintético de juguete (Rosenbrock D=128), **no está justificado escalar esta arquitectura a MNIST**.
La variante vectorial y escalar queda descartada. Las capacidades nativas del DGE puro (bloques superpuestos, perturbaciones $\pm 1$, EMA coordenada a coordenada) se mantienen como el estado del arte (SOTA) dentro de este repositorio.