# DGE Findings v58-v60: Redes Híbridas (Capa Binaria + Capas Continuas)

## Objetivo
Explorar la viabilidad de entrenar una red neuronal híbrida donde la **primera capa está estrictamente binarizada a valores {0, 1}** (actuando como un selector rígido de píxeles), mientras que el resto de las capas (ocultas a salida) mantienen pesos continuos flotantes.

Este experimento pone a prueba la capacidad de DGE para manejar paisajes de pérdida mixtos, donde una parte de los parámetros opera en un espacio discreto (con gradientes nulos casi en todas partes) y otra en un espacio continuo.

## Iteraciones y Problemas Encontrados

### v58: El Colapso del Presupuesto Dinámico
- **Configuración:** Arquitectura híbrida usando el optimizador "Megazord" (v55) con *Dynamic Noise Budget*.
- **Problema:** La precisión se estancó en ~31%.
- **Diagnóstico:** La binarización se logra mediante la función escalón $W = (W_{latente} > 0)$. Al evaluar perturbaciones, la gran mayoría no cruza el umbral $0$, resultando en una varianza nula. Sin embargo, las pocas perturbaciones que sí lo cruzan generan un cambio discreto brusco en la salida. Al calcular el gradiente dividiendo por un $\Delta$ pequeño ($10^{-3}$), la varianza estimada de la capa 1 se disparaba astronómicamente. El *Dynamic Budget* reaccionó robando casi todos los bloques de evaluación para la capa 1 (ej. `[582, 2, 2]`), "matando de hambre" a las capas continuas e impidiendo el aprendizaje profundo.

### v59: La Destrucción por Delta Global
- **Configuración:** Presupuesto de bloques fijo (`[512, 64, 10]`) para garantizar evaluaciones en todas las capas. Incremento global de $\Delta = 0.05$ y $LR = 0.1$ para forzar más cruces del umbral binario.
- **Problema:** La red colapsó al 10.32% (adivinanza aleatoria).
- **Diagnóstico:** Si bien un $\Delta$ gigante de $0.05$ ayudaba a la capa binaria a explorar, ese mismo radio de perturbación aplicado a las capas continuas destrozaba por completo la aproximación de diferencias finitas. Las estimaciones de gradiente de las capas profundas eran puro ruido.

### v60: La Solución (Hiperparámetros Desacoplados)
- **Configuración:** Se reescribió el optimizador para soportar listas de hiperparámetros independientes por capa:
  - **Capa 1 (Binaria):** $\Delta$ inmenso ($0.05$) y LR alto ($0.1$) para explorar y cruzar el umbral discreto.
  - **Capas 2 y 3 (Continuas):** $\Delta$ fino ($10^{-3}$) y LR bajo ($0.005$) para una estimación de gradientes precisa.
- **Resultado:** La red convergió de manera estable, alcanzando un **51.44%** de precisión en el presupuesto asignado (500k evals).

## Conclusiones

1. **Las Redes Híbridas son Posibles:** DGE es capaz de optimizar simultáneamente parámetros discretos y continuos, un régimen donde Backpropagation tradicional fracasaría masivamente sin trucos complejos como el *Straight-Through Estimator*.
2. **Separación de Escalas de Ruido:** La lección más crítica de este experimento es que **los espacios discretos y continuos requieren regímenes de perturbación incompatibles**. No se puede usar un $\Delta$ global. Los parámetros discretos necesitan "martillazos" (alto $\Delta$) para registrar señal, mientras que los continuos necesitan "bisturís" (bajo $\Delta$).
3. **Desempeño:** Aunque un 51% es un logro técnico notable para una red con una primera capa puramente binaria entrenada *zeroth-order*, queda lejos del >90% de las redes totalmente continuas. Restringir la primera capa a $\{0, 1\}$ impone un cuello de botella informacional severo (funciona como una máscara de recorte binaria de la imagen), lo que limita la capacidad expresiva de la red.

## Siguientes Pasos
Este descubrimiento sobre la necesidad de **hiperparámetros desacoplados por capa** es una mejora estructural que podríamos aplicar a nuestras redes continuas normales. Es posible que las capas iniciales y finales de una red profunda continua también se beneficien de tener radios de perturbación $\Delta$ ligeramente diferentes adaptados a la magnitud típica de sus pesos y activaciones.
