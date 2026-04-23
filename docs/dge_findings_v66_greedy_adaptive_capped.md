# DGE Findings v66: Deltas Adaptativos Capped (Estrategias Evolutivas)

## Objetivo del Experimento
Tras implementar la Regla del 1/5 de Rechenberg en el experimento `v66`, la red demostró una aceleración inicial sin precedentes en la historia del repositorio, saltando del 38% al **77% de precisión en tan solo 200,000 evaluaciones** (menos de un minuto de ejecución en CPU).
Sin embargo, después de este arranque meteórico, la red sufrió un severo problema de sobreajuste por lotes (batch overfitting) y comenzó a dar tumbos entre el 74% y el 79%.

### Análisis de los Tumbos (v66 Original)
El problema radica en la falta de un "techo" para el multiplicador adaptativo:
1. El algoritmo evalúa el éxito basándose en un mini-batch aleatorio de 256 imágenes.
2. Si un bloque encuentra una perturbación que tiene "suerte" en esas 256 imágenes específicas, el algoritmo premia al bloque multiplicando su $\Delta$ por `1.2`.
3. A lo largo de miles de *sweeps*, los bloques "suertudos" inflaron su tamaño de paso hasta alcanzar el máximo permitido en el código original (`0.5`). 
4. Un $\Delta$ multiplicativo de `0.5` significa que el bloque intentaba alterar todos sus pesos un 50% de golpe. Estos pasos de gigante destruían las características aprendidas para las otras 59,744 imágenes del dataset, causando caídas drásticas en el Test Accuracy.

### La Solución: Techos de Cristal (v66b)
Se diseñó la versión `v66b` (*Capped Adaptive*) para mantener la aceleración meteórica inicial pero previniendo la auto-destrucción posterior. Se aplicaron los siguientes límites estrictos a la recompensa biológica:
- **`max_delta_add = 0.01`**: Un cable no puede sumar más de un 1% absoluto en un solo paso.
- **`max_delta_mul = 0.05`**: Un cable no puede escalar su valor más de un 5% relativo en un solo paso.

De esta forma, cuando los deltas intenten volverse destructivos chocarán con el límite y se verán obligados a afinar la puntería, estabilizando el vuelo.

## Resultados Empíricos (2.5M Evals)
- **v66 (Adaptive Sin Límites):** ~79.89% (con alta varianza y tumbos).
- **v66b (Adaptive Capped):** *(Pendiente de ejecución por el usuario)*

## Siguientes Pasos
Esperando la ejecución de la versión `v66b` puramente en CPU para validar si la restricción de agresividad (Capping) permite estabilizar la curva de aprendizaje por encima de la barrera del 80% en un régimen *Greedy* estricto sin memoria.