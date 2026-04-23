# DGE Findings v40: Bloques Específicos para Biases

## Objetivo
Evaluar si darle a los biases (parámetros de alta sensibilidad) bloques de tamaño 1 (sin ruido) mejora la convergencia general.

## Configuración
- **Baseline:** Bloques aleatorios $O(\sqrt{D})$ mezclando pesos y biases.
- **Bias-Specific:** 202 bloques de tamaño 1 (biases) + 330 bloques aleatorios para pesos.
- **Budget:** 500k evals.

## Resultados
- **Baseline:** 90.49%
- **Bias-Specific:** 89.59%

## Hallazgos
1. **Sacrificio de Presupuesto:** Aunque los biases tienen señal perfecta, el aumento del número de bloques (K=633 vs K=431) reduce el número de pasos de Adam.
2. **Prioridad de los Pesos:** La señal de los biases, por muy limpia que sea, no puede compensar la pérdida de frecuencia de actualización en los cientos de miles de cables (pesos).
3. **Conclusión:** Aislar variables individuales no es rentable si eso dispara el número de evaluaciones por paso.
