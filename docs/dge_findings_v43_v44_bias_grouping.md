# DGE Findings v43 & v44: Agrupación de Biases por Jerarquía

## Objetivo
Determinar si los biases se benefician de estar agrupados entre ellos para reducir el ruido, separándolos de los cables.

## Resultados
- **v43 (1 Bloque Global para todos los biases):** 90.02%
- **v44 (3 Bloques, uno por capa):** 89.58%

## Hallazgos
1. **Denoising Efectivo:** Tener un solo bloque para los 202 biases (v43) funciona mejor que separarlos por capas. La señal conjunta del bias es lo suficientemente fuerte para guiar la red.
2. **Superioridad de la Neurona:** A pesar del buen resultado de v43, sigue siendo inferior a la agrupación por neurona (v42). 
3. **Lección:** La relación espacial (peso-bias de la misma neurona) es más importante para el gradiente que la relación funcional (bias con otros biases).
