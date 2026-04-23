# DGE Findings v46: Canine Sniffing (Olfato Estéreo)

## Objetivo
Emular el sistema de rastreo de un sabueso mediante "barridos laterales" (movimientos ortogonales) para localizar el centro del rastro del gradiente.

## Configuración
- **Base:** DGE Estructural (v45: Sub-divided neurons).
- **Mecanismo:** Un paso de DGE estándar + 1 evaluación lateral (fosas nasales) en una dirección perpendicular al movimiento actual.
- **Corrección:** Se aplica un ajuste lateral proporcional a la diferencia de "olor" (Loss) detectada entre las dos fosas nasales.

## Resultados
- **Sub-divided Neurons (v45):** 91.08%
- **DGE-Canine (v46):** **90.94%**
- **Baseline O(sqrt_D):** 90.49%

## Hallazgos
1. **Superación del Baseline:** A pesar de la naturaleza aleatoria del barrido lateral, el método sigue superando al baseline universal, lo que demuestra la robustez de la arquitectura de neuronas.
2. **Ruido en Alta Dimensión:** En un espacio de 100k dimensiones, un barrido lateral aleatorio tiene pocas probabilidades de encontrar una mejora significativa si no está guiado por la historia del gradiente.
3. **Costo de Oportunidad:** Las 2 evaluaciones extra por paso reducen ligeramente el número de pasos totales, lo que explica la pequeña caída frente al v45.

## Conclusión
El olfato canino es una metáfora poderosa para el **ajuste de segundo orden**. Sin embargo, para que sea superior, el "olfateo" debe ser **selectivo** y no aleatorio.
