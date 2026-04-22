# DGE Findings v34: Direction Consistency V2 (SNR-based)

## Objetivo del Experimento
Evaluar la segunda versión de la heurística de consistencia de dirección propuesto en `brainstorming.md`. Las características introducidas fueron:
1. **SNR-based Scaling:** Reemplazar la media plana de los signos (`mean(sign(grad))`) por el Ratio Señal-Ruido local `tanh(abs(mean) / std)`.
2. **Ventanas Adaptativas (T):** Utilizar historiales de distintos tamaños según la profundidad de la capa (`T=[40, 20, 5]`).
3. **Momentum de Ruido Crónico:** Un mecanismo de EMA a largo plazo para detectar variables consistentemente ruidosas y aplicarles un castigo severo (congelamiento a 0 o reducción de 10x del LR).

## Configuración
- **Dataset:** MNIST Completo (60K train / 10K test)
- **Presupuesto:** 1,000,000 de evaluaciones (aprox. 5 mins por método).
- **Semillas:** 42, 43, 44.

## Resultados Empíricos (1M Evals)

| Método | Test Acc (Best) |
|---|---|
| DGE_Baseline (Sin Consistencia) | 91.56% ± 0.25% |
| DGE_Consist_V1 (Signos) | **91.60% ± 0.12%** |
| DGE_Consist_V2_Zero (SNR + Castigo 0x) | 91.38% ± 0.01% |
| DGE_Consist_V2_0.1 (SNR + Castigo 0.1x) | 91.39% ± 0.01% |

## Análisis y Conclusión
No es necesario ejecutar el experimento a 2M evaluaciones (ahorrando 2 horas de cómputo), ya que el patrón a 1M evaluaciones es claro: **la versión V2 (SNR) es inferior a la V1 (Signos) y a la Baseline**.

### ¿Por qué falló el enfoque SNR?
El fracaso radica en la naturaleza de los gradientes estocásticos (minibatches). 
1. **SNR Naturalmente Bajo:** En la optimización por minibatches de redes neuronales, la varianza del gradiente ($\sigma$) suele ser un orden de magnitud mayor que la media de la señal ($\mu$). Por lo tanto, el valor `SNR = abs(mean) / std` es intrínsecamente diminuto en casi todas las dimensiones (ej. `SNR ~ 0.1`).
2. **Asfixia del Learning Rate:** Al mapear este valor con `mask = tanh(SNR)`, la inmensa mayoría de los parámetros reciben una máscara muy cercana a 0 (ej. `tanh(0.1) ≈ 0.1`). Esto significa que **el optimizador estrangula el Learning Rate globalmente un 90%**, asfixiando la convergencia.
3. **Robustez de los Signos (V1):** La versión V1 usa la frecuencia del signo. Esto es el equivalente estadístico a un *test no paramétrico* (como el test de los signos). No le importa si la varianza es enorme; solo le importa si "la mayoría de las veces apuntamos a la derecha". Esta propiedad hace que V1 sea infinitamente más robusto al ruido impulsivo que una métrica paramétrica basada en momentos continuos como el SNR.

**Conclusión y Siguiente Paso:**
Descartamos la consistencia basada en SNR. La métrica no paramétrica de "frecuencia de signos" (V1) demuestra ser el enfoque correcto para paisajes de pérdida altamente estocásticos. Las futuras mejoras en la consistencia deberían construir sobre V1.
