# Análisis v20: SFWHT en MNIST — Diagnóstico y Plan de Ataque

## Veredicto Rápido

El resultado es **exactamente lo esperado** dado el diseño actual, y no es un fracaso.

- **Lo bueno:** 10% → 53% en ~80K evals demuestra que SFWHT *sí* extrae señal útil de un modelo real. La velocidad (13.5s totales) es brutal.
- **Lo malo:** Colapso posterior con divergencia del loss. El modelo "des-aprende".
- **Lo revelador:** El patrón "spike and crash" es el fingerprint clásico de un estimador que funciona en régimen sparse pero falla cuando la esparsidad desaparece.

## Diagnóstico: 3 Problemas Identificados (por orden de gravedad)

### 🔴 Problema 1: Colisiones de Bucket (el limitante fundamental)

El doc v20 lo identifica correctamente: cuando el gradiente deja de ser sparse, múltiples variables "chocan" en el mismo bucket. El Peeling asigna la magnitud a la variable equivocada.

**Pero el problema es más profundo de lo que parece.** En v19, el gradiente tenía 15 variables activas de 1M → ratio `K/B = 15/1024 ≈ 0.015`. En v20, la capa 1 tiene `784×32 = 25,088` parámetros con `B=256` → ratio `D/B = 98`. Incluso si solo el 10% de variables tienen gradiente significativo, eso son ~2,500 activas en 256 buckets ≈ **~10 colisiones por bucket**.

**Conclusión:** Con esos números, SFWHT Peeling no tiene esperanza de separar variables. No es un problema de threshold — es que la asunción fundamental (K << B) se viola masivamente.

### 🟡 Problema 2: Sin LR/Epsilon Decay

Correcto que falta. Pero esto solo explica la *divergencia* final, no el techo en 53%. Con decay, el modelo probablemente se estabilizaría alrededor de 50-55% en vez de crashear, pero no subiría mucho más.

### 🟡 Problema 3: Perturbación Unidireccional

En el código (líneas 97-98), la estimación usa solo `x + ε·H`, no el par `(x+ε·H, x-ε·H)`. Esto introduce un sesgo DC proporcional a `f(x)` que contamina todos los buckets. La corrección en línea 102 (`U_base[0] -= f_x`) solo limpia el bucket 0, no la contribución de fondo a los demás buckets.

**Fix:** Usar evaluación simétrica `(f(x+ε·H) - f(x-ε·H)) / (2ε)` que cancela los términos pares automáticamente. Cuesta el doble pero elimina el sesgo por completo.

## Propuesta para v21: SFWHT Híbrido

El error conceptual de v20 es intentar usar SFWHT como estimador *completo* del gradiente. En realidad, SFWHT debería usarse como **filtro de importancia** que identifica *qué variables medir*, no como sustituto del gradiente.

### Arquitectura propuesta (3 fases por step):

```
FASE 1: SFWHT Scan (barato)
├── Aplicar SFWHT por capa con B pequeño
├── No intentar Peeling completo
├── Solo identificar los Top-K buckets con mayor |U[b]|
└── Coste: B evals por capa

FASE 2: Refinamiento dirigido (preciso)
├── Para cada bucket activo, medir la respuesta a las perturbaciones
│   de las variables que mapean a ese bucket
├── Usar perturbaciones simétricas ±ε para cada variable candidata
├── O usar DGE estándar (bloques aleatorios) SOLO sobre las variables
│   en los buckets activos
└── Coste: ~2K evals (K = variables en buckets activos)

FASE 3: Adam update
├── Actualizar solo las variables con señal
├── Las demás mantienen su EMA sin nueva información
└── Coste: 0 evals
```

**¿Por qué esto resuelve las colisiones?** Porque no le pides a SFWHT que te dé el gradiente — solo le pides que te diga *dónde mirar*. El gradiente real lo calculas con un estimador que no tiene el problema de colisiones (DGE estándar o finite differences sobre las K variables seleccionadas).

### Presupuesto estimado:

- Capa L1 (25,088 params): B=256 scan → 256 evals, Top-50 variables refinadas → 100 evals → **356 evals**
- Capa L2 (330 params): B=64 scan → 64 evals, Top-20 refinadas → 40 evals → **104 evals**
- **Total por step: ~460 evals** (vs 2,560 en v20)

### Reducción de coste: **5.6x menos evaluaciones por step** con mayor precisión.

## Alternativa más simple: Permutaciones Aleatorias (lo que propone el doc v20)

El doc v20 propone permutaciones aleatorias antes de la WHT para aleatorizar las colisiones. Esto es correcto conceptualmente (es Random Hashing) pero tiene un problema:

- Con D/B ≈ 100, aleatorizar las colisiones no las elimina. Solo las distribuye uniformemente.
- Necesitarías ~D/B ≈ 100 permutaciones diferentes y promediar, lo que equivale a volver a coste O(D). 

**Mi opinión:** Las permutaciones son útiles como complemento pero no resuelven el ratio D/B. El enfoque híbrido (scan + refine) es más potente.

## Otra idea: Adaptar B por capa dinámicamente

En vez de B=256 fijo:

| Capa | D_real | D_pad | B recomendado | Ratio D/B |
|---|---|---|---|---|
| L1 (W1+b1) | 25,120 | 32,768 | 2,048 | 16 |
| L2 (W2+b2) | 330 | 512 | 512 | 1 |

Con B=2,048 para L1, cada bucket tiene ~16 variables en promedio. Si el gradiente tiene sparsity ≥ 90% (≤ 2,500 activas), hay ~1.2 activas por bucket → **Peeling debería funcionar**.

**Coste:** 2048 + 512 = 2,560 evals por step (igual que v20 pero con B correcto por capa).

## Resumen de Opciones para v21

| Opción | Complejidad | Upside | Riesgo |
|---|---|---|---|
| A: Subir B por capa + evaluación simétrica + LR decay | Baja | Medio (mejora precision, evita crash) | Sigue asumiendo sparsity |
| B: SFWHT como scanner + DGE/FD refinamiento | Media | **Alto** (elimina el problema de colisiones) | Más código, más ingeniería |
| C: Permutaciones aleatorias + EMA temporal | Baja | Bajo-medio (no resuelve ratio D/B) | Puede no mejorar mucho |
| **Recomendación: A primero, B después** |  |  |  |

La opción A se puede probar en 1h (cambiar B, añadir `±ε`, añadir cosine annealing). Si el techo sube de 53% a >70%, SFWHT directo aún tiene vida. Si se estanca, ir a B.
