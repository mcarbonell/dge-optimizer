# Findings v29: Paper Statistics — ConsistencyDGE MNIST (6 Seeds + Convergence Curves)

**Fecha:** 2026-04-21
**Código:** `scratch/dge_paper_stats_v29.py`
**Datos crudos:** `results/raw/v29_paper_stats.json`

---

## Configuración

| Parámetro | Valor |
|---|---|
| Arquitectura | MLP [784 → 128 → 64 → 10] (~109K params) |
| `budget` | 800,000 evaluaciones |
| `k_blocks` | (1024, 128, 16) |
| `lr` | 0.05 |
| `delta` | 1e-3 |
| `window (T)` | 20 |
| `n_train` | 3,000 / `n_test` 600 |
| `batch_size` | 256 |
| Seeds | **42, 43, 44, 45, 46, 47** (6 seeds) |
| Checkpoints | Cada 40K evals (20 puntos por curva) |

---

## Resultados principales

### Por seed

| Seed | PureDGE | ConsistencyDGE | Δ |
|---|---|---|---|
| 42 | 79.50% | 85.17% | **+5.67pp** |
| 43 | 79.50% | 88.00% | **+8.50pp** |
| 44 | 81.67% | 87.67% | **+6.00pp** |
| 45 | 83.50% | 87.17% | **+3.67pp** |
| 46 | 83.50% | 88.50% | **+5.00pp** |
| 47 | 85.67% | 89.00% | **+3.33pp** |

**ConsistencyDGE supera a PureDGE en las 6/6 seeds. Margen mínimo: +3.33pp.**

### Resumen estadístico

| Método | Media | ±Std | 95% CI | Mejora |
|---|---|---|---|---|
| PureDGE | 82.22% | 2.25% | [80.38%, 84.06%] | — |
| **ConsistencyDGE** | **87.58%** | **1.23%** | **[86.58%, 88.59%]** | **+5.36pp / +6.5%** |

**Los intervalos de confianza del 95% no se solapan → la mejora es estadísticamente significativa.**

Std reducida: 2.25% → 1.23% (**−45%**). ConsistencyDGE no solo mejora la media — reduce la varianza entre inicializaciones.

---

## Curva de convergencia

| Evals | PureDGE | ConsistencyDGE | Gap |
|---|---|---|---|
| 42,048 | 68.14% | 71.91% | **+3.77pp** |
| 121,472 | 74.00% | 81.78% | **+7.78pp** |
| 200,896 | 73.89% | 83.72% | **+9.83pp** |
| 280,320 | 74.94% | 84.67% | **+9.73pp** |
| 362,080 | 77.53% | 85.86% | **+8.33pp** |
| 441,504 | 79.61% | 86.39% | **+6.78pp** |
| 520,928 | 80.67% | 87.25% | **+6.58pp** |
| 600,352 | 81.36% | 87.19% | **+5.83pp** |
| 682,112 | 81.75% | 87.42% | **+5.67pp** |
| 761,536 | 81.69% | 87.33% | **+5.64pp** |

### Interpretación de la curva

**Fase temprana (0–200K):** La separación abre rápidamente. A 42K evals (solo el 5% del budget), ConsistencyDGE ya lidera en +3.77pp. A 200K, el gap alcanza casi 10pp. Esto demuestra que ConsistencyDGE **converge más rápido**, no solo más lejos.

**Pico de divergencia (200–400K):** El gap máximo se produce entre 200K–360K evals (~9–10pp). En este rango, ConsistencyDGE ya está cerca de su plateau mientras PureDGE sigue subiendo despacio.

**Saturación (400K–800K):** Ambos métodos se estabilizan. PureDGE llega a ~82%, ConsistencyDGE a ~87%. El gap convergido es de ~5.6pp sostenido. No hay regresión en ningún método.

**Implicación para el paper:** La figura de convergencia muestra una separación clara desde el inicio y estable hasta el final. No es un efecto de plateau — es un cambio en la velocidad y calidad de aprendizaje desde el primer paso.

---

## Análisis

### 1. La mejora es robusta y sin excepciones

**6/6 seeds** muestran mejora positiva. La peor seed (47) tiene +3.33pp y la media de PureDGE en esa seed (85.67%) es la más alta de todo el conjunto. Esto sugiere que incluso cuando PureDGE funciona bien, ConsistencyDGE lo supera.

### 2. La consistencia de signo es un indicador mejor que la varianza de Adam

Adam's `v_t` (segundo momento) captura la **magnitud** de las actualizaciones históricas. La máscara de consistencia captura la **dirección** — si el gradiente estimado apunta consistentemente en la misma dirección. Son señales ortogonales:

- `1/sqrt(v_t)` normaliza la magnitud → parámetros muy móviles no dominan.
- `consistency_mask` filtra la dirección → parámetros con signo oscilante no mueven.

La combinación es más informativa que cada métrica por separado.

### 3. La mejora en velocidad temprana tiene implicaciones prácticas

Si tuviésemos que usar un **budget de 300K** (en lugar de 800K), los resultados serían:

| Budget | PureDGE | ConsistencyDGE |
|---|---|---|
| 40K | ~68% | ~72% |
| 200K | ~74% | ~84% |
| 300K | ~75% | ~85% |

ConsistencyDGE es más **eficiente en evaluaciones**: logra en 200K lo que PureDGE apenas alcanza en 800K.

### 4. Comparación con referencias históricas

| Versión | Método | Seeds | Acc | Std |
|---|---|---|---|---|
| v25b | PureDGE ~109K | 3 | 81.17% | — |
| v28 | PureDGE | 3 | 80.00% | 1.57% |
| v28 | ConsistencyDGE | 3 | 87.56% | 0.77% |
| **v29** | **PureDGE** | **6** | **82.22%** | **2.25%** |
| **v29** | **ConsistencyDGE** | **6** | **87.58%** | **1.23%** |

El resultado de 6 seeds confirma el de 3 seeds con gran precisión (87.58% vs 87.56%).

---

## Conclusiones

### Veredicto

✅ **CONFIRMADO CON ALTA CONFIANZA ESTADÍSTICA.**

| Criterio | Resultado |
|---|---|
| Mejora positiva en todas las seeds | ✅ 6/6 |
| CIs del 95% no solapados | ✅ [80.4%, 84.1%] vs [86.6%, 88.6%] |
| Reducción de varianza | ✅ −45% std |
| Mejora en velocidad temprana | ✅ +3.77pp ya a 42K evals |
| Convergidos sin regresión | ✅ gap sostenido en 400K–800K |

### Para el paper

Los datos de v29 son suficientes para una reclamación cuantitativa sólida:

> **ConsistencyDGE logra 87.6% ± 1.2% de accuracy en MNIST** (n=6 semillas, 95% CI: [86.6%, 88.6%]), comparado con **82.2% ± 2.3%** del baseline PureDGE (95% CI: [80.4%, 84.1%]). La mejora de +5.4pp es estadísticamente significativa (intervalos de confianza no solapados) y consistente en las 6 semillas evaluadas. La convergencia más rápida de ConsistencyDGE es visible desde el 5% del budget de evaluaciones.

---

## Trazabilidad

- Código: `scratch/dge_paper_stats_v29.py`
- JSON: `results/raw/v29_paper_stats.json`
- Experimento previo (3 seeds): `docs/dge_findings_v28_consistency_mnist.md`
- Synthetics (base teórica): `docs/dge_findings_v27_consistency_lr.md`
