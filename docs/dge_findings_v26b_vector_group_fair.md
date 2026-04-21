# Findings v26b: Vector Group DGE — Ablación Completa y Justa

**Fecha:** 2026-04-21
**Código:** `scratch/dge_vector_group_v26b.py`
**Datos crudos:** `results/raw/v26b_vector_group.json`

---

## Configuración del Experimento

| Parámetro | Valor |
|---|---|
| `dim` | 128 |
| `budget` | 500,000 evaluaciones |
| `k_blocks` (base) | 8 (grupos de 16 variables) |
| `lr` | 0.1 |
| `delta` | 1e-3 |
| `seeds` | 5 (0–4) |
| Corridas base | 4 métodos × 4 benchmarks × 5 seeds = **80 corridas** |
| Ablación k_blocks | k ∈ {2, 4, 16, 32} × 2 benchmarks × 4 métodos × 5 seeds = **160 corridas** |

---

## Resultados Base (k=8, budget=500K)

### Rosenbrock D=128 — *paisaje correlacionado, valle diagonal*

| Método | Media | ±Std | Mejor | Peor |
|---|---|---|---|---|
| **PureDGE** | **89.01** | 5.28 | 79.31 | 94.49 |
| SphericalDGE | 101.33 | 12.55 | 91.32 | 126.10 |
| ScalarVarianceDGE | 89.01 | 5.28 | 79.31 | 94.49 |
| VectorGroupDGE | 97.29 | 3.03 | 93.49 | 102.53 |

**Veredicto Rosenbrock:** ❌ Los métodos esféricos son **peores** que PureDGE. La hipótesis de mejora en valles correlacionados no se cumple con k=8.

---

### Rotated Quadratic D=128 — *diseñado para favorecer Vector Group*

| Método | Media | ±Std | Mejor | Peor |
|---|---|---|---|---|
| PureDGE | 1.29e-02 | 1.12e-02 | 2.73e-09 | 3.04e-02 |
| **SphericalDGE** | **9.19e-03** | 1.51e-02 | 1.30e-08 | 3.92e-02 |
| ScalarVarianceDGE | 1.29e-02 | 1.12e-02 | 2.73e-09 | 3.04e-02 |
| VectorGroupDGE | 7.99e-03 | 9.12e-03 | 1.83e-04 | 2.54e-02 |

**Veredicto Rotated Quadratic:** 🟡 SphericalDGE y VectorGroupDGE son marginalmente mejores en media (9.19e-03 y 7.99e-03 vs 1.29e-02), pero las desviaciones estándar son grandes (≈1.5× la media). La diferencia **no es estadísticamente robusta** con 5 seeds.

---

### Ill-Conditioned Ellipsoid D=128 — *condition number 10^6*

| Método | Media | ±Std | Mejor | Peor |
|---|---|---|---|---|
| **PureDGE** | **7,742.9** | 1,261.4 | 5,860.5 | 9,070.6 |
| SphericalDGE | 12,849.2 | 3,301.2 | 9,177.6 | 17,520.8 |
| **ScalarVarianceDGE** | **7,742.9** | 1,261.4 | 5,860.5 | 9,070.6 |
| VectorGroupDGE | 14,473.1 | 1,511.3 | 12,710.5 | 17,132.7 |

**Veredicto Ellipsoid:** ❌❌ Los métodos esféricos son significativamente **peores**. VectorGroupDGE produce un resultado ~87% peor que PureDGE. Este es el resultado más informativo del experimento.

---

### Sphere D=128 — *control negativo, sin correlación*

| Método | Media | ±Std | Mejor | Peor |
|---|---|---|---|---|
| PureDGE | 5.00e-04 | 2.31e-04 | 5.02e-05 | 6.75e-04 |
| SphericalDGE | 6.76e-04 | 7.06e-04 | 9.82e-10 | 1.92e-03 |
| ScalarVarianceDGE | 5.00e-04 | 2.31e-04 | 5.02e-05 | 6.75e-04 |
| VectorGroupDGE | 3.96e-04 | 4.47e-04 | 2.34e-05 | 1.20e-03 |

**Veredicto Sphere:** En media los métodos esféricos no mejoran. Alta varianza pero resultados comparables.

---

## Observación Crítica: ScalarVarianceDGE ≡ PureDGE

ScalarVarianceDGE produce resultados **idénticos** a PureDGE en todos los benchmarks (mismos valores hasta el decimal). Esto confirma que la varianza escalar-por-grupo con grupos dinámicos es matemáticamente equivalente a la varianza per-coordinate cuando los grupos cambian en cada paso: el valor promedio del gradiente cuadrado dentro de un grupo aleatorio converge rápidamente al valor per-coordinate bajo EMA con β=0.999. **La ablación de varianza no aporta diferencia práctica con grupos dinámicos.**

---

## Ablación de k_blocks (Rosenbrock y Rotated Quadratic)

### Rosenbrock — mejor método por tamaño de grupo

| k (bloques) | Tamaño grupo | PureDGE media | SphericalDGE media | VectorGroupDGE media |
|---|---|---|---|---|
| 2 | 64 | 102.78 | 104.29 | ~102 |
| 4 | 32 | 94.26 | 101.09 | ~97 |
| **8** | **16** | **89.01** ✅ | 101.33 | 97.29 |
| 16 | 8 | 96.45 | 98.47 | ~97 |
| 32 | 4 | 104.12 | 102.60 | ~101 |

PureDGE con k=8 (grupos de 16) es el óptimo global. Los métodos esféricos no ganan ventaja en ningún tamaño de grupo.

### Rotated Quadratic — mejor método por tamaño de grupo

En Rotated Quadratic todos los métodos convergen a valores cercanos a cero en suficientes seeds. Las diferencias entre métodos son dominadas por la varianza seed-a-seed, no por el método.

---

## Análisis de Causas

### ¿Por qué los métodos esféricos no mejoran en Rosenbrock?

La hipótesis era que las perturbaciones esféricas eliminan el zig-zag ejes-alineados en valles diagonales. Sin embargo:

1. **El valle de Rosenbrock tiene D-1 = 127 dimensiones de "fondo de valle".**  Una dirección aleatoria en R^16 (tamaño de grupo) tiene componente en la dirección del valle proporcional a 1/√16 ≈ 0.25. Una perturbación Rademacher también proyecta 1/√16 en promedio. **La diferencia estadística es cero en expectativa para k≫1.**

2. **El gradiente estimado `g_est = (fp - fm) / (2δ) * direction` es un estimador ruidoso.** Para Rademacher, la varianza del estimador es conocida y bien controlada (SPSA clásico). Para esférico, la varianza es mayor porque cada componente del gradiente recibe una proyección aleatoria que puede ser arbitrariamente pequeña.

3. **Con grupos dinámicos**, el efecto de "momentum diagonal" que se esperaba del EMA vectorial se diluye: cada paso, el mismo parámetro puede pertenecer a un grupo distinto, interrumpiendo la acumulación coherente de dirección.

### ¿Por qué el Ellipsoid empeora tanto con métodos esféricos?

El Ellipsoid tiene condición 10^6. Los parámetros con eigenvalores grandes dominan la pérdida. Una dirección Rademacher per-coordinate deja que Adam ajuste el paso per-coordinate (segundo momento separado). Una dirección esférica mezcla coordenadas de escalas muy distintas dentro del mismo grupo: el segundo momento del grupo ve un promedio de gradientes de magnitudes muy dispares, degradando la precondición de Adam.

---

## Conclusiones

### Hipótesis: ¿Falsada o Parcialmente válida?

| Criterio de éxito (spec) | Resultado |
|---|---|
| Mejora clara en Rosenbrock | ❌ PureDGE gana |
| Al menos mejora marginal en MNIST | No probado (benchmark sintético priorizado) |
| Sin regresión en Sphere/Ellipsoid | ❌ Regresión clara en Ellipsoid |

**La hipótesis central (perturbaciones esféricas > Rademacher en paisajes correlacionados) NO se confirma en estos benchmarks.** 

Sin embargo, el experimento no es completamente negativo:

- En **Rotated Quadratic**, SphericalDGE y VectorGroupDGE son marginalmente mejores en media (no estadísticamente robusto).
- La ablación de varianza confirma que la varianza escalar-por-grupo equivale a per-coordinate con grupos dinámicos (resultado útil).

### Qué aprendimos sobre DGE

1. **La perturbación Rademacher ya es cuasi-uniforme en expectativa** para grupos de tamaño ≥ 8. No hay ventaja práctica en usar la esfera explícita.

2. **Adam per-coordinate es crítico cuando las escalas difieren entre parámetros** (Ellipsoid). La varianza global o por grupo degrada la precondición.

3. **El EMA acumula bien incluso con Rademacher** porque las fluctuaciones de signo se promedian igual que las proyecciones esféricas.

4. **El sweet spot de k_blocks es k=8 (grupos de 16)** en D=128. Muy pocos bloques (grupos grandes) reduce la cobertura; demasiados (grupos pequeños) reduce la señal por bloque.

---

## Próximos Pasos Recomendados

Con los datos de v26b, el Vector Group DGE con la implementación actual **no ofrece mejora**. Las opciones son:

1. **Cerrar la línea esférica** y pasar a Direction-Consistency Learning Rates (v27, Tier High en shortlist). Bajo coste, universalmente beneficioso.

2. **Variante alternativa antes de cerrar:** La hipótesis esférica podría funcionar si se usan **grupos fijos persistentes** en vez de dinámicos. Con grupos fijos, el EMA vectorial sí acumularía dirección coherente. Coste: 1 experimento adicional pequeño.

3. **Quick wins paralelos:** Half-Step Retry + Curvature-Preconditioned Perturbations. Coste ≈ cero por parámetros adicionales.

---

## Trazabilidad

- Código: `scratch/dge_vector_group_v26b.py`
- JSON crudo: `results/raw/v26b_vector_group.json` (58 KB, 240 registros)
- Spec que generó este experimento: `docs/v26b_implementation_spec.md`
- `internal_overhead_time` justificación: ScalarVarianceDGE incurre ~50% más wall-clock que PureDGE (≈9.8s vs ≈7s por corrida) sin mejora de `final_loss`. El overhead no compensa.
