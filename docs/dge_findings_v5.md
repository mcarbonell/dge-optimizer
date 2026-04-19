# DGE — Hallazgos Preliminares (actualizado)

**Última actualización:** 2026-04-18  
**Estado:** Prototipo experimental — iteración activa  
**Archivos:** `scratch/dge_prototype_v3.py` … `scratch/dge_prototype_v5.py`

---

## Contexto

La idea de DGE (Estimación Dicotómica de Gradiente) surgió el 2026-04-18 como alternativa
de coste `O(log D)` al cálculo de gradiente por diferencias finitas `O(D)`.
El whitepaper conceptual está en `docs/denoised_gradient_estimation_idea.md`.

---

## Arquitectura del algoritmo (estable desde v3)

Cada paso de optimización:

1. Genera `k = ⌈log₂(D)⌉` bloques aleatorios de variables.
2. Evalúa `f(x + δ·pert)` y `f(x − δ·pert)` por bloque (signos SPSA aleatorios).
3. **Paso greedy:** desplazamiento en la dirección del bloque con mayor `|Δf|`.
4. **Adam (v4+):** momentos por variable del gradiente estimado acumulado.
5. **Decay coseno** de `lr` y `delta` para convergencia final sin oscilación.

**Coste por iteración:** `2k = 2⌈log₂(D)⌉` evaluaciones de `f(x)`.

---

## Evolución de versiones

| Versión | Cambio principal | Resultado |
|---------|-----------------|-----------|
| v1 | Implementación inicial | Diverge (bug signo) |
| v2 | Fix signo + escala directa | Diverge (sin normalización) |
| v3 | Paso greedy normalizado | Converge estable. Lento en alta-D |
| v4 | **Adam + cosine decay** | Salto 440–7300x en convergencia |
| v5 | **lr escalado 1/√k + benchmark SPSA** | Comparación honesta obtenida |

---

## Resultados experimentales

### v3 vs v4 — mejora por Adam

| Test | v3 gap | v4 gap | Mejora |
|------|--------|--------|--------|
| Esfera D=512 (denso) | 1.96e+03 | **4.44e+00** | ~440x ✅ |
| Esfera Sparse 5% D=512 | 2.07e+01 | **2.84e-03** | ~7300x ✅ |
| Rosenbrock D=2 | oscila ~0.17 | **0.099 monótono** | Sin oscilación ✅ |
| Esfera D=65536 | mejora lenta | diverge ❌ | lr demasiado agresivo |

---

### v5 — Benchmark DGE vs SPSA (presupuesto justo: 20,000 evals)

> **Metodología:** mismo `x0`, mismo presupuesto total de evaluaciones.
> SPSA calibrado con `lr = lr_base / D` para evitar divergencia (estándar en la literatura).
> DGE usa `lr` escalado por `1/√k` para invarianza dimensional.

| Benchmark | D | DGE gap | SPSA gap | Ganador | Nota |
|-----------|---|---------|----------|---------|------|
| Esfera densa | 512 | 3.31e-01 | 2.09e-02 | **SPSA 16x** | Esperado |
| Esfera sparse 5% | 512 | 6.66e-04 | 8.92e-07 | **SPSA 750x** | Ver análisis |
| Esfera alta-D | 65536 | 2.31e+05 ❌ | 1.99e+04 | **SPSA** | DGE diverge |
| Rosenbrock | 10 | 6.90e+00 | 8.74e+67 ❌ | **DGE** | SPSA diverge |

#### Análisis crítico de los resultados

**¿Por qué gana SPSA en convexas?**  
SPSA tiene una ventaja estructural de frecuencia: con el mismo presupuesto hace
`10,000 pasos × 2 evals` frente a `1,111 pasos × 18 evals` de DGE. En funciones
suaves y convexas, muchos minipasos ruidosos superan a pocos pasos más informados.
Es la misma razón por la que SGD-minibatch supera a GD exacto en ML.

**¿Por qué gana DGE en Rosenbrock?**  
SPSA con `lr = 0.02/D = 0.002` en D=10 aun diverge por el valle curvo no convexo.
Adam en DGE amortigua las oscilaciones usando varianza acumulada por variable —
SPSA no tiene memoria de segundo orden.

**La conclusión real sobre el dominio de DGE:**  
El caso de uso de DGE **no** es reemplazar SPSA en funciones convexas simples.
Su ventaja aparece donde SPSA falla o es inestable:

1. **Paisajes no convexos / valles curvos** — Adam hace a DGE más robusto.
2. **Caja negra real con llamadas costosas** — DGE extrae más información por evaluación.
3. **Redes neuronales sin backprop** — paisaje altamente no convexo y no diferenciable.
4. **Presupuesto extremadamente bajo** — pocas evaluaciones, máxima información por paso.

---

## Hallazgos confirmados

1. ✅ **Adam sobre gradiente acumulado = mejora 440–7300x** sobre paso EMA simple (v3→v4).
2. ✅ **Decay coseno de lr+delta elimina oscilación** (Rosenbrock).
3. ✅ **SPSA gana en convexas con presupuesto igual** — ventaja de frecuencia de pasos.
4. ✅ **DGE gana en Rosenbrock** donde SPSA diverge — robustez Adam vs ruido SPSA.
5. ⚠️ **Alta dimensión sigue siendo el talón de Aquiles** — Adam diverge en D=65536 incluso con lr escalado por `1/√k`. Requiere gradient clipping o normalización adicional.
6. ⚠️ **El caso de uso real de DGE es ML/no-convexo**, no optimización convexa pura.

---

## Próximas iteraciones

### v6 — Red XOR sin backprop ← en curso
- Red neuronal 2→4→1 con activación sigmoide.
- Entrenamiento con DGE como único optimizador (sin PyTorch autograd, sin backprop).
- Métrica: convergencia del loss de entrenamiento vs. SPSA con mismo presupuesto.
- **Hipótesis:** DGE supera a SPSA en este paisaje no convexo real.

### v7 — Gradient clipping en Adam para alta dimensión
- Clip de norma del Adam update: `update = clip(adam_update, max_norm)`.
- Objetivo: estabilizar D=65536 sin sacrificar la velocidad de convergencia en D pequeños.

---

## Conclusión actualizada

> DGE v5 con Adam confirma su fortaleza en paisajes no convexos (Rosenbrock)
> donde SPSA diverge. En funciones convexas simples, SPSA gana por frecuencia
> de pasos. El experimento clave para validar la propuesta original del whitepaper
> es el entrenamiento de una red neuronal sin backprop (v6), donde el paisaje es
> genuinamente no convexo y la robustez de Adam sobre gradiente acumulado
> debería proporcionar una ventaja real.

---

*Documento actualizado tras prototipo v5 — 2026-04-18.*
