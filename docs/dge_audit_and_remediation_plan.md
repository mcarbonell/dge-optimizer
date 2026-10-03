# Auditoría Técnica Integral y Plan de Remediación — DGE (Denoised Gradient Estimation)

**Alcance:** revisión completa de idea, código, documentación, experimentos, resultados y paper; evaluación de suficiencia para publicación científica.
**Rol del auditor:** revisor externo escéptico (referee de workshop/conferencia) + ingeniero de software.
**Estado del repo auditado:** rama `main`, commit `2c4b1d2` ("docs: update DGE quantization performance benchmarks in README").
**Ubicación del repositorio:** `c:\Users\mrcm_\Local\proj\algorithms\dge-optimizer`.

> Este documento es el resultado de una auditoría solicitada por el autor. Incluye tanto fortalezas como defectos, y prioriza los problemas que un revisor científico o un revisor de reproducibilidad detectarían.

---

## 1. Resumen ejecutivo

### 1.1 Veredicto global

| Dimensión | Estado | Comentario |
|---|---|---|
| **Idea / contribución** | 🟡 Prometedora pero mal encuadrada | El núcleo (bloques aleatorios + EMA + máscara de consistencia de signo) es una combinación sólida, pero gran parte de las piezas son conocidas (SPSA por bloques, diferencias finitas por bloques). El elemento potencialmente novedoso es **DS-EMA**, y aún no está aislado con rigor. |
| **Código** | 🔴 Necesita trabajo | Bugs reproducibles (baselines duplicados, contabilidad de tiempos errónea, configs que crashean), sin tests, sin empaquetado. |
| **Reproducibilidad** | 🔴 Insuficiente | Muchas afirmaciones del paper/README no tienen JSON crudo versionado; figuras del paper con números *hardcodeados*. |
| **Documentación** | 🟡 Abundante pero inconsistente | 80 documentos, excelente historial de iteración, pero con cifras contradictorias y docs obsoletos. |
| **Paper (estado listo para envío)** | 🔴 No enviar todavía | Hay problemas de integridad metodológica (baseline MeZO ficticio, ablación sin datos, presupuesto asimétrico no declarado). Subsanables, pero bloqueantes para revisión por pares. |
| **Potencial de publicación** | 🟢 Realista (workshop o venue de zeroth-order/DFO) | Con las remediaciones P0–P2, el trabajo tiene una historia honesta y publicable. El hallazgo más fuerte y defendible es el **entrenamiento nativo en regímenes no diferenciables/cuantizados (sin STE)**. |

### 1.2 Los 3 riesgos que un revisor detectará primero

1. **El baseline "MeZO" no es MeZO**: en `scratch/dge_fullmnist_comparison_v30e.py` la función `run_spsa()` se invoca dos veces (`"SPSA"` y `"MeZO"`), produciendo resultados **bit a bit idénticos**. El paper reporta dos filas distintas en la Tabla 1 que en realidad son el mismo método.
2. **La figura de ablación usa números inventados/aproximados**: `paper/figures/generate_figures.py` (`fig4_ablation_components`) codifica a mano `88.0` y `91.0` etiquetados como "approx from v29 intermediate", sin JSON que los respalde.
3. **Presupuesto asimétrico no declarado**: DGE recibe 3,000,000 evaluaciones mientras SPSA/MeZO reciben 300,000 (10×), y el paper no lo dice en la sección *Setup*. El propio roadmap del repo exige "equal evaluation budgets first".

---

## 2. Inventario del repositorio

- **Raíz:** `README.md`, `GEMINI.md` (reglas internas de agentes), `.gitignore`.
- **Librería (`dge/`, ~514 LOC):** `optimizer.py` (canónico, NumPy), `torch_optimizer.py` (batched, PyTorch), `__init__.py`.
- **Experimentos (`experiments/`, ~1217 LOC):** `run.py`, `run_ml.py`, `baselines.py`, `benchmarks.py`, `aggregate.py`, `plot.py`, `utils.py` + 22 configs JSON.
- **Sandbox de investigación (`scratch/`, ~25.9K LOC):** ~90 scripts `dge_*_vN.py` (v1→v69). Es el verdadero laboratorio del algoritmo.
- **Documentación (`docs/`, 80 archivos):** idea original, teoría, roadmaps, vision, feedback y ~67 archivos `dge_findings_v*.md`.
- **Paper (`paper/`):** `dge_paper_v2.tex` (+`references.bib`, `figures/`), borrador previo `dge_paper_draft.tex`.
- **Resultados (`results/`):** 184 artefactos versionados en `raw/` (JSON crudos y agregados), `summary/`, `figures/`.
- **Datos (`data/`):** MNIST + CIFAR-10, ~404 MB **no versionados** (correcto; ver `.gitignore`).
- **Ejemplos (`examples/`):** `train_mnist.py`.

### 2.1 Métricas de salud del repo

| Chequeo | Resultado |
|---|---|
| Tests (`pytest`/`unittest`) | ❌ Ninguno |
| `LICENSE` | ❌ Ausente |
| `requirements.txt` / `pyproject.toml` / `setup.py` | ❌ Ausente |
| `CITATION.cff` | ❌ Ausente |
| CI (GitHub Actions) | ❌ Ausente |
| Tamaño de `data/` | 404 MB (ignorado por git) |
| Archivos versionados | 419 |
| Último commit | `2c4b1d2` |

---

## 3. Evaluación de la idea y de la contribución científica

### 3.1 ¿Qué es realmente DGE?

Formalmente, el algoritmo canónico (`dge/optimizer.py`, `step()`, líneas 169–229) hace, en cada paso:

1. Genera una permutación aleatoria de las `D` coordenadas y las parte en `k` bloques **no solapados** que cubren *todo* el espacio (`np.array_split`).
2. Para cada bloque, muestrea signos Rademacher `s`, perturba `±δ` sólo ese bloque y forma la diferencia centrada: `grad[block] = (f(x+pert) − f(x−pert)) / (2δ) · s`.
3. Inyecta el gradiente (disperso por bloques, pero cubriendo todas las coordenadas) en un filtro **Adam/EMA** (`m`, `v`).
4. Escala la actualización por una **máscara de confianza** = *Dual Sign-EMA* (dos EMAs de `sign(grad)`, con *crossover gate*).

Coste: `2k` evaluaciones de función por paso.

### 3.2 ¿Qué hay de nuevo y qué no?

| Componente | ¿Novedoso? | Referencia / observación |
|---|---|---|
| Perturbación por bloques con signos Rademacher | ❌ No | Es **SPSA por bloques** / *random block-coordinate finite differences*. Conocido (Spall 1992; Nesterov 2011; conn2009introduction). |
| Diferencia centrada → estimador insesgado | ❌ No | Resultado clásico (elasticidad/SPSA). El propio paper lo demuestra como si fuera un teorema, pero es el lema estándar de SPSA. |
| Reducción de varianza al achicar el bloque | ❌ No | Consecuencia directa de perturbar menos dimensiones por bloque. Está en la literatura de DFO (block-SPSA, random coordinate methods). |
| Filtro temporal tipo Adam sobre el gradiente estimado | ❌ No | Acoplar un estimador ZO a Adam es práctica común (incluido el propio MeZO con SGD/Adam). |
| **DS-EMA** (máscara *crossover* MACD sobre EMAs de signo) | 🟡 Parcialmente | La pieza con apariencia más original, pero (a) no tiene análisis teórico y (b) es una variante de *sign-momentum gating* / signSGD. Necesita ablación limpia para acreditarse como contribución. |

**Conclusión de novedad:** la combinación es ingeniosa y produce resultados plausibles, pero si el paper se vende como "resolvemos la maldición de la dimensionalidad" o "estimador O(log D)", es **falsable y falso** (de hecho, el propio `docs/dge_feedback.md` ya lo señaló: "No es O(log D). Es O(D/log D) de ruido por O(log D) de coste = O(D) como SPSA"). La narrativa honesta y publicable es: *"un optimizador ZO por bloques con denoising temporal y gating de consistencia de signo, especialmente útil en regímenes no diferenciables y cuantizados, donde backprop falla sin STE"*.

### 3.3 Disciplina de claims: qué está verificado, preliminar o especulativo

El README ya usa una taxonomía 🟢/🟡/🔴 que es una buena práctica. Pero al cruzar con los artefactos:

| Claim | Fuente | Artefacto crudo | Veredicto |
|---|---|---|---|
| 94.36% MNIST (MLP 109K) sin backprop | `paper/dge_paper_v2.tex`, `README` | `results/raw/v30e_fullmnist_comparison.json` | ✅ Verificado (mean 0.94357) |
| DS-EMA > SMA(T=20) en sintéticos y MNIST | `docs/dge_findings_v67_dual_ema.md` | `results/raw/v67_dual_ema.json`, `v68_...json` | 🟡 Verificado en 5 semillas, pero falta ablación con presupuesto igual y sin la ventana "edge effect" sin datos |
| INT8 82.20% / INT4 77.80%, Adam ~9% | `docs/dge_findings_v32.md`, `paper` Tabla 2 | ❌ **No hay JSON en `results/raw/`** | 🔴 No reproducible desde el repo |
| Sign activations 73.20% vs Adam 61.20% | `docs/dge_findings_v31.md`, `paper` Tabla 2 | ❌ **No hay JSON** | 🔴 No reproducible desde el repo |
| MeZO colapsa a ~21% | `paper` Tabla 1 | `v30e` (pero **es SPSA**, no MeZO) | 🔴 Baseline incorrecto |
| Ablación de componentes (bloques / +EMA / +consistencia) | `paper` Fig. 4 | ❌ Valores *hardcodeados* en `generate_figures.py` | 🔴 No reproducible / potencialmente fabricado |
| K = O(√D) óptimo | `paper` §Method | `results/raw/v35_k_scaling` (existe `dge_findings_v35`) | 🟡 Verificar que el JSON existe y respalda 90.49% |
| Fine-tuning LLM / trillones de parámetros | `docs/dge_vision_and_capabilities.md` | — | 🔴 Especulativo (correctamente fuera del paper) |

---

## 4. Auditoría de código

### 4.1 Estructura y API

- La librería `dge/` expone **dos** optimizadores, pero `dge/__init__.py` sólo reexporta `DGEOptimizer`:
  ```python
  from .optimizer import DGEOptimizer
  __all__ = ["DGEOptimizer"]
  ```
  El resultado estrella del paper (94.36%) se produjo con `TorchDGEOptimizer` (`scratch/dge_fullmnist_comparison_v30e.py`), que **no es importable** como `from dge import ...`. Es una incoherencia de API que confunde al usuario externo.
- El parámetro `consistency_window` de `DGEOptimizer` es engañoso: no controla ninguna ventana, sólo se usa como booleano (`> 0`) para activar DS-EMA (cuyas constantes `alpha_f=0.3`, `alpha_s=0.05` están fijas en el código, no parametrizadas).
- Firma con deuda técnica visible (`dge/optimizer.py` líneas 105–110): `clip_norm`, `lr_scale`, `greedy_step`, `dense_update` marcados como "Deprecated / kept as no-op". Mantener parámetros muertos en la API pública invita a errores.

### 4.2 Bugs y defectos concretos (reproducibles)

| # | Severidad | Ubicación | Descripción | Evidencia |
|---|---|---|---|---|
| B1 | 🔴 Alta | `scratch/dge_fullmnist_comparison_v30e.py` | **SPSA y MeZO son el mismo código**: se llama `run_spsa("SPSA", ...)` y `run_spsa("MeZO", ...)`. En `results/raw/v30e_...json` ambos métodos tienen `values` idénticos (`[0.3757,0.1175,0.1328]`). | Verificado en el JSON crudo |
| B2 | 🔴 Alta | `paper/figures/generate_figures.py::fig4_ablation_components` | La barra de ablación usa enteros **hardcodeados** (`accuracy = [20.87, 93.00, 88.0, 91.0, 94.36]`), con dos valores "approx" sin fuente. | Líneas 223–231 |
| B3 | 🟠 Media | `experiments/configs/mnist_mlp_dge.json` (y `mnist_binary_dge.json`, `mnist_step_dge.json`, `mnist_ternary_dge.json`) | Los configs pasan `greedy_w`, que `DGEOptimizer` **no acepta** → `TypeError`. Ejecutar `python experiments/run_ml.py --config experiments/configs/mnist_mlp_dge.json` falla. | Reproducido: `TypeError: DGEOptimizer.__init__() got an unexpected keyword argument 'greedy_w'` |
| B4 | 🟠 Media | `experiments/run.py` (líneas 61–81) y `experiments/run_ml.py` (líneas 171–207) | **Contabilidad de `internal_overhead_time` incorrecta**: `f_time` es acumulativo sobre toda la run y se le resta a cada intervalo, por lo que el "overhead" sale mal calculado (puede ser negativo). Además `internal_time` y `f_time` son acumuladores, no medidas por-step. | Inspección de código |
| B5 | 🟡 Baja | `README.md` línea 16 | Cita `94.16%` cuando el valor correcto (media de 3 semillas) es `94.36%`; `94.16%` es en realidad el valor de la seed 42 (0.9416) de `v30e`. | JSON `v30e` |
| B6 | 🟡 Baja | `docs/dge_findings_v30d_fullmnist_comparison.md` | Reporta 92.98% (correcto para v30d) pero quedó **obsoleto** frente a v30e (94.36%) y no se marca como superado. | JSON `v30d` vs `v30e` |
| B7 | 🟠 Media | `experiments/benchmarks.py::ellipsoid` | Usa pesos `1000**((i-1)/(d-1))` → **κ=1000**, mientras el paper habla de **κ=10^6** (los sintéticos del paper vienen de `v67`, no de este módulo). Dos "Ellipsoid" distintos coexisten en el repo. | Comparar `benchmarks.py` con `v67_dual_ema.json` |
| B8 | 🟡 Baja | `experiments/utils.py::get_system_info` | "hardware_info" sólo reporta CPU/OS (`platform.*`), no GPU ni memoria, incumpliendo el propio estándar de logging de `GEMINI.md` ("CPU, GPU y memoria"). | `utils.py` líneas 13–19 |

### 4.3 Calidad y mantenibilidad

- **Duplicación masiva:** `scratch/` contiene ~90 scripts con grandes bloques copiados (carga de MNIST, `BatchedMLP`, `zo_acc`, `AdamState`…). Es coherente con la "REGLA DE ORO" de `GEMINI.md` (no tocar experimentos previos), pero impide mantenibilidad y propaga bugs (p. ej. B1).
- **Sin separación librería/experimentos para el resultado principal:** el resultado del paper vive en `scratch/`, no en `dge/` ni en `experiments/` con config + agregación. Un revisor no encontrará un único comando canónico que reproduzca la Tabla 1.
- **Sin tests:** no existe ningún test de correctitud del estimador (p. ej. comprobar `E[g]=∇f` en una cuadrática, comprobar la reducción de varianza, comprobar la máscara DS-EMA).
- **Sin empaquetado:** no hay `requirements.txt`, `pyproject.toml`, `LICENSE` ni `CITATION.cff`. `GEMINI.md` hardcodea una ruta local de venv (línea 24), lo que no es reproducible para terceros.

### 4.4 Reproducibilidad de experimentos

- El runner `experiments/` guarda `commit_hash`, `hardware`, `config`, `metrics` y `history` (`utils.py::save_raw_result`) — buen diseño.
- **Pero** el resultado principal del paper (v30e) y los resultados no-diferenciables (v31/v32) **no** usan ese runner ni guardan JSON (v31/v32 no tienen artefacto en `results/raw/`).
- `paper/figures/generate_figures.py` mezcla figuras cargadas de JSON (fig1, fig3, fig5) con figuras hardcodeadas (fig2, fig4), lo que rompe la trazabilidad.

---

## 5. Auditoría de documentación

**Fortalezas.** El repo tiene una trazabilidad histórica inusual y valiosa: ~67 `dge_findings_vN.md` documentando hipótesis → experimento → resultado → conclusión, más roadmaps (`dge_validation_roadmap.md`, `implementation_mini_roadmap.md`) y feedback externo (`dge_feedback.md`, `dge_feedback_2.md`). El roadmap de validación con "Success Gates" y la disciplina 🟢/🟡/🔴 son buenas prácticas.

**Problemas.**

1. **Cifras contradictorias sin jerarquía de verdad.** Un mismo resultado aparece con valores distintos:
   - MNIST full MLP-v3: `94.16%` (README) vs `94.36%` (paper) vs `92.98%` (v30d) — ver §8, Tabla A.
   - El README cita `v30e` pero con el número de una seed suelta.
2. **Documentos obsoletos no marcados.** `dge_findings_v30d` describe 92.98%/28.98% como resultados vigentes; luego v30e los mejora. No hay cabecera "SUPERSEDED BY v30e".
3. **Vocabulario inconsistente para el mismo concepto:** el algoritmo se llamó *Dichotomous* ("Estimación Dicotómica de Gradiente"), luego *Denoised Gradient Estimation* (DGE). `dge_vision_and_capabilities.md` y `denoised_gradient_estimation_idea.md` siguen usando "Dicotómica". Para el paper hay que fijar un único nombre y sigla.
4. **La idea original promete más de lo que el código hace.** `denoised_gradient_estimation_idea.md` describe una *búsqueda binaria/dicotómica* `O(log D)`; el código implementa algo distinto (partición en `k` bloques + EMA). El documento "idea madre" no se actualizó y puede inducir a error al lector del paper.
5. **`GEMINI.md` es interno** (reglas para agentes, typo "exprimentos", ruta local de venv). No debería ser lo primero que vea un revisor; conviene moverlo a `CONTRIBUTING.md` o a un doc interno.

---

## 6. Auditoría del paper (`paper/dge_paper_v2.tex`)

### 6.1 Estructura y calidad formal

Buena: abstract cuantitativo, secciones estándar (Intro/Related/Method/Theory/Experiments/Discussion/Conclusion), algoritmo en pseudocódigo, 5 figuras, 2 tablas, `references.bib` con 20 entradas. El estilo es legible.

### 6.2 Problemas de integridad/metodología (bloqueantes)

1. **Tabla 1 — baseline MeZO ficticio.** Filas `SPSA` y `MeZO` con exactamente `20.87% ± 11.83%`. Son el mismo `run_spsa()` (bug B1). Además, el `dge_findings_v30d` admite que su "MeZO_Full" usa Adam y Rademacher, es decir, *no* es el MeZO de Malladi et al. (2023). Un revisor pedirá el MeZO real (perturbación gaussiana, `z` regenerado por seed, típicamente SGD). **Acción:** implementar y correr MeZO real, o renombrar honestamente el baseline ("Global SPSA").
2. **Figura 4 — ablación sin datos.** Números hardcodeados (B2). Además el orden es incoherente: `PureDGE (93.00%)` > `DGE+EMA (88.0%)`, es decir, *añadir* el EMA empeoraría. El texto afirma que cada mecanismo "contribuye multiplicativamente", lo que no se sostiene con la propia figura.
3. **Presupuesto asimétrico no declarado.** DGE = 3M evals; SPSA/MeZO = 300K evals (10×). El paper dice "Budget: DGE methods receive 3M evals; backprop 30 epochs" y **omite** el presupuesto de SPSA/MeZO. El roadmap interno exige comparación a igual presupuesto. Aunque el argumento ("colapsan igual con más budget") sea cierto, debe (a) declararse explícitamente y (b) idealmente respaldarse con una corrida a igual presupuesto (3M) al menos para 1 semilla.
4. **Métrica mal etiquetada.** La columna se llama "Test accuracy" pero el JSON guarda `best_test_acc` (pico alcanzado, no valor final). El `std = ±11.83%` de SPSA proviene de picos seguidos de colapso. Hay que renombrar la columna ("best test accuracy") o reportar el valor final.
5. **El abstract mezcla regímenes.** "…achieving 73–82% versus ~9% for Adam on fully quantized (INT4/INT8) **and sign-activation networks**": para sign-activations Adam obtiene ~61% (Tabla 2), no ~9%. Redacción imprecisa.

### 6.3 Problemas de teoría/notación

6. **`K` se usa con dos significados incompatibles.** En el código, `K` = número de bloques y el tamaño de bloque es `D/K`. En el texto (`\\begin{lemma}[Variance Bound]`) se dice "exactly K of the D dimensions are active", es decir `K` = tamaño de bloque. El ejemplo numérico "D=100,000, K=128 → 781×" usa `K` como nº de bloques. Hay que **definir** `B` = tamaño de bloque = `D/K` y reescribir el lema como `Var(g_i) ≲ (B/D)·‖∇f‖² = (1/K)·‖∇f‖²`.
7. **Cota de varianza débil y presentada como fuerte.** `Var(g_i) ≤ (K/D)‖∇f‖²` se sigue de `max_j ∇f_j² ≥ ‖∇f‖²/D` (desigualdad holgada). Conviene dar la forma exacta `Var(g_i) = Σ_{j∈B, j≠i} (∇f_j)²` y su cota de forma honesta.
8. **Teorema de insesgadez con error de Taylor despreciado.** Bajo primer orden `f(x+Δ)≈f(x)+∇fᵀΔ`, el estimador es insesgado; pero el sesgo de segundo orden es `O(δ²)` y depende de `Tr(∇²f)`. Un paper serio debe declararlo (y el `δ` decreciente ya lo mitiga). El documento `dge_theory_and_analysis.md` menciona la aproximación pero el paper no cuantifica el sesgo.
9. **El "SNR O(√T)" asume estacionariedad estricta.** La propia §Discussion admite que el supuesto se rompe en redes discretas profundas ("butterfly effect"). Es coherente, pero el enunciado del Corolario 1 debería llevar la hipótesis explícita como condición.

---

## 7. Auditoría metodológica y estadística

1. **Nº de semillas insuficiente para el resultado principal.** El paper usa **3 semillas** para la Tabla 1 (full MNIST) y **6** para la curva de 3K. El propio `GEMINI.md`/roadmap exigen "mínimo 5 semillas por configuración" (10+ para resultados finales). Las 3 semillas de la Tabla 1 no permiten intervalos de confianza robustos ni test de significancia.
2. **Se reporta el pico (`best_test_acc`), no el valor final.** Esto favorece métodos ruidosos (un pico de suerte cuenta tanto como una convergencia estable). Para comparaciones honestas hay que reportar el valor final y/o una media de las últimas K evaluaciones.
3. **Sin intervalos de confianza ni tests apareados** en la Tabla 1 (sólo mean±std). El v29 (3K) sí hace 95% CI y "no solapamiento", pero con 6 semillas y usando solo comparación de rangos; un test bilateral (t de Welch o Wilcoxon) sería más apropiado.
4. **Comparación de coste equitativa ausente.** La narrativa "sin gradientes" es correcta, pero falta el coste real: evals de función vs tiempo de pared vs memoria. Para redes diferenciables, backprop usa `O(1)` backward por paso; el paper lo admite en §Discussion, pero no lo cuantifica con una tabla de wall-clock de la Tabla 1.
5. **Generalización limitada.** Todo el resultado fuerte es MNIST con un MLP de 109K parámetros. Faltan: (a) un segundo dataset (Fashion-MNIST, CIFAR-10 MLP), (b) un benchmark de verdad *black-box* (BBOB/RL) para sostener el claim "universal", (c) un test de escalado (¿qué pasa a 1M, 10M parámetros?).
6. **El baseline RandomDirection existe en `experiments/baselines.py` pero no se usa** en la Tabla principal, desperdiciando una comparación barata que reforzaría el resultado.
7. **Reproducibilidad de la suite no-diferenciable.** Los números de INT4/INT8/Sign (los más atractivos del paper) no tienen JSON ni script con config fija + semillas en `results/raw/`; sólo un finding en prosa.

---

## 8. Tabla A — Discrepancias numéricas entre fuentes

| Métrica | README | Paper (v2.tex) | Findings / doc | JSON crudo | Verdad probable |
|---|---|---|---|---|---|
| MNIST full MLP (DGE) | 94.16% | 94.36% | 92.98% (v30d), 94.36% (v30e) | v30e mean=0.94357 | **94.36%** (`v30e`); 94.16% = seed 42 |
| MNIST full (PureDGE) | — | 93.00% | 75.46% (v30d) | v30e mean=0.930 | **93.00%** (v30e) |
| SPSA / MeZO (full) | ~colapsa | 20.87% ± 11.83% | 28.98% ± 22.58% (v30d) | v30e mean=0.2087 | **20.87% (v30e)** pero **es el mismo método** |
| Adam (full) | 98% (implícito) | 98.00% | 97.96% (v30d) | v30e mean=0.98003 | **98.00%** |
| SGD+momentum | — | 97.78% | 97.78% (v30d) | v30e mean=0.9778 | **97.78%** |
| Sign activations | ~73% | 73.20% (Adam 61.20%) | 73.20% (v31) / 65.00% (v11) | ❌ sin JSON | **73.20% (v31)**, sin artefacto |
| INT8 / INT4 | 86.40% / 82.70% (README dice v32) | 82.20% / 77.80% | 82.20% / 77.80% (v32) | ❌ sin JSON | **inconsistente entre README y paper** |
| MNIST 3K (Consistency) | — | 87.58% ± 1.23% | 87.58% (v29) | v29 | **87.58%** |

**Nota sobre INT4/INT8:** el README afirma `86.40% INT8 / 82.70% INT4` (línea 21) mientras el paper y `dge_findings_v32.md` reportan `82.20% / 77.80%`. Dos conjuntos de números distintos para la misma afirmación. Hay que fijar un único resultado con su JSON.

### 8.1 Otras inconsistencias detectadas

- El título/sigla: *Dichotomous / Dicotómica* vs *Denoised*. Fijar "Denoised Gradient Estimation (DGE)" y corregir los docs antiguos.
- El nombre del baseline "MeZO" (ver §6.2.1).
- El número de semillas y presupuestos varía entre docs; falta una "tabla maestra" única de resultados.

---

## 9. Plan de remediación

El plan está priorizado por **riesgo para la publicación**. Las fases P0 y P1 son bloqueantes: sin ellas, enviar el paper expone al autor a acusaciones de reproducibilidad/integridad. P2–P4 convierten el trabajo en una contribución sólida y revisable.

### Fase P0 — Integridad y coherencia (bloqueante, ~2–4 días)

| Tarea | Detalle | Entregable | Criterio de aceptación |
|---|---|---|---|
| P0.1 | **Arreglar o eliminar el baseline MeZO.** Implementar MeZO real (perturbación gaussiana `z~N(0,1)`, `z` regenerado por seed, optimizador SGD/Adam, `proj_grad`), o renombrar la fila a "Global SPSA" | `experiments/baselines.py::MeZOptimizer` + JSON | Ambas filas (SPSA, MeZO) tienen valores **distintos** en el JSON |
| P0.2 | **Rehacer la figura de ablación con datos reales.** Correr 4 variantes (SPSA, solo-bloques, bloques+EMA, bloques+EMA+DS-EMA) con budget igual y ≥3 semillas; guardar JSON y generar la figura desde él | `scratch/dge_ablation_vXX.py`, `results/raw/ablation_*.json`, fig regenerada | `generate_figures.py` **no** contiene números literales; la figura se regenera del JSON |
| P0.3 | **Declarar y justificar el presupuesto.** Añadir al §Setup del paper el presupuesto exacto de cada baseline y añadir una corrida a **igual presupuesto** (3M) para SPSA/MeZO (≥1 semilla) | Texto del paper + JSON | El paper declara ambos presupuestos explícitamente |
| P0.4 | **Unificar cifras.** Crear una "tabla maestra" `docs/results_master.md` generada/validada contra `results/raw/`; corregir README (94.16→94.36, INT4/INT8) y marcar v30d como *superseded* | `docs/results_master.md`, README, headers de findings | Cero discrepancias entre README/paper/docs |
| P0.5 | **Corregir bugs de infraestructura.** B3 (configs `greedy_w`) y B4 (contabilidad de tiempos) | parches en `experiments/` | `run_ml.py --config ...mnist_mlp_dge.json` corre sin error; `internal_overhead_time ≥ 0` y coherente |
| P0.6 | **Reetiquetar métrica** (`best_test_acc` vs final) en tablas y textos | paper/README | La columna del paper coincide con el JSON |

### Fase P1 — Trazabilidad y reproducibilidad (~1 semana)

| Tarea | Detalle | Entregable |
|---|---|---|
| P1.1 | **Artefactos para cada claim del paper.** Guardar JSON crudo de v31 (sign) y v32 (INT4/INT8) y de cualquier número del paper sin resolve | `results/raw/v31_sign.json`, `v32_quant.json` |
| P1.2 | **Runner canónico del experimento principal.** Mover v30e a `experiments/` con config JSON y agregación de semillas, de modo que un comando reproduce la Tabla 1 | `experiments/configs/mnist_full_v30e.json` + `run` documentado |
| P1.3 | **Empaquetado.** `pyproject.toml`/`setup.py`, `requirements.txt` (torch, torchvision, numpy), `LICENSE` (p. ej. MIT/Apache-2.0), `CITATION.cff` | archivos en raíz |
| P1.4 | **Tests mínimos.** pytest para: (a) insesgadez en cuadrática (`E[g]=∇f`), (b) `evals_per_step==2k`, (c) reducción de varianza al aumentar `k`, (d) DS-EMA devuelve máscara en [0,1] y suprime signos oscilantes | `tests/test_optimizer.py` |
| P1.5 | **Corregir la teoría/notación.** Definir `B=D/K` (tamaño de bloque), reescribir lema de varianza como `Var(g_i)=Σ_{j∈B,j≠i}(∇f_j)²`, añadir sesgo `O(δ²)` en el teorema de insesgadez, y condicionar el SNR a estacionariedad | `paper/dge_paper_v3.tex` |

### Fase P2 — Fortaleza científica (~2–4 semanas)

| Tarea | Detalle |
|---|---|
| P2.1 | **Head-to-head a igual presupuesto** vs: SPSA, block-SPSA (varios `k`), MeZO real, Random Direction, y `(1+λ)`-ES/NES. En sintéticos (Rosenbrock, Ellipsoid κ=10^6, Rastrigin) y MNIST. |
| P2.2 | **Estadística rigurosa:** ≥5 (idealmente 10) semillas; reportar media±std, IC 95%, tests apareados (Welch/Wilcoxon) y tamaños de efecto. Reportar valor **final** y pico. |
| P2.3 | **Ablación limpia de DS-EMA** vs SMA(T=20) vs gating de consistencia simple vs signSGD puro, a igual budget. Es lo que acredita DS-EMA como contribución. |
| P2.4 | **Coste real:** tabla de wall-clock y memoria (pesos vs activaciones) DGE vs Adam/SGD en la Tabla 1. |
| P2.5 | **Segundo dataset** (Fashion-MNIST y/o CIFAR-10-MLP) y **test de escalado** D hasta ~1M parámetros. |
| P2.6 | **Reproducir y versionar los experimentos no-diferenciables** (sign, INT4/INT8, binario/ternario) con config + semillas + JSON, y unificar sus cifras (ver Tabla A). |

### Fase P3 — Escalado y validez externa (opcional, investigación futura)

| Tarea | Detalle |
|---|---|
| P3.1 | **Benchmark genuinamente black-box** (BBOB / entorno RL) para respaldar el claim de universalidad, o bien retirarlo. |
| P3.2 | **Fine-tuning de LLM con LoRA** comparado contra MeZO real (ajustar expectativa: MeZO está optimizado para ese régimen cercano al óptimo). |
| P3.3 | **Motor GPU sin cuello de botella de Python** (Triton/`scatter_` ya existe; medir y documentar *speedup* y caso de uso). |
| P3.4 | En caso de SNNs/hardware neuromórfico: demostración mínima real o degradar a "future work" (ya está bien encuadrado en README 🔴). |

### Fase P4 — Paper y difusión

| Tarea | Detalle |
|---|---|
| P4.1 | Reescribir con **disciplina de claims**: quitar todo lenguaje de "resuelve la maldición"/"reemplaza backprop"; centrar en el nicho no diferenciable/cuantizado. |
| P4.2 | Reposicionar la novedad: la contribución debe ser **DS-EMA + evidencia empírica en regímenes no diferenciables**, no el estimador por bloques (que es conocido). |
| P4.3 | Añadir sección de **Limitaciones** (butterfly effect, O(K) evals/paso, escalado) y **Reproducibility Statement** con el comando canónico. |
| P4.4 | Elegir venue realista: **workshop** (p. ej. sobre zeroth-order/DFO, neuroevolución o "efficient ML") antes de una conferencia principal. Preparar artifact/anonimización. |
| P4.5 | Publicar release con DOI (Zenodo) al enviar, para asegurar la trazabilidad de la versión enviada. |

### 9.1 Orden de ejecución recomendado

1. P0.1–P0.6 (integridad) → 2. P1.1–P1.5 (repro) → 3. P2.3 (ablación DS-EMA, define la contribución) → 4. P2.1–P2.2 (comparativas y estadística) → 5. P2.4–P2.6 → 6. P4 (paper) → 7. P3/posterior.

### 9.2 Esfuerzo estimado

| Fase | Esfuerzo | Bloquea publicación |
|---|---|---|
| P0 | 2–4 días | Sí |
| P1 | ~1 semana | Sí |
| P2 | 2–4 semanas | Sí (para un venue serio) |
| P3 | investigación abierta | No |
| P4 | 1–2 semanas | Sí (última etapa) |

---

## 10. Riesgos de publicación y recomendación de venue

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Revisor detecta que MeZO==SPSA y acusa de baseline inflado | Alta | Crítico | P0.1 |
| Revisor detecta ablación sin datos | Alta | Crítico | P0.2 |
| Rechazo por "no igual-fairness" en presupuestos | Media-alta | Alto | P0.3 |
| Rechazo por novedad insuficiente (bloques ya conocidos) | Media | Alto | P2.3 + P4.2 |
| Rechazo por generalización (solo MNIST-MLP) | Media | Medio | P2.5 |
| Reproducibilidad (json faltantes) | Media | Alto | P1.1–P1.2 |
| "Efecto del masaje de hiperparámetros" (ZO muy tuneado vs Adam por defecto) | Media | Medio | Documentar búsqueda de HP equitativa para backprop y ZO |

**Recomendación:** el trabajo NO está listo para una conferencia principal (ICML/NeurIPS) tal cual: le falta igualdad de presupuesto, baselines correctos, ablación real y generalización. **Sí** es un candidato razonable para un **workshop** sobre optimización zeroth-order/DFO, eficiencia o neuroevolución una vez completadas P0–P2. La historia más fuerte y honesta es el **entrenamiento nativo sin STE en redes cuantizadas/no diferenciables** (INT4/INT8/sign), que es un nicho con audiencia real.

---

## 11. Apéndices

### 11.1 Checklist pre-supmisión

- [ ] Todos los números del paper trazables a un JSON en `results/raw/` (P1.1).
- [ ] Baselines correctos: SPSA ≠ MeZO; MeZO real o renombrado (P0.1).
- [ ] Presupuestos declarados e iguales (o justificados con corrida igualitaria) (P0.3).
- [ ] Ablación generada desde datos (P0.2).
- [ ] ≥5 semillas y tests estadísticos para cada claim (P2.2).
- [ ] Teoría con notación consistente (`B=D/K`) y sesgo `O(δ²)` explícito (P1.5).
- [ ] `LICENSE`, `CITATION.cff`, `requirements.txt`, release con DOI (P1.3, P4.5).
- [ ] Comando canónico único que reproduce la Tabla 1 (P1.2).
- [ ] Sección de Limitaciones y Reproducibility Statement (P4.3).
- [ ] Nombre/sigla unificados ("Denoised Gradient Estimation") en todo el repo.

### 11.2 Comandos de verificación usados en esta auditoría

```bash
# 1. Confirmar que SPSA y MeZO son el mismo método (B1)
python -c "import json;d=json.load(open('results/raw/v30e_fullmnist_comparison.json'));\
print(d['summary']['SPSA']);print(d['summary']['MeZO'])"

# 2. Reproducir el crash de configs (B3)
python -c "from dge.optimizer import DGEOptimizer; DGEOptimizer(dim=10, greedy_w=0.1)"
# -> TypeError: unexpected keyword argument 'greedy_w'

# 3. Verificar valores de la Tabla 1 del paper (v30e)
python -c "import json;d=json.load(open('results/raw/v30e_fullmnist_comparison.json'));\
[print(k,round(v['mean']*100,2),'%') for k,v in d['summary'].items()]"

# 4. Confirmar que la figura de ablación está hardcodeada (B2)
grep -n "accuracy = " paper/figures/generate_figures.py
```

### 11.3 Inventario de artefactos crudos clave

| Artifacto | Método | Uso en el paper |
|---|---|---|
| `results/raw/v30e_fullmnist_comparison.json` | DGE/PureDGE/SPSA/MeZO/SGD/Adam, MNIST full, 3 seeds | Tabla 1, Fig. 1 |
| `results/raw/v30d_fullmnist_comparison.json` | idem (versión superseded) | — (obsoleto) |
| `results/raw/v29_paper_stats.json` | PureDGE vs ConsistencyDGE, MNIST-3K, 6 seeds | Fig. 5 |
| `results/raw/v67_dual_ema.json` | DS-EMA en sintéticos | Fig. 3, Tabla 2 (sintéticos) |
| `results/raw/v68_mnist_dual_ema.json` | DS-EMA en MNIST | `docs/dge_findings_v67` |
| ❌ (ausente) | Sign (v31), INT4/INT8 (v32), binary/ternary | Tabla 2 del paper — **sin artefacto** |

### 11.4 Referencias internas relevantes

- Idea: `docs/denoised_gradient_estimation_idea.md`
- Teoría: `docs/dge_theory_and_analysis.md`
- Roadmap de validación: `docs/dge_validation_roadmap.md` (Success Gates A–E)
- Feedback crítico externo: `docs/dge_feedback.md`, `docs/dge_feedback_2.md`
- Hallazgos (última iteración): `docs/dge_findings_v67_dual_ema.md`
- Paper: `paper/dge_paper_v2.tex`, `paper/references.bib`

---

*Fin del documento.*