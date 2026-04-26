# DGE: Oportunidades y Experimentos Exploratorios
## Brainstorming — Abril 2026

**Contexto:** Este documento recoge ideas de alto potencial derivadas del análisis del repositorio DGE tras ~70 iteraciones de desarrollo. El objetivo no es criticar lo existente, sino identificar **vectores de exploración** donde DGE podría tener ventajas estructurales que aún no se explotan.

**Metodología de trabajo:** Cada idea incluye una hipótesis, un experimento concreto (timeboxed), y un criterio de éxito/fallo. Si el experimento falla, se archiva. Si funciona, se integra al roadmap.

---

## Idea 1: Curvatura Diagonal por Bloque — Segundo Orden "Gratis"

### La observación
DGE evalúa `f(x+δ)` y `f(x-δ)` para cada bloque. La diferencia da la pendiente (primer orden). Pero la suma da información de curvatura:

```
curv_ii ≈ (f(x+δ) + f(x-δ) - 2·f(x)) / δ²
```

Esta es una estimación de la **diagonal de la Hessiana** a lo largo de la dirección de perturbación del bloque. DGE actualmente desecha esta información.

### Hipótesis
Si una dimensión tiene alta curvatura local, un paso grande de gradiente puede ser inestable (sobrepaso). Si tiene curvatura baja, un paso pequeño es ineficiente. Adaptar el tamaño de paso local por la curvatura estimada acelerará la convergencia sin aumentar el coste de evaluaciones.

### Experimento (2h)
1. Modificar `optimizer.py` para guardar, por cada bloque evaluado:
   ```python
   curvature_block = (fp + fm - 2*f0) / (delta**2)
   ```
2. Acumular esta curvatura en un buffer EMA separado (beta=0.9).
3. Usarla como precondicionador: dividir el update por `sqrt(1 + |curv_ema|)`.
4. Correr en MNIST v30 (full MNIST, 3M evals) comparando:
   - PureDGE (baseline)
   - DGE + Curvatura adaptativa

### Criterio de éxito
- Mejora de ≥1pp en accuracy final, O
- Reducción del número de steps para alcanzar 90% accuracy en ≥20%

### Criterio de fallo
- La curvatura estimada es demasiado ruidosa para ser útil (varianza > señal), O
- El overhead computacional del precondicionador es >5% del wall-clock

---

## Idea 2: Reconstrucción por Compressed Sensing — Gradiente Sparse

### La observación
DGE muestrea gradientes con máscaras de bloques aleatorios. Esto es formalmente un problema de **Compressed Sensing**:
- Señal: gradiente verdadero `g ∈ R^D`
- Mediciones: `k` proyecciones por bloques
- Matriz de sensing: matriz de incidencia bloque-dimensión

El EMA actual es una reconstrucción iterativa "barata" pero subóptima. La teoría de CS dice que si `g` es `s`-sparse, se puede reconstruir exactamente con `O(s·log(D/s))` mediciones usando un solver como OMP o LASSO.

### Hipótesis
En cualquier paso de entrenamiento de una red neuronal, el 70-90% de los gradientes son cercanos a cero (gradiente efectivamente sparse). Un solver de CS sobre una ventana de mediciones recientes podría reconstruir un gradiente mucho más preciso que el EMA puro.

### Experimento (4h)
1. En MNIST con backprop habilitado (para tener ground truth), correr DGE durante 50 steps guardando:
   - Todas las mediciones de bloques `(idx_block, fp, fm, signs)`
   - El gradiente Adam real en cada paso
2. Construir la matriz de sensing acumulada y resolver:
   ```
   min ||g_hat||_1  sujeto a  A·g_hat ≈ y
   ```
   donde `A` es la matriz de incidencia y `y` son las mediciones observadas.
3. Comparar correlación coseno entre:
   - `g_hat_CSA` (reconstrucción CS) vs `g_Adam`
   - `g_hat_EMA` (DGE actual) vs `g_Adam`

### Criterio de éxito
- Correlación coseno del estimador CS ≥ 0.7 con el gradiente real (vs ~0.3 del EMA actual en early training)

### Criterio de fallo
- El problema de CS es demasiado grande para resolverlo en tiempo real (D=100K)
- El gradiente no es suficientemente sparse como para que la reconstrucción mejore sobre el EMA

---

## Idea 3: DGE Multi-Objetivo — Optimización Pareto

### La observación
Backpropagation requiere un **único escalar** de pérdida. Si tienes múltiples objetivos (accuracy + regularización L1 + fairness + energy), Adam los combina con pesos fijos a priori.

DGE evalúa forward passes. Puedes evaluar **la misma red** con múltiples funciones de pérdida simultáneamente sin coste adicional de gradientes.

### Hipótesis
Un optimizador multi-objetivo basado en DGE puede encontrar soluciones Pareto-óptimas que Adam con loss combinada no puede alcanzar, porque DGE puede seleccionar bloques que mejoran múltiples objetivos simultáneamente o hacer trade-offs dinámicos.

### Experimento (3h)
1. MNIST con dos objetivos:
   - `L_acc`: cross-entropy (accuracy)
   - `L_sparse`: L1 de los pesos (sparsidad)
2. Implementar DGE multi-objetivo:
   - Un EMA y un buffer de consistencia por objetivo
   - Cada bloque se evalúa con ambas funciones de pérdida
   - Selección del bloque: el que tenga mejor producto de mejoras normalizadas (o usando métricas de dominancia de Pareto)
3. Comparar contra:
   - Adam con loss combinada `L = L_acc + λ·L_sparse`
   - DGE single-objective (solo accuracy)

### Criterio de éxito
- A igual accuracy, la red multi-objetivo tiene ≥20% más pesos exactamente cero, O
- Se encuentra un frente de Pareto con 3+ puntos no dominados

### Criterio de fallo
- Los objetivos entran en conflicto de tal manera que DGE no puede progresar en ninguno
- El overhead de evaluar múltiples pérdidas domina el tiempo (aunque forward es compartido, backward no existe)

---

## Idea 4: Transferencia de Momentos DGE — "Warm Start" Estructural

### La observación
Adam de cero necesita ~1000 pasos para que `m` y `v` se estabilicen. DGE también. Pero los momentos DGE representan "sensibilidad local de cada parámetro" — una propiedad semántica que podría ser transferible entre arquitecturas del mismo tipo.

### Hipótesis
Si entrenas un MLP pequeño con DGE, los momentos acumulados capturan patrones de sensibilidad (por ejemplo: "los pesos de los bordes de MNIST son más importantes que los del centro"). Al escalar a una arquitectura más grande del mismo tipo, estos patrones de sensibilidad pueden transferirse como inicialización "caliente" de los momentos DGE.

### Experimento (3h)
1. Entrenar MLP pequeño `[784→32→10]` con DGE durante 5000 steps. Guardar `m`, `v`, y buffers de consistencia.
2. Inicializar MLP grande `[784→128→64→10]` con:
   - Pesos: He init (normal)
   - Momentos DGE: interpolados espacialmente desde los del modelo pequeño (replicar o promediar)
3. Entrenar el modelo grande desde este warm-start vs desde cero.
4. Métrica: "steps hasta alcanzar 80% accuracy".

### Criterio de éxito
- El warm-start alcanza 80% accuracy en ≤50% de los steps del baseline (desde cero)

### Criterio de fallo
- La interpolación de momentos destruye más información de la que transfiere
- El warm-start no acelera la convergencia (momentos son demasiado específicos de la topología)

---

## Idea 5: Optimización de Arquitectura con DGE — NAS sin Backprop

### La observación
DGE no necesita grafos computacionales. Cualquier función ejecutable que acepte un vector de parámetros y devuelva un escalar es optimizable. Esto incluye funciones donde la **estructura misma** de la red es un parámetro.

Backpropagation no puede optimizar estructuras porque cambiar la arquitectura rompe el grafo de computación.

### Hipótesis
DGE puede co-optimizar los pesos de una red Y su arquitectura (número de neuronas, conexiones, skip-connections) simultáneamente, porque ambos solo afectan el forward pass.

### Experimento (6h)
1. Definir una "super-red" con capas opcionales:
   - Capas base: `[784→128→64→10]`
   - Conexiones skip opcionales (binarias, entrenables como parámetros)
   - Neuronas ocultas con máscaras de activación (grupos de 8 neuronas que pueden "apagarse")
2. El vector de parámetros incluye:
   - Pesos normales (continuos)
   - Máscaras de skip (continuos, cuantizados a 0/1 en forward)
3. DGE optimiza todo el vector como si fuera un problema de optimización black-box.
4. Observar si DGE descarta conexiones y neuronas innecesarias.

### Criterio de éxito
- DGE encuentra una sub-arquitectura con ≥10% menos parámetros que mantiene ≥95% del accuracy
- Se descubren skip-connections que mejoran el accuracy vs la arquitectura base

### Criterio de fallo
- El espacio de arquitecturas es demasiado grande y DGE se pierde
- Las máscaras binarias hacen el paisaje demasiado discontinuo para que DGE progrese

---

## Idea 6: Minibatch Fijo por Ventana — Eliminación de Varianza Estocástica

### La observación
En MNIST, DGE evalúa `f(x+δ, batch_i)` y `f(x-δ, batch_i)`. El batch cambia cada step. El ruido del minibatch se superpone al ruido de la perturbación.

¿Qué pasa si usamos el **mismo minibatch para T steps consecutivos**? Adam no puede hacer esto sin overfitting catastrófico (se especializa en un batch). Pero DGE, por su naturaleza estocástica de exploración por bloques, podría resistir el overfitting porque cada paso solo actualiza una fracción de parámetros.

### Hipótesis
Fijar el batch durante una ventana de 20-30 pasos elimina la varianza del minibatch, permitiendo que el EMA de DGE filtre solo el ruido de perturbación. Esto acelerará la convergencia o la estabilizará.

### Experimento (2h)
1. Implementar DGE con "batch windowing":
   - Generar un batch aleatorio de 256 muestras
   - Correr 20 steps de DGE sobre ese batch fijo
   - Generar nuevo batch, repetir
2. Comparar contra:
   - DGE con batch nuevo cada step (baseline)
   - DGE con batch nuevo cada 5 steps
3. Métricas: curva de accuracy vs evals, varianza entre seeds, señal de overfitting (train acc >> test acc).

### Criterio de éxito
- A igual número de evaluaciones, el accuracy es ≥2pp superior al baseline, O
- La varianza entre seeds se reduce ≥30%

### Criterio de fallo
- Overfitting severo: train acc → 99%, test acc se estanca o baja
- No hay mejora significativa en convergencia

---

## Idea 7: Perturbaciones Aprendidas — "Meta-Noise"

### La observación
DGE usa perturbaciones Rademacher aleatorias (`±1`). Pero quizás algunas direcciones de perturbación son sistemáticamente más informativas que otras para una arquitectura dada.

### Hipótesis
Se puede aprender un vector de "escalas de perturbación" `σ ∈ R^D` (inicializado en 1.0) que se adapta durante el entrenamiento. Si una dimensión consistentemente da gradientes con alta señal, `σ_i` aumenta. Si da ruido, `σ_i` disminuye.

Esto es meta-aprendizaje del proceso de muestreo mismo.

### Experimento (3h)
1. Añadir un vector `sigma` de escala por parámetro (inicializado en 1.0).
2. Las perturbaciones se generan como `pert = sigma * sign * delta`.
3. Actualizar `sigma` con un meta-learning rate muy pequeño basado en la magnitud del gradiente estimado o la consistencia de signo.
4. Correr en MNIST y observar si `sigma` desarrolla patrones estructurados (por ejemplo, capas iniciales con sigma alto, capas finales con sigma bajo).

### Criterio de éxito
- `sigma` converge a un patrón no uniforme que mejora la SNR del gradiente estimado
- Mejora ≥1pp en accuracy final

### Criterio de fallo
- `sigma` colapsa a un valor uniforme (no hay estructura aprendible)
- La meta-optimización es inestable

---

## Sprint Recomendado — Priorización

| Prioridad | Idea | Tiempo | Impacto potencial | Riesgo |
|---|---|---|---|---|
| 🔥 1 | Curvatura diagonal (Idea 1) | 2h | Alto | Bajo — es información gratis |
| 🔥 2 | Batch fijo por ventana (Idea 6) | 2h | Alto | Bajo — cambio simple |
| 3 | Multi-objetivo (Idea 3) | 3h | Alto | Medio — nicho muy valioso |
| 4 | Compressed Sensing (Idea 2) | 4h | Muy alto | Alto — puede ser demasiado lento |
| 5 | Transferencia de momentos (Idea 4) | 3h | Medio | Medio — no garantizado |
| 6 | Perturbaciones aprendidas (Idea 7) | 3h | Medio | Medio — meta-learning inestable |
| 7 | NAS con DGE (Idea 5) | 6h | Muy alto | Alto — ambicioso |

---

## Notas para Futuras Sesiones

- Si Ideas 1 o 6 funcionan, se pueden combinar: curvatura adaptativa + batch fijo = convergencia más estable y más rápida.
- Si Idea 2 funciona, abre una puerta teórica enorme: DGE no sería solo "SPSA con memoria", sino un método de reconstrucción comprimida con garantías teóricas de CS.
- Si Idea 3 funciona, DGE tendría un nicho único e inimitable por backprop: optimización multi-objetivo nativa.

**Principio de acción:** Cada experimento debe ser timeboxed. Si en el tiempo asignado no hay señal clara, se archiva y se pasa al siguiente. No perseguir ideas muertas.
