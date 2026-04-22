# DGE Optimizer (Denoised Gradient Estimation)

📄 **[Read the Scientific Paper Draft (PDF)](paper/dge_paper_draft.pdf)**


**DGE** (**Denoised Gradient Estimation**) es un optimizador zeroth-order (libre de derivadas) que entrena Redes Neuronales y optimiza funciones matemáticas de alta dimensionalidad utilizando exclusivamente evaluaciones de funciones (forward passes), sin utilizar backpropagation.

El algoritmo combina perturbaciones aleatorias por bloques con un suavizado temporal (Adam EMA) y una máscara atencional de consistencia de dirección (Direction-Consistency LR) para extraer el gradiente latente del ruido masivo.

---

## 🟢 Validated Claims (Evidence-Based)

Las siguientes afirmaciones han sido validadas empíricamente a través de múltiples semillas y experimentos documentados rigurosamente en el directorio `scratch/`.

*   **[Validated] Entrenamiento de arquitecturas densas sin gradientes analíticos:** DGE es capaz de entrenar perceptrones multicapa (MLP) continuos de hasta ~110,000 parámetros obteniendo precisiones altamente competitivas en MNIST (`v28`: 87.56% ± 0.77%).
*   **[Validated] Eficiencia sobre SPSA:** A diferencia de SPSA (cuyo ruido escala linealmente con la dimensión), la estrategia de particionado por bloques de DGE mitiga exponencialmente la varianza, permitiendo convergencia donde SPSA colapsa.
*   **[Validated] Supremacía en entornos 100% discretos y no-diferenciables:** DGE puede entrenar con éxito arquitecturas donde Adam y la propagación hacia atrás fallan estrepitosamente (gradiente analítico = 0).
    *   **Redes con Activación Signo (Step/Sign):** DGE logra ~73% de accuracy en redes con activaciones `torch.sign` (Adam fracasa al no poder entrenar capas ocultas).
    *   **Pesos Binarios / Ternarios:** DGE logra un ~73% con pesos restringidos a $\{-1, 1\}$.
    *   **Quantization-Aware Training Nativo (INT4/INT8):** DGE entrena redes donde pesos y activaciones están forzados a una cuadrícula discreta de 4-bits o 8-bits sin usar *Straight-Through Estimators* (`v32`: 82% en INT8, 78% en INT4 vs Adam ~9%).
*   **[Validated] Universalidad en paisajes patológicos:** DGE con `Direction-Consistency LR` resuelve eficazmente topologías sintéticas hostiles como el valle de Rosenbrock y funciones cuadráticas elípticas extremadamente mal condicionadas (cond=$10^6$).

---

## 🟡 Preliminary Claims

Estas afirmaciones han sido observadas experimentalmente pero requieren mayor investigación o no se aplican universalmente.

*   **[Preliminary] Auto-ajuste del hiperparámetro K:** El tamaño óptimo de los bloques de perturbación ($K$) parece seguir una relación sublineal con la anchura de la capa de la red neuronal. Actualmente, el particionado de bloques requiere cierto ajuste manual (`K_BLOCKS`) dependiendo de la red.
*   **[Preliminary] Aceleración mediante GPUs paralelas:** El motor V3 de DGE (`TorchDGEOptimizer`) explota la paralelización masiva de GPU evaluando miles de perturbaciones en un solo *batch*, pero todavía existe un cuello de botella por el despacho secuencial de bucles desde Python que limita el aprovechamiento del hardware en modelos muy pequeños.

---

## 🔴 Speculative Claims (Future Work)

Exploraciones teóricas plausibles que aún no han sido demostradas empíricamente.

*   **[Speculative] Fine-Tuning de LLMs (MeZO):** Dado que DGE escala a 100K parámetros, es plausible que funcione para el *fine-tuning* eficiente de memoria de Modelos de Lenguaje Masivos (optimizando adaptadores LoRA sin almacenar grafos de activación), compitiendo directamente con el algoritmo MeZO.
*   **[Speculative] Spiking Neural Networks (SNNs):** El entrenamiento nativo de hardware neuromórfico no-diferenciable podría beneficiarse del enfoque de caja negra de DGE sin recurrir a gradientes subrogados.

---

## ⚠️ Limitations (Failure Regimes)

De acuerdo a la teoría matemática del algoritmo (ver `docs/dge_theory_and_analysis.md`), DGE presenta los siguientes límites estrictos:

1.  **Imposibilidad del Entrenamiento Continuo contra Backprop:** DGE **NO** pretende sustituir a Adam o al descenso de gradiente estándar en redes diferenciables. El entrenamiento analítico con propagación hacia atrás es matemática y computacionalmente muy superior. El nicho de DGE son los entornos donde backprop *no está disponible* (Black-Box) o *no funciona* (Topologías discontinuas/cuantizadas).
2.  **El "Efecto Mariposa" en Deep Sign Networks:** Aunque DGE logra entrenar redes discretas, su capacidad de aprendizaje se desploma si la red es excesivamente profunda. Dado que DGE asume estacionariedad local temporal para filtrar el ruido (EMA), si un salto de 1-bit en la capa inicial invierte por completo el estado de las capas finales (comportamiento caótico/fractal), el filtro EMA se rompe y el optimizador se estanca irremediablemente en subóptimos (~75% accuracy máximo observado).
3.  **No apto para gradientes altamente dispersos:** En problemas extremadamente densos pero con gradientes reales ralas ($K$-sparse), escaneos deterministas como *SFWHT* mostraron teóricamente mayor eficiencia espacial que la exploración estocástica ciega de DGE (aunque la implementación práctica falló por interferencia densa).

---

## 🔬 Reproducibility

Todos los resultados validados pueden ser reproducidos usando los scripts localizados en el directorio `scratch/`. Las métricas incluyen el rastreo del tiempo de Wall-Clock, tiempo de evaluación de función y gastos de sobrecarga del optimizador.

**Dependencias requeridas:**
`torch`, `torchvision`, `numpy`. (Soporte nativo para CPU, CUDA o DirectML).

### Comandos Exactos para Reproducción:

**1. Entrenamiento continuo a gran escala (Nuevo Récord MNIST ~87.5% en V3):**
Demuestra la escalabilidad de DGE frente a SPSA/MeZO usando *Direction-Consistency LR*.
```bash
python scratch/dge_fullmnist_comparison_v30e.py
```

**2. Entrenamiento en Redes Cuantizadas (INT4/INT8):**
Entrenamiento nativo y sin *Straight-Through Estimators* en hardware de 16 y 256 niveles de precisión.
```bash
python scratch/dge_quantized_mnist_v32.py
```

**3. Entrenamiento con Activaciones Discontinuas (Sign):**
Validación empírica del fallo masivo de Adam frente al éxito de caja negra de DGE.
```bash
python scratch/dge_sign_activation_mnist_v31.py
```

**4. Validación en Funciones Analíticas (Rosenbrock, Ellipsoid):**
El benchmark original de la estabilidad en problemas mal condicionados.
```bash
python scratch/dge_consistency_lr_v27.py
```

---
*Para auditar la historia empírica completa, las lecciones fallidas y el desarrollo del algoritmo, revise los documentos `dge_findings_*.md` en la carpeta `docs/`.*
