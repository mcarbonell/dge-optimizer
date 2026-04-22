# DGE Findings v31: Sign Activation Networks

**Fecha:** 2026-04-22
**Objetivo:** Validar la capacidad de DGE para optimizar topologías discontinuas donde el gradiente analítico es nulo.
**Benchmark:** MNIST Subset (5000 train / 1000 test) con arquitectura MLP `(784, 32, 10)` y funciones de activación `torch.sign(x)`.

## Hipótesis
La función `sign(x)` produce escalones abruptos separados por infinitas llanuras donde la derivada es $0$. Adam será incapaz de optimizar las capas ocultas y solo entrenará la capa de proyección lineal final, comportándose como un clasificador lineal sobre características aleatorias. DGE, utilizando diferencias finitas con un radio de perturbación (delta) lo suficientemente amplio, podrá cruzar el escalón y estimar una dirección de descenso válida para la red profunda.

## Configuración Clave
*   **Optimizador Adam:** PyTorch nativo (`lr=1e-3`), 100 epochs.
*   **Optimizador DGE V3:** Budget de 200,000 evaluaciones. `K_BLOCKS=[16, 4]`.
*   **Delta Expandido:** `DELTA = 5e-3` (para asegurar que las perturbaciones crucen el borde de decisión).
*   **Clip Norm (Crítico):** `clip_norm = 0.05` (imprescindible para amortiguar la explosión de gradiente estimada al cruzar un umbral infinito). `LR = 0.5`.

## Resultados

| Método | Accuracy Final (Test) | Comportamiento |
| :--- | :--- | :--- |
| **Adam** | ~61.20% | Estancamiento severo. Incapaz de entrenar capas ocultas debido al gradiente 0. Actuó como regresor lineal sobre pesos aleatorios fijos. |
| **DGE V3** | **73.20%** | Entrenamiento estable cruzando la barrera de no-diferenciabilidad. |

## Conclusiones
1.  **DGE triunfa sobre topologías discontinuas:** El experimento demuestra que DGE puede entrenar con éxito componentes donde el cálculo del gradiente falla fundamentalmente por definición matemática.
2.  **Sensibilidad a hiperparámetros:** En entornos de pérdida discreta, el uso de recortes estrictos de gradiente (`clip_norm`) combinados con tasas de aprendizaje elevadas y `deltas` que crucen el umbral de discretización, es estrictamente necesario para evitar inestabilidad catastrófica.
3.  **Techo de Capacidad:** Una red de `(784, 32, 10)` con neuronas binarias (1 bit por salida) tiene un cuello de botella de información masivo. El ~73% alcanzado es prácticamente el límite teórico empírico de esta arquitectura bajo búsqueda local (similar a lo hallado en `v11` con Step Activations).

**Fase 6 Roadmap:** Este experimento da por concluido el bloque de funciones de activación de escalón/signo.
