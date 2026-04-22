# DGE v3: Optimizador Nativo PyTorch Batched (Scaling Head)

Este documento registra los hallazgos y la arquitectura técnica de la versión 3 del algoritmo Denoised Gradient Estimation (DGE), implementada en `dge/torch_optimizer.py`.

## Motivación del Rediseño (Cuellos de Botella en V2)

Durante las iteraciones previas (como la `v30d` aplicada a MNIST completo), el tiempo real (*Wall-Clock time*) del algoritmo era prohibitivo, a pesar de usar pocas evaluaciones por iteración. El análisis de rendimiento reveló que el 70-80% del tiempo se consumía en el **overhead del framework**:

1.  **Bucles Secuenciales**: El algoritmo iteraba secuencialmente sobre las capas y luego sobre los bloques $k$ para construir las perturbaciones, lanzando miles de pequeños *kernels* desde Python.
2.  **Evaluaciones Fragmentadas**: Cada bloque se evaluaba de forma independiente (o en grupos pequeños), impidiendo saturar el pipeline masivo de la GPU.
3.  **Transferencias de Memoria**: La inicialización en NumPy (v2) obligaba a copiar tensores de CPU a GPU repetidamente.

## Arquitectura V3: Vectorización Total (Batched Pure)

Para solucionar estos problemas, hemos rediseñado el core matemático bajo el paradigma `Batched_Pure`, logrando una ejecución $O(1)$ a nivel de llamadas Python:

### 1. Estado 100% PyTorch Tensor
Todos los vectores internos (momentos de Adam $m$ y $v$, histórico de consistencia, permutaciones) residen nativamente en el dispositivo (`CPU`, `CUDA` o `DirectML`), eliminando copias.

### 2. Matriz Global de Perturbaciones (`scatter_`)
En lugar de iterar bloque a bloque, generamos una única matriz gigante de perturbaciones $P$ de tamaño `(2 * total_k, dim)` mediante indexación avanzada y la operación `scatter_` de PyTorch. 
*   **Velocidad**: La construcción matemática de los escenarios de perturbación dura apenas microsegundos.
*   **Paralelismo Masivo**: $x_{batch} = x + P$ produce $2k$ configuraciones de la red que se propagan en paralelo en un solo *forward pass*.

### 3. Integración Nativa "Scaling Head" (Multi-Capa)
El gran hito de esta versión es portar la lógica de particiones por capa (comprobada en `v25b` como fundamental para la convergencia profunda) al motor vectorizado:
*   Se pueden especificar `layer_sizes` y un array de `k_blocks` (ej. `[1024, 128, 16]`).
*   El optimizador construye una matriz combinada donde las filas iniciales mutan la capa 1, las siguientes mutan la capa 2, etc.
*   **Evaluación Única**: Para una red donde la suma de bloques es $k=1168$, se lanza **1 único *forward pass* de tamaño batch 2336**, saturando por completo la capacidad de la tarjeta gráfica y maximizando los FLOPS.

## Resultados de Benchmark de Velocidad

En pruebas con una arquitectura realista ($D=109K, K=16$) ejecutada puramente en una CPU AMD Ryzen (que simula el escenario más adverso para tensores grandes frente a una GPU dedicada):

| Implementación | Steps / Segundo | Speedup |
| :--- | :--- | :--- |
| **DGE V2 (Numpy + Secuencial)** | 53.79 | 1.0x |
| **DGE V3 (Torch Batched Pure)** | **127.84** | **2.4x** |

**Conclusión:** Solo por rediseñar el software, el optimizador es **2.4 veces más rápido** en CPU. Al usar redes mayores en una GPU, este diseño en matriz (`TorchDGEOptimizer`) escalará dramáticamente mejor que cualquier implementación basada en bucles.

## Integración y Uso

La nueva clase oficial se llama `TorchDGEOptimizer`. 
Requiere un pequeño cambio en el ecosistema: la función de evaluación proporcionada por el usuario (`f_batched`) debe aceptar ahora tensores 2D `(Batch, Params)` y devolver un vector 1D de `Losses`, delegando todo el paralelismo y cálculo interno del batch a la red neuronal (típicamente mediante `torch.vmap` o `BatchedMLP`). Se incluye un mecanismo de `chunk_size` para prevenir errores de memoria (OOM) en redes extremadamente grandes.
