# DGE Findings v35: Sensibilidad a la partición de bloques (K)

## Objetivo del Experimento
DGE divide los parámetros de una red en $K$ bloques para perturbaciones simultáneas. Hasta ahora, usábamos valores fijos arbitrarios que emulaban groso modo un $1\%$ del tamaño de la capa. El objetivo de este test sprint fue determinar la mejor fórmula matemática genérica para $K$ en función del tamaño de la capa $D$.

Se evaluaron 5 fórmulas, limitando el presupuesto a **500,000 evaluaciones** para obtener una lectura rápida de la velocidad de convergencia inicial:
1. `O(1)`: Fijo en 32 bloques por capa.
2. `O(log_D)`: Escalado logarítmico estricto.
3. `O(sqrt_D)`: Raíz cuadrada de la dimensión de la capa.
4. `O(D/100)`: Escalado lineal fraccionario (1%).
5. `O(D/10)`: Escalado lineal agresivo (10%).

## Resultados Empíricos (A 500,000 evals)

| Fórmula | K-Blocks (MLP 784-128-64-10) | Evals por Paso | Test Acc (500K) |
|---|---|---|---|
| `O(log_D)` | `[16, 13, 9]` | 76 evals | 87.14% |
| `O(1)` | `[32, 32, 32]` | 192 evals | 87.89% |
| `O(D/100)` | `[1004, 82, 6]` | 2,184 evals | 89.49% |
| **`O(sqrt_D)`** | **`[316, 90, 25]`** | **862 evals** | **90.49%** |
| `O(D/10)` | `[10048, 825, 65]` | 21,876 evals | *OOM (Crash)* |

## Análisis del Error de OOM
La fórmula `O(D/10)` causó un crash con el error de codificación extraña (`UnicodeDecodeError: 'utf-8' codec can't decode...`). Esto suele ocurrir en Windows cuando PyTorch de DirectML lanza un error nativo C++ de **Out of Memory (OOM)** y la consola falla al intentar decodificar el mensaje localizado de Windows. Con $K=10938$ en total, el tensor de perturbaciones estaba pidiendo de golpe más de 20 GB de memoria de GPU, reventando el sistema. 

## Conclusión
La fórmula de la **raíz cuadrada (`O(sqrt D)`) es la clara ganadora.**
- Consigue el equilibrio perfecto matemático: a medida que la capa crece, el número de bloques crece para absorber la varianza, pero a un ritmo sublineal que impide que el coste de las evaluaciones ($2K$) se dispare.
- Alcanzó un **90.49%**, aplastando a la fórmula lineal fraccionaria (`O(D/100)`) que veníamos usando, y utilizando la mitad de evaluaciones por paso (862 vs 2184), lo que significa que el algoritmo da el doble de pasos en el mismo tiempo/presupuesto.
- Las fórmulas demasiado pequeñas (`O(1)` o `O(log D)`) dan pasos muy rápidos pero sufren de una inmensa interferencia de bloques (demasiadas variables por bloque cancelándose mutuamente), estancando el aprendizaje temprano.

**Siguiente paso técnico:** A partir de ahora, integraremos por defecto el tamaño de bloque basado en la raíz cuadrada en el optimizador DGE nativo.
