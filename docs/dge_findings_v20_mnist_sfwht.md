# Hallazgos de Investigación - DGE v20 (SFWHT MNIST)

## 1. Objetivo de la Iteración
Tras el rotundo éxito del experimento analítico `v19` usando SFWHT para descomprimir un gradiente esparso artificial, el objetivo de la iteración `v20` era aplicar **Layer-wise SFWHT** a una arquitectura multicapa real entrenando en la base de datos MNIST.
El enfoque consistió en separar los pesos por capa (haciendo *padding* a la potencia de 2 más cercana) y aplicar la estimación de Hadamard de forma estructural, actualizando los parámetros con un optimizador Adam.

## 2. Configuración del Experimento
- **Archivo:** `scratch/dge_sfwht_mnist_v20.py`
- **Arquitectura:** MLP `784 -> 32 -> 10` (~25,450 parámetros).
- **Parámetros SFWHT:** Tamaño de Bucket $B = 256$, $\epsilon = 5e-3$, *threshold* dinámico.
- **Evaluaciones por Paso:** Alrededor de $2,560$ evaluaciones por actualización de Adam.
- **Presupuesto Total:** $200,000$ llamadas a la función de pérdida.

## 3. Resultados Cuantitativos (Métricas)
Los resultados detallados se encuentran en `results/raw/sfwht_mnist_v20.json`.
- **Accuracy Pico:** **53.33%** (Entrenamiento) / **53.33%** (Test) alcanzado alrededor de las $112,640$ y $168,960$ evaluaciones.
- **Divergencia Final:** Al final del presupuesto, la precisión de prueba cayó al **45.17%** y el loss aumentó a 3.21.
- **Rendimiento Computacional:** $13.5$ segundos de tiempo real. Extremadamente veloz gracias al paralelismo en CPU/GPU y la transformada rápida.

## 4. Conclusiones y Análisis (Diagnostics)
1. **Comportamiento "Spike and Crash":** La estimación inicial funciona increíblemente bien. El modelo avanza del 10% al 50% de precisión en solo $\approx 80,000$ evaluaciones. Esto demuestra que la señal es correcta. Sin embargo, llegados a este punto, la optimización colapsa y el loss empieza a fluctuar hacia arriba.
2. **Causa del Colapso (Sparsity Dinámica):** SFWHT asume que el gradiente subyacente es fuertemente esparso. Al inicio del entrenamiento, unas pocas "features" clave dominan el gradiente. Conforme el modelo aprende lo básico, los errores se vuelven sutiles, el gradiente se vuelve "denso" (ruido blanco distribuido) y SFWHT empieza a tener colisiones en los *buckets* (múltiples variables activas cayendo en el mismo bucket).
3. **Decaimiento de Hiperparámetros:** El experimento carecía de decaimiento en la tasa de aprendizaje (`lr`) y en la perturbación (`epsilon`). El aumento en la pérdida indica fuertemente que estamos rebotando alrededor de un mínimo o que la estimación está inyectando ruido puro.
4. **Validación del Enfoque por Capas:** Abordar cada capa por separado permitió hacer *padding* limpio (L1 a $32,768$ y L2 a $512$). Esto simplificó mucho la lógica matricial y es el camino a seguir.

## 5. Próximos Pasos
- **Mitigación de Colisiones:** Para una versión `v21`, es necesario usar **Permutaciones Aleatorias** de los índices (o una matriz diagonal aleatoria como *pre-conditioner*) antes de la transformada de Hadamard. Esto asegura que, en cada paso, variables distintas choquen en los buckets, promediándose en el tiempo vía el EMA (similar al DGE original pero estructurado con FWHT).
- **Learning Rate Schedule:** Reintroducir el *Cosine Annealing* para el `lr` y el `epsilon` empleado en `examples/train_mnist.py`.