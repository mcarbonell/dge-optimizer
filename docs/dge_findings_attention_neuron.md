# DGE Findings: Neurona de Atención y Unificada (v1-v6)

## Objetivo del Experimento
Explorar la viabilidad de tratar la neurona completa (todo su *fan-in* y *fan-out*) como una única variable optimizable en lugar de entrenar pesos individuales. Este enfoque conceptual, derivado de discusiones teóricas sobre escalabilidad computacional y estrategias de atención local, busca reducir la cantidad de parámetros entrenables de $O(\text{pesos})$ a $O(\text{neuronas})$.

Se plantearon y evaluaron seis arquitecturas incrementales sobre el dataset MNIST:

### 1. V1: Neurona de Energía Constante (Atención Local + Bias Angular)
- **Hipótesis:** Normalizar el fan-in de forma estricta ($\sum |W| = 1$) y aplicar un escalar $delta\_mult$ para que la neurona funcione como un mecanismo puro de atención de energía constante. Se utilizó un bias modelado como $sin(\theta)$ para acotar la activación.
- **Resultados empíricos:** 11.35% (Puro Azar).
- **Análisis:** Fracaso por cancelación matemática. Al aplicar un multiplicador escalar a todos los pesos de la neurona y luego normalizar el vector resultante mediante L1, el escalar se anula matemáticamente ($c \cdot W / |c \cdot W| = sign(c) \cdot W / |W|$). El gradiente respecto a $delta\_mult$ se vuelve cero, impidiendo el aprendizaje.

### 2. V2: Neurona Unificada (Backpropagation sobre 2 variables)
- **Hipótesis:** Tratar a cada neurona como una entidad controlada por exactamente 2 parámetros ($delta\_mult$ y $delta\_add$), manteniendo los pesos aleatorios originales congelados.
- **Resultados empíricos (10 epochs):** **76.18% Accuracy**
- **Análisis:** Éxito rotundo en compresión paramétrica. Un MLP estándar equivalente (784 -> 512 -> 10) requiere ~400,000 parámetros. Esta red alcanzó más de un 76% de precisión utilizando **solo 1,566 parámetros entrenables**. Demuestra empíricamente que una red puede extraer características útiles alterando la "relevancia" topológica global de la neurona en lugar de afinar conexiones atómicas.

### 3. V3: Neurona Estocástica (Subconjuntos Aleatorios + Bias Angular)
- **Hipótesis:** Introducir asimetría y robustez aplicando los deltas solo a un 50% de los cables aleatoriamente en cada paso (Evolución Divergente), combinado con el bias angular $sin(\theta)$ para evitar la explosión de activaciones.
- **Resultados empíricos (15 epochs):** **69.36% Accuracy**
- **Análisis:** La máscara aleatoria actúa como un regularizador extremadamente fuerte (similar a DropConnect). Previene el sobreajuste pero frena dramáticamente la velocidad de convergencia con Backpropagation tradicional. El bias angular se mostró muy efectivo para mantener la estabilidad numérica.

### 4. V4: Optimizador Greedy Adaptativo Puro
- **Hipótesis:** Abandonar Backpropagation y el cálculo de gradientes. Utilizar Búsqueda Local (Greedy) con lotes masivos (8192) y la regla evolutiva 1/5 de Rechenberg, alterando aleatoriamente una neurona por iteración.
- **Resultados empíricos (1000 pasos / 13 min):** ~25.43% Accuracy
- **Análisis:** El algoritmo logra descender el Loss de forma monotónica perfecta (sin rebotes), pero al evaluar una neurona de manera puramente secuencial en CPU, la optimización es prohibitivamente lenta comparada con los enfoques vectorizados. Valida el descenso topológico pero resalta la necesidad de paralelismo masivo para el entrenamiento Greedy On-Chip.

### 5. V5 y V5b: Escala Bruta (Red Masiva / Mediana)
- **Hipótesis (V5):** Escalar la arquitectura a 4096x4096 neuronas (aprox. 20M conexiones originales) para verificar si la superabundancia de cables aleatorios permite "encontrar" las configuraciones ideales usando los mismos multiplicadores globales.
- **Hipótesis (V5b):** Probar con un tamaño intermedio (1024x1024) para descartar problemas puramente dimensionales.
- **Resultados empíricos:** V5 estancada (11.35%), V5b mejora lenta (41.47%).
- **Análisis:** El intento de escalar la red fracasó debido a un desvanecimiento extremo del gradiente y problemas de escala. Al aplicar inicialización He/Kaiming en redes tan anchas, los pesos aleatorios iniciales ($w\_init$) son diminutos ($1 / \sqrt{N}$). La optimización con solo 2 variables globales no tiene capacidad para sortear este estrecho valle, demostrando que la "fuerza bruta dimensional" no es suficiente.

### 6. V6 y V6b: Cables Independientes (Dual Neuron Factorization)
- **Hipótesis (V6 - Rank 1):** Permitir la divergencia asimétrica cruzando la orden multiplicativa de la neurona de entrada con la neurona de salida para cada cable ($w\_evolved = w\_init \cdot (\delta\_in\_m \otimes \delta\_out\_m)$). Esto logra variaciones únicas por cable (Rank 1) usando pocos parámetros.
- **Hipótesis (V6b - Rank 2):** Aumentar ligeramente la capacidad dotando a la factorización cruzada de rango 2 (cada neurona usa 2 vectores en lugar de un escalar).
- **Resultados empíricos (10 epochs):**
  - **V6 (Rank 1):** **88.32% Accuracy** (4,158 parámetros)
  - **V6b (Rank 2):** **94.53% Accuracy** (7,794 parámetros)
- **Análisis:** **¡Hito absoluto!** Rompimos el techo paramétrico. La factorización de Rango 1 logra un 88.3% con una reducción de parámetros del 99%. Sin embargo, la **Factorización de Rango 2 (V6b)** cruzó contundentemente el muro del 93-94%, logrando un **94.53% de precisión** con tan solo **7,794 parámetros entrenables** (comparado con ~400k de la base original).

## Conclusiones y Próximos Pasos
1. **Reducción de parámetros:** La hipótesis de la "Neurona como Variable" (V2) es viable y permite reducir la huella de memoria del optimizador en más de un 99.6%.
2. **Incompatibilidad de Normalización Escalar:** Los deltas multiplicativos globales son incompatibles con la normalización de la suma del fan-in a menos que exista un término sumativo que rompa la escala lineal.
3. **El Triunfo del Rango Bajo:** El éxito abrumador del V6b (94.5% con ~7.7k parámetros) confirma que las matrices de pesos de las redes Fully Connected son altamente redundantes. Imponer una modulación estructural basada en el producto cruzado de las neuronas que conectan el cable es inmensamente más eficiente que escalar su dimensionalidad estática (como demostró el fracaso del V5).
4. **Potencial:** Integrar esta arquitectura de "Cables Independientes" factorizados dentro del propio algoritmo DGE como método nativo de compresión podría generar arquitecturas ultra-profundas entrenables con presupuestos de cómputo marginales.