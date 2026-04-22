# DGE Optimizer: Brainstorming & Advanced Concepts

*Documento vivo para recopilar ideas experimentales, intuiciones teóricas y posibles vías de evolución para el algoritmo DGE.*

---

## 1. Adaptación Dinámica del Learning Rate (LR)

### 1.1. LR Independiente por Variable guiado por Dirección
* **Concepto:** En lugar de depender de la magnitud del gradiente (que puede causar inestabilidad u *overshooting*), ajustar el paso basándose en la **consistencia de la dirección** a lo largo del tiempo.
* **Mecanismo:** Si el gradiente histórico (EMA en DGE) de una variable `x_i` mantiene el mismo signo consistentemente, aumentamos su LR local (aceleración). Si el signo oscila (ruido o cruce de un valle), reducimos drásticamente su LR local (frenado).
* **Inspiración:** RPROP, RMSprop, Adam.

### 1.2. Backtracking Line Search en Black-Box
* **Concepto:** Evitar el *overshooting* (empeorar la función por dar un paso demasiado grande) dividiendo dinámicamente el tamaño del paso.
* **Mecanismo:** Si al aplicar una actualización el Loss empeora, en lugar de descartar la dirección (que probablemente era buena), intentamos iterativamente con `LR/2`, `LR/4`, etc., hasta encontrar una mejora.
* **Aplicación en DGE:** Dado que DGE es *Greedy* y ya evalúa constantemente la función, aplicar una búsqueda dicotómica sobre la magnitud del paso una vez encontrada una buena dirección de bloques podría garantizar convergencia sin atascos.

### 1.3. Evolución del Direction-Consistency LR (V2)
* **Consistencia Ponderada por Magnitud (SNR-based):** En lugar de calcular una media plana de signos (`mean(sign(grad))`), medir la "confianza" utilizando el Ratio Señal-Ruido (SNR) local de la ventana de tiempo. Un pequeño pico de ruido no debería pesar lo mismo que un gradiente fuerte constante.
* **Ventanas de Tiempo Adaptativas ($T$ dinámico):** Ajustar el hiperparámetro $T$ (tamaño de la memoria histórica) dependiendo de la profundidad de la capa. Las capas finales (señal más limpia) pueden usar un $T$ bajo para converger rápido, mientras que las primeras capas (inundadas por el ruido cruzado) requerirían un $T$ alto para filtrar mejor la señal.
* **Momentum de la Consistencia:** Implementar un mecanismo de penalización (decay) para variables que demuestren ser "crónicamente ruidosas" durante cientos de pasos, congelándolas agresivamente para ahorrar perturbaciones.

---

## 2. Variables Oscilantes y Estimación por Frecuencia

### 2.1. "Snapshots" sobre Direcciones Oscilantes
* **Concepto original:** En lugar de tener un valor estático para cada peso, hacer que el peso oscile cíclicamente alrededor de su centro: `x = x_centro + amplitud * sin(t)`.
* **Variante Multiplicativa:** La amplitud de la oscilación puede ser proporcional al valor del peso (`amplitud = x_centro / 2`), permitiendo exploración agresiva en valores altos y convergencia fina cerca de cero.
* **Mecanismo de "Snapshot":** Si durante esta oscilación continua de las variables se detecta una mejora en la función objetivo, se toma una "foto" (snapshot) del estado exacto de las variables en ese instante $t$ y se actualizan los centros hacia esos valores.

### 2.2. DGE de Frecuencias (Estimación Analítica por Espectro)
* **Concepto Avanzado:** Asignar a cada dimensión $D$ una frecuencia de oscilación única y ortogonal ($\omega_i$). 
* **Mecanismo:** Al evaluar la función objetivo continua (el Loss) a lo largo del tiempo $t$, la señal resultante será una onda compleja (la suma de las influencias de todas las variables vibrando).
* **Resolución:** Aplicar una **Transformada de Fourier** (o técnicas de *Compressed Sensing* / Transformada de Fourier Esparsa) sobre la onda del Loss para aislar qué frecuencias (variables) están contribuyendo constructivamente o destructivamente al gradiente, obteniendo el gradiente de miles de variables simultáneamente con una fracción de evaluaciones.

### 2.3. Transformada de Walsh-Hadamard (El Fourier Binario)
* **Concepto Avanzado:** La Transformada de Fourier asume ondas senoidales continuas, lo cual choca con la naturaleza discreta y basada en bloques aleatorios de DGE. La alternativa es usar **Ondas Cuadradas Binarias** ($\pm 1$) y aplicar la **Transformada de Walsh-Hadamard**.
* **Mecanismo:** Asignar a cada variable un "código de barras" binario único y ortogonal (una secuencia temporal de perturbaciones de $+1$ y $-1$). Evaluando la función a lo largo del tiempo con estas secuencias, se puede reconstruir el gradiente exacto de todas las variables con extrema eficiencia computacional.
* **Potencial Cruzado:** Esta idea es altamente transferible. Aunque difícil de integrar con el muestreo aleatorio puro de DGE, es una técnica brutal para extraer derivadas en cualquier optimizador Black-Box o incluso para mejorar algoritmos clásicos, inspirada tangencialmente en técnicas de paralelismo cuántico (ej. algoritmo de Bernstein-Vazirani).

### 2.4. Temblor Desfasado (Desynchronized Tremor) con Tiempo Discreto
* **Concepto:** En lugar de oscilar todos los pesos sincrónicamente a lo largo del vector de dirección, cada variable recibe un desfase (fase) aleatorio o independiente: `W_temp = W + (D * sin(t + phi_i))`.
* **Mecanismo:** Actúa como un enjambre ("nube de abejas") explorando la topología local alrededor del centro actual. Permite que las variables aprendan de manera independiente probando combinaciones cruzadas (ej. variable A sube temporalmente mientras B baja).
* **Desafío (Tiempo Discreto):** Dado que la optimización ocurre en iteraciones discretas ($t = 1, 2, 3...$), para dar una "vuelta" significativa a la onda y explorar el espacio de fases se requieren bastantes incrementos de $t$. Posibles soluciones incluyen usar saltos de tiempo $\Delta t$ más grandes, frecuencias diferentes por variable ($\omega_i \cdot t$), o directamente ruido aleatorio en lugar de un seno puro.
* **Variante Estocástica (Ruido Gaussiano):** Para evitar la complejidad de controlar la fase y el avance del tiempo discreto, se puede reemplazar la onda senoidal por ruido puro: `W_temp = W + (D * N(0, 1))`. Esto hace que cada variable "tiemble" aleatoriamente a lo largo de su propia dirección preferida $D$ en cada iteración, logrando el mismo efecto de exploración cruzada.

---

## 3. Topología y Correlación Espacial

### 3.1. Agrupación Vectorial y EMA Direccional (Rompiendo la Ceguera de Ejes)
* **Concepto:** En lugar de tratar cada variable $D$ como un escalar escalar independiente ($\pm 1$), agrupar las variables en bloques lógicos (ej. pares 2D, tríos 3D, o los pesos de una misma neurona en $N$-D) formando **"Super-Variables" Vectoriales**.
* **Mecanismo (Perturbación):** Al evaluar un grupo, en lugar de sumar o restar a cada escalar de forma aislada, se aplica un **vector aleatorio unitario** (una dirección en la hiperesfera $N$-dimensional).
* **Mecanismo (Acumulación):** El EMA deja de ser un escalar y pasa a ser un **Vector Promedio**. Si vectores aleatorios que apuntan "hacia el noreste" suelen mejorar la función, el EMA del grupo se alinea diagonalmente.
* **Ventaja (Correlación Natural):** Resuelve el problema del "zig-zag" en valles diagonales (como Rosenbrock) porque permite saltar directamente en la diagonal óptima, en lugar de chocar contra las paredes de los ejes cartesianos independientes. Obliga al optimizador a encontrar la dirección conjunta de un grupo de variables correlacionadas.

### 3.2. SPSA Jerárquico (Transformada de Hadamard Truncada)
* **Concepto:** Fusión de SPSA (perturbar todo a la vez), búsqueda dicotómica y Transformada de Walsh-Hadamard. En lugar de evaluar bloques aislados (donde muchas variables son 0), evaluamos **todas** las variables simultáneamente ($+1$ o $-1$), pero el patrón de signos se define jerárquicamente por recursión espacial.
* **Mecanismo (Generación de Patrones):** 
  - Nivel 0: Todas las variables $+1$ (y luego todas $-1$).
  - Nivel 1: Mitad izquierda $+1$, mitad derecha $-1$.
  - Nivel 2: Cuartos alternos ($+1, -1, +1, -1$).
  - Este proceso dibuja exactamente las filas de una **Matriz de Walsh-Hadamard** ordenadas por "frecuencia espacial" (secuencia de Walsh).
* **Beneficio Matemático ($O(\log D)$):** Si nos detenemos a una profundidad aceptable (ej. $\log_2(D)$ evaluaciones), no obtenemos el gradiente individual de cada variable, sino la **proyección del gradiente sobre las frecuencias espaciales más bajas**. Dado que los pesos contiguos en redes neuronales suelen estar correlacionados, dar un paso usando estas proyecciones "macro" mueve grandes bloques de pesos en direcciones coherentes, eliminando el ruido microscópico con un coste irrisorio.
* **Sinergia con Temporal Denoising (EMA):** Al igual que en el algoritmo base, los resultados de estas evaluaciones jerárquicas no tienen que descartarse en cada paso $t$. Como $t$ y $t+1$ no son independientes (el paisaje subyacente cambia despacio), podemos mantener un EMA de la "sensibilidad" de cada nivel de la matriz de Hadamard a lo largo del tiempo, filtrando aún más el ruido y combinando coherencia espacial (Hadamard) con estabilidad temporal (EMA).
