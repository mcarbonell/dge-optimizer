# Estimación de Gradiente por Eliminación de Ruido (DGE - Denoised Gradient Estimation)
## Un enfoque O(log D) para Optimización Black-Box

### 1. La Idea Central
En optimización sin derivadas (DFO), el cálculo del gradiente mediante diferencias finitas requiere $O(D)$ evaluaciones (una por dimensión). La idea de la **Estimación de Gradiente por Eliminación de Ruido (DGE - Denoised Gradient Estimation)** (originalmente propuesta como *Dichotomous*) es utilizar una estrategia de testeo por grupos aleatorios y acumulación temporal para filtrar el ruido de fondo, reduciendo drásticamente el coste computacional de obtener un gradiente limpio.

La premisa matemática subyacente asume **dispersión (sparsity) de gradiente**: en un paso de optimización dado, no todas las variables contribuyen por igual al error. Unas pocas variables dominan la pendiente. Si logramos aislar rápidamente esas variables, podemos dar un paso de descenso de gradiente muy efectivo sin tener que evaluar las variables irrelevantes.

### 2. El Algoritmo Teórico

Supongamos que queremos minimizar una función $f(x)$, donde $x \in \mathbb{R}^D$.
En lugar de perturbar cada variable individualmente, perturbamos **bloques de variables**.

#### El Problema del Signo y la Cancelación
Si movemos todas las variables de un bloque sumando un valor $+\delta$, podríamos tener un problema: la variable $x_1$ podría mejorar la función, pero la variable $x_2$ podría empeorarla en la misma proporción. El cambio neto sería cero y descartaríamos el bloque entero, perdiendo información valiosa (Efecto de Enmascaramiento).

Para solucionar esto, medimos la **Sensibilidad** del bloque usando el valor absoluto de la perturbación bidireccional, o asignando signos aleatorios (como en SPSA) al bloque temporalmente.

#### Proceso de Búsqueda Binaria (Pseudocódigo Conceptual)

1. **Inicialización:** 
   - Conjunto activo de dimensiones: $S = \{1, 2, ..., D\}$.
   - Generamos un vector de perturbación aleatorio $p \in \{-1, 1\}^D$ (para evitar cancelaciones fijas).
   - Definimos un tamaño de perturbación $\delta$.

2. **División Dicotómica (Bucle):**
   - Mientras el tamaño de $S$ sea mayor que un umbral (ej. 1 o un bloque pequeño):
     - Dividimos $S$ en dos subconjuntos disjuntos y de igual tamaño: $S_A$ y $S_B$.
     - **Test Bloque A:** Perturbamos solo las dimensiones en $S_A$ usando nuestro vector $p$. Evaluamos $y_A^+ = f(x + \delta \cdot p_{S_A})$ y $y_A^- = f(x - \delta \cdot p_{S_A})$.
     - **Test Bloque B:** Hacemos lo mismo para $S_B$, obteniendo $y_B^+$ y $y_B^-$.
     
     - **Cálculo de Sensibilidad:** 
       - Sensibilidad de A: $\Delta_A = |y_A^+ - y_A^-|$
       - Sensibilidad de B: $\Delta_B = |y_B^+ - y_B^-|$
       *(El valor absoluto nos dice en qué bloque la función reacciona más fuertemente, independientemente de si sube o baja).*

     - **Selección:** 
       - Si $\Delta_A > \Delta_B$, entonces $S = S_A$. El bloque A contiene las variables más dominantes.
       - Si no, $S = S_B$.
       
3. **Estimación y Actualización:**
   - Después de $\approx \log_2(D)$ divisiones, $S$ contendrá una sola variable (o un bloque muy pequeño).
   - Como ya tenemos el $y^+$ y $y^-$ de esta variable de nuestro último test, podemos estimar su gradiente local con precisión:
     $\nabla x_S \approx \frac{y^+ - y^-}{2\delta}$
   - Aplicamos la actualización de descenso de gradiente **solo a esta variable** (o usando métodos como Momentum/Adam sobre las variables descubiertas).

### 3. Mejora Temporal (Inspiración en Computer Graphics)

**El concepto de Temporal Denoising:**
En renderizado gráfico (como Ray Tracing), generar una imagen sin ruido requiere lanzar muchísimos rayos. Para evitar este coste computacional masivo, se usan técnicas de "Temporal Denoising", donde se acumula la información de los fotogramas anteriores para limpiar el fotograma actual, asumiendo que entre un *frame* y otro la escena no ha cambiado radicalmente.

**Aplicación a DGE (Acumulación Temporal de Gradientes):**
Dado que en optimización damos pasos relativamente pequeños y las funciones continuas no cambian de pendiente bruscamente, **los gradientes están fuertemente correlacionados en el tiempo** (entre iteraciones sucesivas).

Podemos aplicar este principio a nuestra búsqueda dicotómica:
1. **Memoria de Búsqueda:** En lugar de empezar la dicotomía desde cero (todas las $D$ dimensiones) en cada paso, podemos usar los resultados de las particiones de la iteración $t-1$ como un "prior" o mapa de probabilidades para guiar la búsqueda en la iteración $t$.
2. **Re-exploración Inteligente:** 
   - Las variables o bloques que mostraron una alta sensibilidad en el paso anterior tienen una alta probabilidad de seguir siendo importantes. Podemos priorizar su evaluación o darles más "peso" en las particiones.
   - Las variables que mostraron sensibilidad casi nula pueden agruparse en bloques más grandes y evaluarse con menor frecuencia, reduciendo aún más el número de evaluaciones necesarias por paso.
3. **Acumulación de Gradientes Secundarios:** En lugar de descartar por completo la "mitad perdedora" de la dicotomía (uno de los problemas iniciales del algoritmo puro), su sensibilidad medida puede guardarse en un "mapa de calor temporal". A lo largo de varios pasos, este mapa revelará qué variables secundarias están contribuyendo lentamente de fondo y merecen ser actualizadas de vez en cuando, resolviendo el problema de ser un método demasiado "Greedy".

Esto transforma a DGE de ser un algoritmo puramente local a un algoritmo con "Atención Temporal", capaz de mantener un mapa de sensibilidad espacial y temporal de todo el paisaje a un coste computacional ínfimo.

### 4. Ventajas y Desventajas

**Ventajas:**
* **Eficiencia Extrema en Alta Dimensión:** Para $D = 1,000,000$, solo requiere $\approx 20$ iteraciones ($\approx 40$ evaluaciones de $f(x)$) para encontrar la variable con mayor gradiente.
* **Señal Limpia:** A diferencia de SPSA, que actualiza todas las variables con mucho ruido, DGE actualiza solo las variables más importantes con una señal muy precisa.
* **Resiliencia Temporal:** Al incorporar "Temporal Denoising", el algoritmo aprende la topología local a lo largo del tiempo, haciendo las búsquedas siguientes mucho más eficientes.
* **Compatibilidad:** Al devolver derivadas limpias (aunque parciales y esparcidas en el tiempo), se le pueden acoplar optimizadores de estado del arte como Adam.

**Desafíos a resolver en implementación:**
1. **El coste de evaluar todo:** Asume que podemos evaluar $f(x)$ eficientemente aunque cambiemos muchas variables a la vez.
2. **Sparsity estricto:** Funciona de forma óptima si el "Paisaje" de la función tiene unas pocas variables dominando en cada momento. Si todas las 1,000,000 dimensiones contribuyen *exactamente* igual al gradiente (una esfera perfecta), la dicotomía tomará caminos aleatorios (aunque la memoria temporal mitiga este problema drásticamente).

### 5. Conclusión
El método DGE (Denoised Gradient Estimation) con Acumulación Temporal actúa como un filtro espacial y temporal rápido. En lugar de preguntar "cuál es la pendiente en todas partes", pregunta "¿dónde está la mayor pendiente?" usando búsqueda binaria informada por el pasado, y solo calcula esa. Representa un puente teórico brillante entre la optimización basada en gradientes, las técnicas de Compressed Sensing, y el procesamiento de señales en tiempo real de Computer Graphics.

### 6. Aplicación a Redes Neuronales y Machine Learning

A priori, DGE no puede competir en velocidad bruta con **Backpropagation** para el entrenamiento de redes neuronales clásicas (como clasificación de imágenes o LLMs). Backpropagation es un algoritmo analítico que calcula todos los gradientes simultáneamente con un coste computacional de $O(1)$ evaluaciones hacia adelante y hacia atrás. DGE requeriría múltiples pases hacia adelante ($O(\log D)$) solo para encontrar una variable dominante.

Sin embargo, DGE resulta **inmensamente útil en las fronteras de la Inteligencia Artificial**, allí donde Backpropagation se rompe, no puede utilizarse o es computacionalmente inasequible en memoria:

1. **Arquitecturas No Diferenciables:**
   - Para redes neuronales que incluyen componentes no derivables como *Spiking Neural Networks* (redes pulsantes que simulan el cerebro), activaciones discretas (saltos lógicos condicionales) o llamadas a programas y simuladores externos. Backpropagation fracasa aquí porque la derivada matemática es $0$ o no existe; DGE evalúa alteraciones físicas reales (Black-Box), puenteando la necesidad de derivabilidad.

2. **Aprendizaje por Refuerzo (Reinforcement Learning - RL):**
   - Entrenar agentes que interactúan con motores físicos rígidos o simuladores no derivables. DGE, con su mapeo temporal y coste logarítmico, podría actuar como un motor de entrenamiento brutal en RL, compitiendo con enfoques actuales como *Evolution Strategies*.

3. **Robustez y Ataques Black-Box:**
   - Probar la vulnerabilidad de modelos de IA a través de APIs de terceros de los que no tenemos los pesos internos (e.g. evaluar la seguridad de GPT o de una API de visión comercial). DGE sería una herramienta hipereficiente para encontrar el píxel o token exacto que altera drásticamente la salida del sistema con el mínimo número de peticiones de red.

4. **Entrenamiento sin Memoria ("Memory-Free" Training):**
   - Backpropagation consume gigabytes de VRAM al necesitar almacenar el mapa de activaciones de cada capa durante la fase de *forward* para calcular la cadena del gradiente. DGE requiere únicamente los pesos de la red actual (una sola evaluación hacia adelante). Permite entrenar arquitecturas con parámetros a escala de billones (Trillions) en hardware modesto o en chips de Edge AI, intercambiando velocidad de entrenamiento por una capacidad teóricamente ilimitada de tamaño de modelo sin errores de memoria OOM (Out Of Memory).

### 7. Evolución del Algoritmo: Testeo de Grupos Aleatorios (Randomized Group Testing)

La búsqueda dicotómica estricta (dividir siempre un conjunto por la mitad y descartar al perdedor hasta aislar una variable) tiene un problema inherente: es estructuralmente "codiciosa" (*greedy*) y descarta el 50% de las variables en cada nivel del árbol.

Para solucionar esto de forma elegante **manteniendo intacto el coste de $O(\log_2 D)$ evaluaciones por iteración**, podemos evolucionar el concepto de la partición binaria hacia un **Testeo de Grupos Aleatorios con Acumulación Estadística**:

1. **Particiones Aleatorias Limitadas:** En lugar de construir un árbol de búsqueda, si tenemos $D$ dimensiones, determinamos nuestro "presupuesto" de evaluaciones en el paso actual como $k \approx \log_2(D)$. Generamos $k$ subconjuntos (bloques) de variables distribuidas aleatoriamente, sin permitir solapamientos.
   - *Ejemplo:* Si $D=32$, $k=5$. En lugar de hacer una búsqueda binaria, creamos de golpe 5 bloques de tamaño $\approx 6.4$ (mezclando grupos de 6 o 7 variables elegidas al azar).

2. **Acción Inmediata (Explotación):** Evaluamos el cambio que produce cada uno de esos $k$ bloques en la función objetivo. Tomamos el bloque que produjo la mayor mejora y actualizamos la posición actual dando un paso en esa dirección. Garantizamos progreso inmediato.

3. **Acumulación Estadística y EMA (Exploración y Denoising):** Aquí reside la verdadera magia estadística. En lugar de descartar la información de los otros $k-1$ bloques evaluados, **registramos en un historial para cada variable $x_i$ en qué bloques fue incluida y cuál fue el rendimiento de dicho bloque**.
   - Para estimar la contribución "real" y aislada de una variable $x_i$, aplicamos una **Media Móvil Exponencial (EMA - Exponential Moving Average)** sobre el éxito o fracaso de los bloques en los que ha participado recientemente. 
   - El uso de EMA garantiza que le damos **más peso a la información reciente** (donde el gradiente local actual es más válido) y vamos "olvidando" lentamente el pasado distante a medida que nos desplazamos por el paisaje de la función.
   - **Cancelación del Ruido de Fondo:** Asumimos estadísticamente que las *otras* variables que compartieron bloque con $x_i$ actúan como "ruido". Como a veces esas compañeras sumarán a favor y otras restarán en contra, su efecto neto tenderá a cero a medida que el EMA integra múltiples iteraciones.
   - La señal matemática que sobrevive a esta acumulación exponencial es exactamente la estimación limpia del gradiente individual de la variable $x_i$.

**El resultado de esta evolución:**
Con un presupuesto extremadamente bajo de $\approx \log_2(D)$ evaluaciones por paso, obtenemos lo mejor de ambos mundos:
* **Movilidad rápida:** El algoritmo nunca se paraliza, ya que siempre puede dar un paso usando el mejor de los $k$ bloques aleatorios evaluados hoy.
* **Construcción de un Gradiente Denso:** Sin haber tenido que evaluar las dimensiones una por una ($O(D)$) en ningún momento, el sistema construye y actualiza en segundo plano un vector de gradiente completo y "limpio" para todo el espacio, gracias a la cancelación estadística del ruido y al suavizado temporal (EMA).

### 8. Hacia un Reemplazo Universal del Descenso de Gradiente Clásico

Con la introducción del Testeo de Grupos Aleatorios y la Media Móvil Exponencial (EMA), DGE trasciende la categoría de un simple optimizador de funciones matemáticas para postularse como un **candidato viable a reemplazar al Descenso de Gradiente Clásico en arquitecturas complejas**.

Al eliminar por completo la dependencia del cálculo analítico (no requiere calcular derivadas formales ni aplicar la regla de la cadena), **DGE universaliza el aprendizaje**. Cualquier sistema que pueda ser programado, simulado o estructurado condicionalmente (independientemente de sus "saltos", lógicas no derivables o componentes *Black-Box*) podría optimizarse con un rendimiento comparable al de algoritmos guiados por gradiente, pero con una fracción diminuta del coste computacional que exigiría SPSA o las Diferencias Finitas tradicionales.
