# Denoised Gradient Estimation (DGE)
## Theory and Analytical Support

Este documento formaliza matemáticamente el optimizador **DGE** (Denoised Gradient Estimation), proporciona su fundamento analítico y establece los límites teóricos del estimador. 

---

## 1. Definición Formal del Estimador

El objetivo de la optimización zeroth-order es minimizar una función objetivo $f: \mathbb{R}^D \rightarrow \mathbb{R}$ teniendo acceso exclusivo a evaluaciones ruidosas u oráculos de $f(x)$, sin información de $\nabla f(x)$.

### Perturbaciones de Bloque Aleatorias
DGE descompone la dimensionalidad $D$ en un hiper-parámetro $K$ (número de bloques simultáneos, o `K_BLOCKS`). En cada paso $t$:

1. Se genera un vector de perturbación $s_t \in \{-1, 1\}^D$ muestreado de una distribución de Rademacher (probabilidad $0.5$ para $1$ y $-1$).
2. Se genera una máscara binaria $M_t \in \{0, 1\}^D$ de forma rala, donde exactamente un subconjunto de tamaño $K$ variables es seleccionado uniformemente. En la práctica, el espacio se particiona dinámicamente y se muestrean múltiples sub-bloques ortogonales para asegurar cobertura.
3. Se construye el vector de prueba final: $\Delta_t = \delta \cdot (s_t \odot M_t)$, donde $\delta > 0$ es la magnitud de exploración y $\odot$ es el producto de Hadamard.

### Estimador Direccional Base ($g_t$)
El gradiente estimado bruto $g_t$ en el paso $t$ se calcula evaluando la función simétricamente a lo largo de la perturbación $\Delta_t$:

$$ g_t = \frac{f(x_t + \Delta_t) - f(x_t - \Delta_t)}{2\delta} \cdot (s_t \odot M_t) $$

Para las variables $i$ donde $M_{t,i} = 0$, el gradiente estimado es estrictamente $0$.

---

## 2. Comportamiento Esperado bajo Modelos Simplificados

Asumamos que $f$ es localmente bien aproximada por su expansión de Taylor de primer orden:

$$ f(x + \Delta) \approx f(x) + \nabla f(x)^T \Delta $$

Entonces la diferencia evaluada es:
$$ f(x + \Delta_t) - f(x - \Delta_t) \approx 2 \nabla f(x)^T \Delta_t = 2\delta \sum_{j=1}^D \nabla f(x)_j s_{t,j} M_{t,j} $$

Sustituyendo en la ecuación del estimador para una coordenada activa $i$ (donde $M_{t,i} = 1$):
$$ g_{t,i} = \left( \sum_{j=1}^D \nabla f(x)_j s_{t,j} M_{t,j} \right) s_{t,i} $$
$$ g_{t,i} = \nabla f(x)_i s_{t,i}^2 + \sum_{j \neq i} \nabla f(x)_j s_{t,j} s_{t,i} M_{t,j} $$

Como $s_{t,i} \in \{-1, 1\}$, tenemos $s_{t,i}^2 = 1$. Por lo tanto:
$$ g_{t,i} = \nabla f(x)_i + \sum_{j \neq i} \nabla f(x)_j s_{t,j} s_{t,i} M_{t,j} $$

**Expectativa:** Dado que $s_{t,j}$ y $s_{t,i}$ son variables aleatorias independientes de media cero para $j \neq i$, la expectativa del término sumatorio es $0$.
$$ \mathbb{E}[g_{t,i}] = \nabla f(x)_i $$
El estimador es un estimador insesgado del gradiente verdadero.

---

## 3. Análisis de Señal-Ruido (SNR) e Impacto Dimensional

Aunque insesgado, el problema central de los métodos tipo SPSA es la varianza catastrófica en altas dimensiones. 
La varianza condicional para la coordenada $i$ activa es:

$$ \text{Var}(g_{t,i}) = \mathbb{E}\left[ \left( \sum_{j \neq i} \nabla f(x)_j s_{t,j} s_{t,i} M_{t,j} \right)^2 \right] $$

Dado que $\mathbb{E}[s_{t,j} s_{t,k}] = 0$ para $j \neq k$, la varianza colapsa a la suma de los cuadrados:
$$ \text{Var}(g_{t,i}) = \sum_{j \neq i} (\nabla f(x)_j)^2 M_{t,j} $$

Aquí reside la innovación crítica de DGE:
* En **SPSA puro**, $M_t = \mathbf{1}$ (se perturban las $D$ dimensiones). La varianza es $\approx ||\nabla f(x)||^2$, que crece linealmente con $D$. En redes neuronales con $D > 10^5$, el ruido inunda por completo la señal ($\text{SNR} \rightarrow 0$).
* En **DGE**, $M_t$ limita el número de dimensiones activas concurrentes a $K_{blocks} \ll D$. La varianza se reduce proporcionalmente a la fracción de densidad de la máscara:
  $$ \text{Var}_{DGE}(g_{t,i}) \approx \frac{K_{blocks}}{D} ||\nabla f(x)||^2 $$

DGE comercia tiempo (requiere más iteraciones para muestrear todo el espacio) a cambio de una **Varianza Exponencialmente Menor**, permitiendo convergencia donde SPSA diverge.

---

## 4. Temporal Aggregation (El Secreto del Éxito)

La reducción de varianza por bloques no es suficiente para entrenar redes de 100K parámetros si actualizamos los pesos directamente (SDG). DGE inyecta los gradientes ralos y ruidosos en un filtro de media móvil exponencial (EMA), análogo al primer momento de Adam:

$$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t $$

### Matemáticas de la Aniquilación del Ruido
Expandiendo la recursión (asumiendo que el gradiente verdadero local $\nabla f(x)$ es aproximadamente estacionario en la ventana de tiempo $\tau = \frac{1}{1-\beta_1}$):

$$ m_t \approx \mathbb{E}[M_t] \nabla f(x) + \text{Ruido}(\mu=0, \sigma^2 \rightarrow 0) $$

El filtro EMA acumula de forma constructiva la señal de gradiente constante $\nabla f(x)_i$, pero acumula de forma *destructiva* el ruido cruzado $\pm \nabla f(x)_j$, ya que su signo alterna caóticamente en cada evaluación. Como resultado, DGE *desentierra* el gradiente real del ruido masivo.

---

## 5. Direction-Consistency Learning Rates (v27)

El hallazgo empírico más importante de DGE es que, en paisajes altamente patológicos (Rosenbrock, Ellipsoid), el EMA por sí solo no previene que el optimizador de pasos fatales motivados por ráfagas de ruido anómalo.
Para solucionar esto, DGE introduce un coeficiente de confianza local $C_t \in [0, 1]^D$:

$$ C_{t,i} = \left| \frac{1}{T} \sum_{k=0}^{T-1} \text{sgn}(g_{t-k, i}) \right| $$

El paso de actualización se redefine como:
$$ \theta_{t+1} = \theta_t - \alpha \cdot C_t \odot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

### Justificación Teórica
Si una variable $i$ está dominada por ruido de alta varianza, la probabilidad de observar un signo positivo es $0.5$. La suma acumulada describe un *Random Walk* que converge a $0$ por la Ley de los Grandes Números, haciendo que $C_{t,i} \rightarrow 0$ y deteniendo la oscilación térmica.
Si la variable tiene una fuerte señal de gradiente latente $\nabla f(x)_i$, la probabilidad del signo es asimétrica y $C_{t,i} \rightarrow 1$.
Esta regularización no añade coste de evaluaciones de función y es matemáticamente análoga a un mecanismo de atención temporal.

---

## 6. Análisis de Régimen de Fracaso (Limitaciones)

A pesar de su robustez, DGE tiene límites teóricos y empíricos estrictos.

### A. Deep Sign Networks y el Cuello de Botella Cuántico
En la Fase 6 empírica (Redes `INT4` y `Sign`), observamos que el *accuracy* tiene un techo duro (~73-78%). 
Esto ocurre por la **destrucción del supuesto de estacionariedad local**. Para que el Temporal Aggregation (EMA) aniquile el ruido, el gradiente real $\nabla f(x)$ no debe cambiar dramáticamente en $\tau$ iteraciones.
Sin embargo, en redes discontinuas profundas, el cambio de *un solo bit* en la primera capa causa un *Efecto Mariposa* que invierte el signo de miles de variables en las capas subsiguientes. El paisaje es fractal y cambia completamente a cada paso, destruyendo la premisa de estacionariedad temporal del EMA.

### B. El Compromiso de Escalabilidad SFWHT
La exploración inicial usando SFWHT (Sequency-Ordered Fast Walsh-Hadamard Transform) demostró matemáticamente que escanear un bloque $B$ cuesta $B$ evaluaciones. Funciona si la red es $K$-rala ($K \ll B$). Pero en redes neuronales densas reales, cada subespacio de parámetros contribuye a la pérdida. El escaneo falla porque cada cubo SFWHT colisiona ruido denso, invalidando el intento de compresión. DGE puro triunfó porque asume ruido denso implícito y usa el tiempo (no el espacio) para filtrarlo.
