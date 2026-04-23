# DGE Findings v63: Multiplicative Structural DGE (Node Perturbation)

## Objetivo del Experimento
Revisitar la idea original de optimización topológica o estructural (DGE *v36*), la cual se asemeja al concepto de **"Node Perturbation"** utilizado en el diseño de hardware neuromórfico y redes neuronales analógicas (VLSI). 

En lugar de perturbar bloques aleatorios de parámetros sin tener en cuenta su función, los parámetros se agrupan en bloques que representan la topología de la red:
1. **Bloques Fan-In:** Todos los pesos (cables) que entran a una neurona específica.
2. **Bloques Fan-Out:** Todos los pesos (cables) que salen de una neurona específica.
3. **Bloques Bias:** Sesgos individuales.

### El Giro (El Problema del v36)
En la versión `v36` se intentó perturbar sumando un escalar global a todo el bloque ($W_{nuevo} = W + \Delta$). Esto causó un efecto destructivo conocido como **"Lavado de Gradiente" (*Gradient Washing*)**, donde los pesos pequeños con poca activación eran inundados por el ruido global de la suma de todo el bloque. La red falló miserablemente, logrando solo ~40.16% de precisión tras 2.5M de evaluaciones.

### La Hipótesis (v63)
¿Qué pasa si en lugar de sumar un $\Delta$ fijo, perturbamos multiplicativamente por $(1 \pm \Delta)$?
Al hacer **$W_{nuevo} = W \times (1 \pm \Delta)$**, la perturbación efectiva depende de la magnitud individual de cada cable. Los cables "muertos" casi no se ven afectados, y los cables fuertes envían una señal clara. El gradiente escalar del bloque se aplica luego de vuelta escalado por el peso original.

## Resultados Empíricos (2.5M Evals)
- **v36 (Aditiva):** 40.16%
- **v63 (Multiplicativa):** **78.63%**

## Conclusión y Análisis
¡La hipótesis era matemáticamente correcta! Al cambiar la perturbación estructural de una suma global a un factor multiplicativo, hemos logrado casi duplicar la precisión del modelo (de 40% a ~79%).

1. **Resolución del Lavado de Gradiente:** La perturbación multiplicativa preserva la ortogonalidad y la jerarquía interna de los pesos. Un peso de $0.001$ recibe un $\Delta$ efectivo de $0.000001$, mientras que un peso de $2.0$ recibe un $\Delta$ efectivo de $0.002$. La señal ya no se ahoga bajo el ruido uniforme de las conexiones irrelevantes.
2. **Viabilidad en Hardware Analógico:** Este resultado es una prueba de concepto sólida de que DGE puede entrenar arquitecturas emulando restricciones de hardware (perturbar una neurona entera escalando su voltaje/ganancia) en lugar de tener que direccionar microscópicamente cada memristor o sinapsis de forma individual.
3. **Límite Teórico:** Aunque un 78.6% demuestra que el aprendizaje ocurre (superando con creces a Backprop bajo estas mismas restricciones severas de caja negra estructural), el agrupamiento forzoso sigue introduciendo un cuello de botella informacional en comparación con la exploración de bloques aleatorios ortogonales (que llega al >92%). Al final, atar cientos de pesos a un único escalar de exploración ($\Delta$ de neurona) siempre será menos expresivo que la perturbación independiente.

Este hallazgo rescata la idea de la DGE estructural y la convierte en una técnica viable para dominios donde las actualizaciones individuales de pesos no son posibles (e.g., neuromórfica o fotónica).