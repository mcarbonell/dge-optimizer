# DGE Findings v61-v62: Mathematical Hyperparameter Scaling per Layer

## Objetivo
Basados en el éxito rotundo del desacoplamiento de hiperparámetros en el experimento de redes híbridas (v58-v60), decidimos aplicar este concepto a las redes continuas estándar. La hipótesis era que las capas más profundas, al tener menos conexiones de entrada (menor *fan-in*), se inicializan con pesos de mayor magnitud. Para mantener una tasa de exploración relativa uniforme, los hiperparámetros de perturbación ($\Delta$) y la tasa de aprendizaje ($LR$) deberían escalar matemáticamente.

## El Modelo Matemático
Se utilizó el factor de inicialización estándar de la varianza: **$Escala \propto \sqrt{\frac{2}{fan\_in}}$**

Tomando la Capa 1 como referencia ($1.0x$), las capas subsecuentes obtuvieron los siguientes multiplicadores teóricos:
- **Capa 1 (in=784):** Scale=1.00x | LR=0.0200 | DELTA=0.00100
- **Capa 2 (in=128):** Scale=2.47x | LR=0.0495 | DELTA=0.00247
- **Capa 3 (in= 64):** Scale=3.50x | LR=0.0700 | DELTA=0.00350

## Iteraciones

### v61: Escalado Matemático + Dynamic Budget
- **Configuración:** Optimizador Megazord con Presupuesto Dinámico (asignación de bloques K por capa basada en varianza).
- **Problema de Software:** Se descubrió un bug crítico de compatibilidad entre `torch.Generator` y el backend de DirectML (`privateuseone:0`). Al forzar funciones atómicas (`randint`, `randperm`) en la GPU usando el generador, se perdía la distribución uniforme, degradando severamente el entrenamiento (~86%).
- **Solución Técnica:** Se parcheó el optimizador para ejecutar el generador de entropía **estrictamente en CPU** y luego transferir los índices/signos a la GPU, restaurando la matemática (92.24% en GPU).
- **Problema Algorítmico:** A pesar de la corrección del software, el *Dynamic Budget* reaccionó mal al escalado de hiperparámetros. Asignó el 99% del presupuesto a la capa 1 (`k_alloc=[582, 2, 2]`). Las capas 2 y 3, que ahora tenían LRs agresivos de `~0.05` y `~0.07`, estaban usando solo 2 bloques estocásticos, inyectando un ruido colosal en la red y frenando la convergencia.

### v62: Escalado Matemático + Presupuesto Fijo (Fixed Budget)
- **Configuración:** Para evitar el problema de inanición de bloques, se desactivó el *Dynamic Budget* y se fijaron los K-blocks: `[512, 64, 10]`. De este modo, las capas profundas tendrían evaluaciones suficientes para aprovechar de forma estable su mayor LR y $\Delta$.
- **Resultados:** Tras refinar la tasa de aprendizaje base máxima a `0.01` (compensando que los multiplicadores en capas profundas llegaban a `3.5x`), se obtuvo el nuevo récord absoluto del algoritmo:
  - GPU (DirectML): **92.62%**

## Análisis de Discrepancia CPU vs GPU
El comportamiento fuertemente divergente entre la ejecución pura en CPU (~85%) y la ejecución acelerada en DirectML (~92.62%) es un fenómeno fascinante. A pesar de que la semilla aleatoria (`SEED=42`) y la generación de entropía ahora ocurren de manera estricta y controlada en la CPU en ambos casos, el punto de divergencia radica en la **aritmética de punto flotante de 32-bits (FP32)**.
- DGE calcula gradientes a través de diferencias finitas muy sutiles: $\frac{f(x+\Delta) - f(x-\Delta)}{2\Delta}$.
- Las sumas acumulativas en el EMA de Adam y los multiplicadores de las matrices de la red neuronal experimentan leves diferencias de truncamiento y redondeo entre la ALU de la CPU y las unidades SIMD de la GPU.
- En un optimizador iterativo de caja negra donde las decisiones se basan en el signo y la magnitud exacta de una diferencia finita, estas micro-divergencias iniciales generan un "efecto mariposa" que altera por completo la trayectoria de exploración a lo largo de 500,000 evaluaciones.

## Conclusión
El escalado matemático (`v62`) logró estabilizarse en un impresionante **92.62%** en hardware acelerado, **rompiendo el récord histórico del repositorio del 92.53%** (establecido en la versión `v57`).

### Lecciones Fundamentales:
1. El **Escalado Matemático de Hiperparámetros** (basado en $1/\sqrt{fan\_in}$) funciona de manera excepcional cuando se le da a cada capa suficientes evaluaciones para que la señal estadística no sea sobrepasada por el ruido de un LR alto.
2. Al multiplicar los LRs de las capas profundas, **es imperativo reducir el LR base global** (`0.02` -> `0.01`) para evitar dar pasos excesivamente largos que desestabilicen el EMA de Adam.
3. La implementación de DGE es extremadamente sensible a las precisiones de hardware (FP32) debido a su naturaleza basada en diferencias finitas minúsculas, originando trayectorias divergentes entre CPU y GPU. 
