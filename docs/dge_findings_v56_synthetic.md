# DGE Findings v56: Synthetic Landscapes & The Fall of the Canine

## Objetivo
Después de comprobar en los experimentos v52-v55 que las técnicas geométricas ("Canine Sniffing") no lograban superar a la simple reducción de varianza (Dynamic Budget) en el entrenamiento de redes neuronales sobre MNIST, surgió una duda razonable:
**¿Es culpa de la técnica geométrica, o es culpa de MNIST?**

MNIST (al igual que cualquier problema de Deep Learning con SGD) está dominado por el **ruido estocástico** de los mini-batches y la varianza intrínseca de estimar gradientes en dimensiones altísimas con perturbaciones aleatorias.
Para aislar el efecto de la geometría, decidimos enfrentar al DGE Estándar contra el "Curvature Canine (v52)" en **funciones sintéticas deterministas (Rosenbrock y Ellipsoid)**. Estos paisajes matemáticos clásicos no tienen ruido de batch; son problemas puramente geométricos diseñados explícitamente con valles estrechos, curvados y mal condicionados.

## Resultados (D=1000, 50k Evaluaciones, LR = 0.05)

| Función | Algoritmo | Mejor Loss (Más bajo es mejor) | Estado Final |
|---------|-----------|--------------------------------|--------------|
| **Rosenbrock** | Baseline DGE | **1,678.66** | Estable (Lento) |
| **Rosenbrock** | Curvature Canine (v52) | 37,345.44 | **Explosión Numérica ($10^{20}$)** |
| **Ellipsoid** | Baseline DGE | **720.16** | Estable |
| **Ellipsoid** | Curvature Canine (v52) | 250,867.18 | **Explosión Numérica ($10^{14}$)** |

## Análisis de la Catástrofe
Los resultados son demoledores. Lejos de brillar en los problemas puramente geométricos, **el olfateo de curvatura explotó casi inmediatamente.**

### ¿Por qué explotó?
1. **Mal Condicionamiento Extremo:** En funciones como Rosenbrock, el gradiente principal cambia drásticamente en magnitud. Si intentamos ortogonalizar el vector de cambio de gradiente ($\Delta G$) para forzar un "paso lateral", el vector resultante suele apuntar hacia las paredes verticales del valle.
2. **Amplificación del Error:** Al dar un paso ortogonal ciego e inyectarlo asimétricamente en los pesos (incluso escalado), nos salimos bruscamente de la región segura de la función. Al salirnos, el siguiente gradiente calculado es astronómico, lo que se retroalimenta creando un ciclo de explosión incontrolable.
3. **El Baseline DGE sobrevive gracias a Adam:** El DGE estándar calcula su estimación ruidosa y se la entrega a la memoria estable de Adam. Adam utiliza sus promedios móviles ($m_t$, $v_t$) como amortiguadores gigantes que absorben los picos del valle de Rosenbrock y logran ir descendiendo, aunque sea lentamente.

## Conclusión Definitiva (El Fin del Sabueso)
Este experimento cierra definitivamente el debate sobre la exploración ortogonal ciega (Canine Sniffing) en Zero-Order:
- **En paisajes ruidosos (Deep Learning):** El "olfateo" lateral no puede competir contra la asignación de evaluaciones para promediar la varianza.
- **En paisajes geométricos (Rosenbrock):** El "olfateo" lateral es inestable y suicida frente a gradientes mal condicionados.

Toda la evidencia matemática y empírica apunta en la misma dirección: la verdadera ventaja competitiva de un optimizador DGE reside en la **estabilidad de sus estadísticos (EMA) y en la reducción inteligente de varianza (Dynamic Budget)**. Las heurísticas de segundo orden espaciales son demasiado frágiles para ser el núcleo de un optimizador robusto.