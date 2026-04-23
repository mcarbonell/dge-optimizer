# DGE Findings v46-v51: La Saga del Sabueso (Canine Sniffing)

## Objetivo
Explorar la hipótesis de que un "olfato estéreo" (muestreo lateral) y un "movimiento de cabeza" (barrido ortogonal) pueden ayudar al optimizador a localizar y seguir el centro de valles estrechos en paisajes de alta dimensión.

## Resumen de Experimentos

| Versión | Nombre | Mecanismo | Resultado (MNIST) |
|---------|--------|-----------|-------------------|
| **v45** | Sub-divided Neurons | Estructural puro (Baseline actual) | **91.08%** |
| **v46** | Canine Sniffing | Barrido lateral aleatorio post-paso | 90.94% |
| **v47** | Competitive Sniffing | Selección de la mejor fosa nasal pre-paso | 90.99% |
| **v48** | Integrated Adam | Olfato mezclado en m y v de Adam | 82.07% (Fallo) |
| **v49** | Momentum-Only | Olfato solo en m de Adam | 9.80% (Explosión) |
| **v50** | Normalized Sniff | Olfato escalado al 10% del paso principal | 83.06% |
| **v51** | Curvature-Guided | Olfato guiado por la Δ-Gradiente (Giro) | 87.30% |

## Hallazgos Clave

### 1. El Conflicto con Adam (v48-v50)
Integrar señales de muestreo lateral en la estadística de Adam es peligroso. El ruido de una sola dirección perpendicular, aunque se normalice, tiende a inflar el segundo momento ($v_t$), lo que hace que Adam colapse el Learning Rate al interpretar que el paisaje es extremadamente ruidoso. El perro "se asusta" del ruido lateral y deja de caminar.

### 2. La Curvatura como Guía (v51)
El salto del **83.06% (v50)** al **87.30% (v51)** es el hallazgo más potente de esta saga. Demuestra que existe una señal de curvatura útil en el paisaje de MNIST. Al alinear el "olfateo" con la dirección en la que el gradiente está girando, el optimizador captura información geométrica real.

### 3. La Superioridad de la "Fuerza Bruta Estadística" (v45)
A pesar de la elegancia biológica del olfato canino, el **v45 (Sub-divided Neurons)** sigue siendo el rey. 
- **Razón:** En 100.000 dimensiones, es más rentable dedicar el presupuesto de evaluaciones a promediar mejor el gradiente de todos los cables (bloques estructurales) que a intentar adivinar la curvatura con un solo vector lateral ruidoso.

## Conclusión
La analogía del sabueso es excelente para entender la **geometría del descenso**, pero en términos de eficiencia computacional, el **DGE Estructural Puro** es más robusto. El "olfato" funciona, pero el "impuesto" de evaluaciones extra no compensa la ganancia en precisión en este benchmark.
