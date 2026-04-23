# DGE Findings v52: Híbrido Estructural-Curvatura y Adam Asimétrico

## Objetivo
Evaluar dos hipótesis para mejorar el descenso por gradiente empírico en valles estrechos (inspiradas en la "Saga del Sabueso" v46-v51):
1. **Curvatura Local (Híbrido Estructural):** En lugar de buscar un vector de curvatura global en 100.000 dimensiones (como en v51), calculamos el cambio de gradiente ($\Delta G$) y su vector ortogonal de forma *independiente* para cada sub-bloque de la red neuronal.
2. **Adam Asimétrico:** Para evitar que el ruido del muestreo lateral "asuste" a Adam (inflando la varianza $v_t$ y hundiendo el Learning Rate, como vimos en v48-v50), la corrección lateral se aplica directamente a los pesos, *fuera* de la actualización de las estadísticas de Adam.

## Resultados (MNIST, 500k Evaluaciones)

| Versión | Método | Precisión Test |
|---------|--------|----------------|
| **v45** | Sub-divided Neurons (Baseline Estructural) | **91.08%** |
| **v51** | Curvature-Guided (Global) + Adam Integrado | 87.30% |
| **v52** | Híbrido Estructural-Curvatura + Adam Asimétrico | **90.23%** |

## Análisis y Conclusiones

### 1. El Adam Asimétrico funciona
La técnica de inyectar la señal lateral directamente en los pesos, saltándose los acumuladores de Adam ($m_t$, $v_t$), ha demostrado ser vital. Permite aprovechar la geometría del paisaje sin corromper la memoria del optimizador principal. Hemos recuperado la estabilidad perdida en iteraciones anteriores.

### 2. Curvatura Local > Curvatura Global
El salto del **87.30% (v51)** al **90.23% (v52)** confirma que en alta dimensión, la "curvatura" global no significa nada. Al forzar que cada grupo de parámetros busque su propia dirección ortogonal a su propio gradiente local, la señal geométrica capturada es mucho más pura y útil.

### 3. El Coste de Oportunidad (Por qué no superó a la v45)
A pesar de la notable mejora respecto a v51, la **v52 no logró superar a la v45**.
La razón principal sigue siendo el *coste de oportunidad* (o "impuesto de evaluaciones"). Cada "olfateo" lateral cuesta 2 evaluaciones de la función objetivo por paso. En la v45, esas evaluaciones extra se invierten implícitamente en dar más pasos principales en la dirección del gradiente promediado. 
En paisajes ruidosos como el de MNIST, promediar el ruido parece seguir siendo ligeramente más rentable que intentar ser "listo" con la curvatura de segundo orden.

## Siguiente Iteración (Ideas)
La "Fuerza Bruta Estadística" (agrupar neuronas y promediar, como en v45) sigue dominando, pero la inyección asimétrica ha demostrado que podemos añadir componentes sin romper Adam.

Posibles vías:
1. **Búsqueda de Línea Frontal (Foresight):** En lugar de gastar las 2 evaluaciones extra en explorar los lados (ortogonal), gastarlas en un *paso hacia adelante* en la dirección sugerida por Adam. Si el paso empeora el Loss bruscamente (chocamos contra la pared), activamos un *backtracking* local.
2. **Asignación Dinámica de Presupuesto:** ¿Por qué olfatear en todos los bloques estructurales por igual? Medir la varianza del gradiente de cada bloque y solo aplicar la técnica de curvatura a los bloques que muestran un "valle estrecho" (varianza direccional alta).