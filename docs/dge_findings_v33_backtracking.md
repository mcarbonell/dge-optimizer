# DGE Findings v33: Backtracking Line Search

## Objetivo del Experimento
Probar la hipótesis planteada en el documento de `brainstorming.md` sobre el uso de **Backtracking Line Search** en DGE. La idea consistía en evaluar dinámicamente varias magnitudes del paso de actualización (ej. `[2.0, 1.0, 0.5, 0.25, 0.125, 0.0]`) sobre la dirección calculada y seleccionar aquella que minimizara la pérdida en el minibatch actual. El objetivo teórico era evitar el *overshooting* y acelerar la convergencia permitiendo pasos más largos cuando fuera seguro.

## Configuración
- **Dataset:** MNIST Completo (60K train / 10K test)
- **Presupuesto:** 1,000,000 de evaluaciones (aprox. 1 min por run).
- **Semillas:** 42, 43, 44.
- **Variantes Comparadas:**
  1. `DGE_Baseline`: DGE nativo por lotes (TorchDGEOptimizer) con paso fijo dictado por Adam EMA y Cosine Annealing.
  2. `DGE_Backtracking`: Variación que prueba múltiples magnitudes de paso y elige la óptima para el minibatch actual.

## Resultados Empíricos

| Método | Test Acc (Best) |
|---|---|
| DGE_Baseline | 91.08% ± 0.58% |
| DGE_Backtracking | 87.39% ± 0.15% |

Contrario a la hipótesis inicial, **el Backtracking Line Search degradó significativamente el rendimiento del optimizador** (una pérdida de casi 4 puntos porcentuales de accuracy).

## Análisis y Conclusión
El fracaso del backtracking en este contexto se debe a un fenómeno clásico en la optimización estocástica: **el sobreajuste al ruido del minibatch (Minibatch Overfitting)**.

1. **Sesgo de Selección (Selection Bias):** Al evaluar la función de pérdida a lo largo de la línea de búsqueda utilizando *el mismo minibatch* ruidoso, el algoritmo elige sistemáticamente la magnitud del paso que sobreoptimiza el ruido particular de esas muestras. 
2. **Pérdida de Efecto Filtro:** En métodos estocásticos (como SGD o Adam), un tamaño de paso (learning rate) fijo y relativamente pequeño actúa como un filtro paso bajo, promediando el ruido entre distintos minibatches a lo largo del tiempo. Al intentar hacer una "búsqueda exacta" en un paisaje estocástico, el optimizador es engañado por el ruido y a menudo rechaza direcciones que son globalmente buenas pero que casualmente evalúan mal en el minibatch actual (o acepta pasos gigantescos hacia mínimos espurios del minibatch).
3. **Interferencia con Adam:** El estado interno de Adam (EMA de gradientes y varianzas) asume una trayectoria suave. Alterar drásticamente el tamaño del paso en cada iteración rompe la calibración de estos momentos exponenciales.

**Conclusión:** La idea de búsqueda dicotómica/backtracking para el tamaño de paso en DGE debe ser descartada cuando se opera con minibatches pequeños/ruidosos. DGE ya es suficientemente codicioso en su estimación de gradiente; intentar ser codicioso también en el tamaño del paso conduce a un rendimiento subóptimo.
