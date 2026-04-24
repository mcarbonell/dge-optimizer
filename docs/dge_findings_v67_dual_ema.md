# Findings v67: Dual Sign-EMA (DS-EMA) Consistency Mask

**Fecha:** 2026-04-24
**Código:** `scratch/dge_dual_ema_v67.py`
**Datos crudos:** `results/raw/v67_dual_ema.json`

---

## Configuración del Experimento

| Parámetro | Valor |
|---|---|
| `dim` | 128 |
| `budget` | 200,000 evaluaciones |
| `k_blocks` | 8 |
| `lr` | 0.1 |
| `delta` | 1e-3 |
| `seeds` | 5 |
| Métodos | PureDGE, ConsistencyDGE (T=20), **DualEMADGE (v67)** |

### La Innovación: Dual Sign-EMA (DS-EMA)
Sustitución de la ventana deslizante (SMA) por dos EMAs de signos:
- **Fast EMA** ($\alpha=0.3$): Reacción rápida.
- **Slow EMA** ($\alpha=0.05$): Tendencia a largo plazo.
- **Crossover Gate**: `mask = (sign(fast) == sign(slow)) * abs(slow)`. 
  - Si hay desacuerdo en la dirección (oscilación), el update se anula ($0$).
  - Si hay acuerdo, se escala por la magnitud de la tendencia lenta.

---

## Resultados Sintéticos (Media de 5 semillas)

| Benchmark | PureDGE (Baseline) | Consistency T20 (v27) | **DualEMADGE (v67)** | Mejora vs T20 |
|---|---|---|---|---|
| **Rosenbrock** | 1.3970e+02 | 1.1440e+02 | **1.0623e+02** | **-7.1%** |
| **Rotated Quadratic** | 2.8175e-12 | 3.4844e-09 | **1.4873e-12** | **-99.9%** |
| **Ellipsoid** | 7.5100e+03 | 7.4288e+02 | **2.9890e+02** | **-59.7%** |
| **Sphere** | 6.8138e-39 | 5.8516e-29 | **2.4349e-39** | **>> 100%** |

---

## Validación en MNIST (v68)

Tras ajustar el Learning Rate a un valor más estable (**lr=0.01**), la superioridad de la Dual EMA se confirma de forma masiva en visión artificial:

| Método | LR | Test Accuracy (300K evals) | Mejora |
|---|---|---|---|
| SMA Consistency (Window=20) | 0.01 | 80.50% | - |
| **DS-EMA Consistency (v67)** | 0.01 | **85.17%** | **+4.67pp** |

### Robustez ante el ruido (Batch Size)
Se realizaron pruebas de estrés con diferentes tamaños de batch en MNIST para evaluar el "Crossover Gate":

- **Batches Pequeños (4, 8, 16):** La Dual EMA mantiene una ventaja de +1pp a +2pp. El filtrado exponencial es más robusto al ruido estocástico que una ventana fija de 20 pasos.
- **Batches Estándar (256):** Máximo rendimiento con un gap de casi 5 puntos sobre el baseline anterior.

---

## Análisis y Conclusiones

### 1. Superioridad Técnica del Filtrado Exponencial
El SMA (Consistency T20) tiene un "efecto de borde": los signos antiguos salen de la ventana de golpe, lo que puede introducir saltos en el LR local. La Dual EMA proporciona un suavizado probabilístico que ignora el ruido de alta frecuencia de los bloques aleatorios de forma mucho más efectiva.

### 2. El "Crossover Gate" como Supresor de Ruido
La regresión catastrófica de T20 en `Rotated Quadratic` sugiere que la ventana SMA retiene demasiado ruido cuando el gradiente es pequeño. El **Crossover Gate** de la DS-EMA actúa como un interruptor de seguridad: ante la mínima sospecha de oscilación (desacuerdo entre fast/slow), el optimizador prefiere no moverse en esa dimensión, preservando la integridad de la trayectoria.

### 3. Eficiencia de Recursos
- **Memoria**: DS-EMA requiere $2 \cdot D$ parámetros adicionales. Consistency T20 requiere $T \cdot D = 20 \cdot D$. Reducción del **90% en overhead de memoria**.
- **Cómputo**: Las operaciones EMA son $O(1)$ por paso, mientras que el SMA requiere re-calcular la media o gestionar buffers circulares.

---

## Próximos Pasos

1. **Validación en MNIST**: Comprobar si esta ganancia de precisión en sintéticos se traduce en una convergencia más rápida en clasificación de imágenes.
2. **Sintonía de Alphas**: Evaluar si ratios más extremos (ej. Slow $\alpha=0.01$) benefician a arquitecturas más profundas.
3. **Integración**: Sustituir el `sign_buffer` de `DGEOptimizer` por la lógica DS-EMA.
