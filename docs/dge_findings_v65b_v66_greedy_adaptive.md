# DGE Findings v65b-v66: Deltas Adaptativos en Búsqueda Local (Greedy)

## Objetivo del Experimento
Tras validar en el experimento `v65` que la Búsqueda Local (Greedy) estructurada aprende de forma estable pero muy lenta (dando "pasos de hormiga" con un delta estático de `0.001` o `0.0001`), se plantearon dos líneas de ataque para acelerar la convergencia:

1. **v65b (Fuerza Bruta Manual):** Subir el `delta_add` fijo a `1e-3` y ejecutarlo puramente en CPU para minimizar el cuello de botella de transferencia PCI-e que estrangulaba a la GPU en evaluaciones secuenciales.
2. **v66 (Estrategias Evolutivas - Deltas Adaptativos):** Implementar una regla de aprendizaje biológica (inspirada en la regla del 1/5 de Rechenberg) donde *cada neurona individual* tiene su propio tamaño de paso ($\Delta$) dinámico.

### La Hipótesis Adaptativa (v66)
Si una perturbación mejora la red (éxito), significa que vamos en la dirección correcta: el optimizador recompensa a esa neurona multiplicando su $\Delta$ por `1.2` (acelerando). 
Si la perturbación falla (error), significa que hemos sobrepasado el mínimo local o estamos en un valle: el optimizador castiga a la neurona multiplicando su $\Delta$ por `0.95` (frenando y haciendo ajuste fino).

## Resultados Empíricos (2.5M Evals)

- **v65 (Greedy Base - GPU):** 64.77% en 72 minutos.
- **v65b (Greedy Base Delta=1e-3 - CPU):** 74.73% en ~12.5 minutos.
- **v66 (Greedy Adaptativo - CPU/GPU):** *(Pendiente de ejecución por el usuario)*

## Análisis de la Regresión y Rendimiento

### El Éxito de la CPU en el entorno Greedy (v65b)
El usuario ejecutó la versión `v65b` puramente en el procesador (Ryzen 7). El tiempo de ejecución se desplomó de **72 minutos a 12.5 minutos**, multiplicando la velocidad casi por 6. 
Esto confirma definitivamente nuestra sospecha sobre la ineficiencia de la GPU en algoritmos estrictamente secuenciales. Las GPUs brillan en el álgebra tensorial masiva paralela (miles de perturbaciones en un solo paso, como el DGE SOTA). En un escenario *Greedy* puro donde evaluamos micro-pasos dependientes uno detrás de otro, la latencia de lanzamiento del *kernel* de la GPU asfixia el rendimiento. La CPU, con su memoria caché ultrarrápida (L2/L3) y nula latencia de comunicación, es la plataforma óptima para este tipo de optimización secuencial de caja negra.

### El Impacto del Tamaño del Paso
Al subir el `delta` de `1e-4` a `1e-3` en el `v65b`, la precisión al final del presupuesto de 2.5M evals subió de un 64.7% a un respetable **74.73%**. Esto demuestra que la red estaba "hambrienta" de dar pasos más grandes. La curva de convergencia sigue sin aplanarse por completo, demostrando que el paisaje de pérdida topológico es sorprendentemente suave y libre de mínimos locales traicioneros.

## Siguientes Pasos
El código de `v66` (con **Deltas Adaptativos**) ya está programado. Si la intuición evolutiva es correcta, el `v66` debería ser capaz de identificar por sí solo qué neuronas necesitan pasos gigantes (deltas enormes) para romper simetrías, y qué neuronas necesitan pasos minúsculos para hacer ajuste fino, logrando la máxima velocidad de convergencia teórica sin necesidad de un optimizador con memoria (Adam).