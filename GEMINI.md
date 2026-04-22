# Reglas estables del Repositorio DGE Optimizer

## Modificación de código existente
**REGLA DE ORO (No modificar exprimentos previos):** Bajo NINGÚN CONCEPTO debes modificar un archivo de código (`.py`) que ya exista en scratch/ y funcione.
- Todas las iteraciones y mejoras algorítmicas deben realizarse creando NUEVOS archivos.
- Es preferible y totalmente aceptable duplicar código entre archivos si esto evita tocar implementaciones pasadas que ya son estables y sirven de referencia.

**REGLA DE ORO (No rendirse):** Nunca te rindas ni des por buenos los resultados de un experimento fallido culpando al algoritmo sin antes revisar exhaustivamente experimentos pasados similares y verificar si el fallo se debe a un bug de código, hiperparámetros o un error conceptual.

## Metodología de Investigación
Para cada nueva iteración o experimento algorítmico, se debe seguir este flujo de trabajo estricto:
1.  **Nueva Versión**: Crear un nuevo archivo en `scratch/` con la implementación del experimento (p.ej. `dge_prototype_vX.py`).
2.  **Experimento**: Ejecutar las pruebas y recoger métricas.
3.  **Documentación**: Crear un archivo de hallazgos en `docs/` (`dge_findings_vX.md`) detallando resultados y conclusiones.
4.  **Commit**: Realizar un commit con el código y la documentación antes de pasar a la siguiente fase.
5.  **Iteración**: Proponer y ejecutar el siguiente experimento basado en los hallazgos previos.

## Gestión de Ejecución y Cuota (Consumo de Tokens)
**REGLA DE ORO (No lanzar procesos):** Para optimizar el uso de la cuota de Antigravity y el estado de la máquina local, se deben seguir estas normas de ejecución:
- **Ejecución por defecto:** Las ejecuciones de benchmarks, entrenamientos y pruebas largas las debe realizar el **USER** directamente en su terminal. El agente no debe lanzarlas por iniciativa propia. Puede que ya haya un proceso pesado ejecutándose.
- **Ejecución bajo demanda:** Si el USER pide explícitamente al agente que lance un script, el agente **NO** debe lanzarlo en modo background si esto implica que el agente permanezca activo muestreando la salida (lo cual consume tokens rápidamente).
- **Espera de finalización:** En caso de que el agente lance un script, debe hacerlo de forma que el control no vuelva al agente hasta que el script haya terminado (o el agente debe entrar en pausa hasta recibir la señal de finalización), evitando el "sampling" continuo de logs en segundo plano.

Para ejecutar con aceleración GPU usar C:/Users/mrcm_/Local/proj/ajedrez/neural-tablebases/venv_gpu/Scripts/python.exe

## Normas de Logging y Resultados (Sistema de Métricas)

Todo experimento debe registrar obligatoriamente las siguientes métricas para asegurar la transparencia científica:

### 1. Desempeño y Coste (Efficiency)
- `final_objective`: Valor final alcanzado (Loss, Accuracy, etc.)
- `total_evaluations`: Número total de llamadas a la función f(x)
- `wall_clock_time`: Tiempo real total transcurrido
- `function_evaluation_time`: Tiempo neto gastado en forward passes (f(x))
- `internal_overhead_time`: Tiempo neto gastado por la lógica del optimizador (EMA, particionado, Adam). *Calculado como (WallClock - EvalTime).*

### 2. Estabilidad y Rigor (Robustness)
- `num_seeds`: Mínimo 5 semillas por cada configuración de hiperparámetros.
- `std_objective`: Desviación estándar entre semillas.
- `convergence_speed`: Número de evaluaciones necesarias para alcanzar el 90% del objetivo final.

### 3. Señal del Optimizador (Diagnostics)
- `snr_correlation`: Correlación de Pearson entre el gradiente ruidoso del paso actual y el acumulado (EMA).
- `gradient_sparsity`: Porcentaje de parámetros con gradiente acumulado nulo o despreciable.
- `step_efficiency`: Mejora media del objetivo por cada evaluación de función.

### 4. Entorno de Ejecución (Reproducibility)
- `commit_hash`: Hash exacto del código que generó el resultado.
- `hardware_info`: CPU, GPU (si aplica) y memoria disponible.
- `full_config`: Copia completa del JSON de hiperparámetros.

### Almacenamiento de Resultados
- Los resultados crudos se guardan en `results/raw/` como archivos `.json`.
- Los resúmenes estadísticos se guardan en `results/summary/`.
- Las gráficas comparativas se guardan en `results/figures/`.
- **REGLA DE ORO DEL LOGGING:** Ninguna afirmación de mejora es válida si no viene acompañada de un archivo JSON que demuestre que `internal_overhead_time` no anula el ahorro en `total_evaluations`.
