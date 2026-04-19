# Reglas estables del Repositorio DGE Optimizer

## Modificación de código existente
**REGLA DE ORO:** Bajo NINGÚN CONCEPTO debes modificar un archivo de código (`.py`) que ya exista en scratch/ y funcione.
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
