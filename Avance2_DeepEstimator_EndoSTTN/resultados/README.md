# Resultados del Avance 2

`python codigo/evaluar_profundidad.py` escribe aquí:

- `evaluacion_resumen.md`: tablas de métricas por región (imagen, reflejos, anillo) y pruebas de Wilcoxon.
- `evaluacion_resumen.csv`: las mismas medias en CSV.
- `evaluacion_por_frame.csv`: métricas de cada frame de prueba, por modelo y región.

Los historiales de entrenamiento (AbsRel de validación por época) quedan en `modelos/<entrada>_s<semilla>/historial.json`.
Al terminar, copia la tabla de AbsRel a la sección 5 del [documento del avance](../README.md) y anota la semilla usada.
