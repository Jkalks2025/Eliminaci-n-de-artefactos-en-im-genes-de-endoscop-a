# Resultados · Re-entrenar Endo

| Prueba | Fecha | Resumen |
|---|---|---|
| *(pendiente)* | | |

`evaluar_endosttn.py` escribe aquí (en Colab, en `MyDrive/Equipo19/re_entrenar_endo/resultados/`):

- `resumen.md`: error del relleno en el video de prueba (original vs reentrenado), prueba de Wilcoxon y tabla de validación por checkpoint.
- `validacion_por_checkpoint.csv`: MAE y PSNR en el hueco desplazado para el original y cada checkpoint.
- `prueba_por_frame.csv`: MAE y PSNR de cada frame de prueba, por modelo.

Al terminar una corrida, crea una carpeta `prueba_N/` con esos archivos y el notebook ejecutado, y agrega una fila a la tabla.
