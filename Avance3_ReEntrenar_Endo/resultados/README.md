# Resultados del Avance 3 · Re-entrenar Endo-STTN

| Prueba | Fecha | Resumen |
|---|---|---|
| *(pendiente)* | | |

`evaluar_endosttn.py` escribe aquí (en Colab, en `MyDrive/Equipo19/re_entrenar_endo/resultados/`):

- `resumen.md`: configuración usada (máscara, GPU, fecha), PSNR, SSIM y MAE en los videos de prueba (original vs reentrenado), pruebas de Wilcoxon y tabla de validación por checkpoint.
- `validacion_por_checkpoint.csv`: PSNR, SSIM y MAE en el hueco desplazado para el original y cada checkpoint.
- `prueba_por_frame.csv`: PSNR, SSIM y MAE de cada frame de prueba, por modelo.
- `revision_mascaras.png`: 50 frames de entrenamiento al azar con su máscara (la guarda el notebook).

Al terminar una corrida, crea una carpeta `prueba_N/` con esos archivos y el notebook ejecutado, y agrega una fila a la tabla.

`comparar_mascaras.py` (o el notebook de comparación de máscaras) escribe en `mascaras/`: `comparacion_resumen.md`
y `.csv` (precisión, sensibilidad, F1 e IoU por método), `comparacion_por_frame.csv`, `comparacion_metricas.png` y
`comparacion_visual.png`.
