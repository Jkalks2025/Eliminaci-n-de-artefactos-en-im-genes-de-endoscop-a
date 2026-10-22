# Resultados del Avance 3 · Re-entrenar Endo-STTN

| Prueba | Fecha | Resumen |
|---|---|---|
| [`prueba_1_arnold/`](prueba_1_arnold/Prueba_Entrenamiento_con_Arnold_Ejecutado.ipynb) · máscara Arnold, 2000 it., A100 | 22/10/2026 | Mejor checkpoint: iteración 1000. Prueba (`c2_descending_t4`): PSNR 29.09 → 30.33 dB, SSIM 0.841 → 0.890, MAE 6.88 → 5.74. Arnold marca tejido claro como reflejo en `c1_ascending`; no comparable directo con El Meslouhi (hueco distinto). Interpretación en el notebook |

`evaluar_endosttn.py` escribe aquí (en Colab, en `MyDrive/Equipo19/re_entrenar_endo/resultados/`):

- `resumen.md`: configuración usada (máscara, GPU, fecha), PSNR, SSIM y MAE en los videos de prueba (original vs reentrenado), pruebas de Wilcoxon y tabla de validación por checkpoint.
- `validacion_por_checkpoint.csv`: PSNR, SSIM y MAE en el hueco desplazado para el original y cada checkpoint.
- `prueba_por_frame.csv`: PSNR, SSIM y MAE de cada frame de prueba, por modelo.
- `revision_mascaras.png`: 50 frames de entrenamiento al azar con su máscara (la guarda el notebook).

Al terminar una corrida, crea una carpeta `prueba_N/` con esos archivos y el notebook ejecutado, y agrega una fila a la tabla.

El notebook `Equipo19_Avance3_Evaluar_Mascara_Meslouhi_colab.ipynb` escribe en `mascara_meslouhi/`: estadísticas por frame y por video, figuras (pasos del método, revisión de 50 frames, casos extremos) y `hoja_revision_meslouhi.csv` para la revisión manual.

`comparar_mascaras.py` (o el notebook de comparación de máscaras) escribe en `mascaras/`: `comparacion_resumen.md`
y `.csv` (precisión, sensibilidad, F1 e IoU por método), `comparacion_por_frame.csv`, `comparacion_metricas.png` y
`comparacion_visual.png`.

| Corrida | Archivo | Nota |
|---|---|---|
| Comparación de máscaras, 18/10/2026, 6 videos registrados, 30 frames c/u | [`mascaras/Comparar_Mascaras_Resultado.ipynb`](mascaras/Comparar_Mascaras_Resultado.ipynb) | **Referencia no confiable todavía:** con K = 1.3 marca los bordes de los pliegues (desalineación entre el render y el frame) y omite reflejos visibles; las métricas de esta corrida no sirven para elegir método |
