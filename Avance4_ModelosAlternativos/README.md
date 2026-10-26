# Avance 4 · Modelos alternativos

**Equipo 19** · Continúa el [Avance 3](../Avance3_ReEntrenar_Endo/README.md). Reúne las cinco corridas de
reentrenamiento de Endo-STTN con C3VDv2 y sirve de punto de partida para probar alternativas (otras máscaras de
reflejos, otras configuraciones de entrenamiento).

> **Aclaración:** _(pendiente: texto que enviará el equipo)_

## 1. Contenido

| Carpeta | Archivo | Qué es |
|---|---|---|
| `corrida_1/` … `corrida_5/` | `notebook_ejecutado.ipynb` | Notebook de Colab de cada corrida, con sus salidas |

Son **copias** de los notebooks de la rama `claude/beautiful-faraday-drzen3`, donde están también los resúmenes
(`resumen.md`), configuraciones (`config.json`, `mejor.json`), métricas (`*.csv`) y figuras de cada corrida.

## 2. Lo que comparten las cinco corridas

- Endo-STTN reentrenado desde los pesos publicados (`gen_00009.pth`), con el entrenamiento original
  (`Avance3_ReEntrenar_Endo/entrenar_endosttn.py`).
- Máscara de reflejos de **El Meslouhi et al. (2011)**, frames de 288×288, todos los frames de cada video.
- **Mismos videos de prueba:** `c2_descending_t4_v2` y `c2_descending_t4_v3` (250 frames).
- Evaluación con el *hueco desplazado* (pseudo ground truth): la máscara de reflejos movida 50 px, sobre tejido
  cuyo color sí se conoce.

## 3. En qué se diferencian

| Corrida | Validación | lr | Lote | Iteraciones | Semilla | GPU | Mejor checkpoint | PSNR | SSIM | MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| original (`gen_00009`) | — | — | — | — | — | — | — | 29.51 | 0.826 | 6.66 |
| 1 | `c1_ascending_t3_v1` | 1e-4 | 2 | 10 000 | 2020 | T4 | iter. 1000 | 27.18 | 0.808 | 8.68 |
| 2 | `c2_sigmoid_t3_v2` | 1e-4 | 4 | 4 000 | 2020 | L4 | iter. 750 | 31.67 | 0.872 | 4.75 |
| 3 | `c1_ascending_t3_v1` | 1e-4 | 4 | 2 000 | 2020 | L4 | iter. 1250 | 26.67 | 0.802 | 8.71 |
| **4** | `c2_sigmoid_t3_v2` | **1e-5** | 4 | 2 000 | 2020 | L4 | iter. 1750 | **32.52** | **0.879** | **4.30** |
| 5 | `c2_sigmoid_t3_v2` | 1e-5 | 4 | 2 000 | 2021 | L4 | iter. 2000 | 32.43 | 0.876 | 4.37 |

Métricas en el hueco desplazado de los videos de prueba. PSNR (dB) y SSIM: mayor es mejor; MAE (0-255): menor es mejor.

## 4. Conclusiones

1. **El video de validación es lo que más pesa.** Con `c1_ascending` en validación (corridas 1 y 3) ese video queda
   fuera del entrenamiento y el reentrenado es **peor** que el original (Wilcoxon p = 1). Con `c2_sigmoid` en
   validación, `c1_ascending` entra al entrenamiento y el reentrenado es **mejor** (p < 1e-26). `c1_ascending` es
   el único video del colon c1 y aporta variedad que los demás no tienen.
2. **lr 1e-5 es mejor que 1e-4** (corrida 4 vs 2): la curva de validación es estable y el resultado en prueba,
   mejor. Con 1e-4 la validación oscila de un checkpoint a otro.
3. **El resultado es reproducible** (corrida 4 vs 5): cambiar la semilla casi no lo mueve (32.52 vs 32.43 dB).
4. **Entrenar más no ayuda** (corrida 1): con 10 000 iteraciones la validación empeora casi sin parar desde la 1000.
5. **Configuración recomendada:** la de la corrida 4 (lr 1e-5, lote 4, 2000 iteraciones, `c1_ascending` en
   entrenamiento y `c2_sigmoid` en validación).

## 5. Notas importantes

- **Corridas 2 y 4 sin `config.json`.** Sus parámetros salen del notebook y del `resumen.md`; la división de
  entrenamiento se supone igual a la de la corrida 5, pero no está guardada.
- **La validación no es comparable entre corridas** con distinto video de validación (1 y 3 contra 2, 4 y 5). La
  prueba sí lo es: los mismos dos videos en todas.
- **La prueba es pequeña:** un solo segmento (`c2_descending_t4`, 2 videos). Los valores p de Wilcoxon exageran la
  certeza porque los frames de un video están correlacionados.
- **El pseudo ground truth premia copiar reflejos:** el hueco desplazado puede caer sobre otro reflejo cercano, y
  entonces el modelo que lo reproduce obtiene mejor métrica. Revisar también las figuras, no solo los números.
- **Otra máscara cambia la prueba.** El hueco desplazado sale de la propia máscara de reflejos, así que con otra
  máscara (por ejemplo Arnold et al., 2010) los huecos de prueba son otros: el modelo original da 29.09 dB con
  Arnold y 29.51 dB con El Meslouhi. Para comparar máscaras, todos los modelos deben evaluarse con los **mismos
  huecos**.
- **Arnold marca tejido claro como reflejo,** sobre todo en `c1_ascending` (hasta 23.5 % del frame contra 2.4 % de
  El Meslouhi), justo el video que hace ganar a las corridas 2, 4 y 5.

## 6. Siguiente paso

Repetir la configuración de la corrida 4 con la máscara de Arnold
([`Prueba_Entrenamiento_con_Arnold.ipynb`](../Avance3_ReEntrenar_Endo/notebooks/Prueba_Entrenamiento_con_Arnold.ipynb)),
con dos semillas (2020 y 2021), y evaluar contra los huecos de El Meslouhi para que la comparación sea justa. En
ese notebook hay que pasar `--lr 1e-5 --semilla <N>` a `entrenar_endosttn.py` y fijar `DIVISION_MANUAL` con la
división de la corrida 5.
