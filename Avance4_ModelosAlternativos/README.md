# Avance 4 · Modelos alternativos

**Equipo 19** · Continúa el [Avance 3](../Avance3_ReEntrenar_Endo/README.md). Reúne las cinco corridas de
reentrenamiento de Endo-STTN con C3VDv2 y sirve de punto de partida para probar alternativas (otras máscaras de
reflejos, otras configuraciones de entrenamiento).

> **Aclaración:** este avance se hizo con apoyo de herramientas de IA. Su uso, la contribución de cada integrante
> y las decisiones principales están en la [declaración de uso de IA](#7-declaración-de-uso-de-ia-y-contribución-del-equipo).

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

## 7. Declaración de uso de IA y contribución del equipo

### Uso de herramientas de IA

| Herramienta utilizada | Propósito de uso | Alcance del apoyo (celdas o secciones) | Responsabilidad: qué revisó o validó el equipo |
|---|---|---|---|
| Anthropic. (2026). Claude Code [Herramienta de inteligencia artificial generativa para programación].<br>https://claude.ai/code | Organizar la carpeta, comparar las corridas a partir de sus `resumen.md` y `config.json`, y redactar este README | Secciones 1 a 6 de este README; copia de los notebooks a `corrida_1/` … `corrida_5/` | _Por completar por el equipo_ (por ejemplo: se cotejaron las métricas de la tabla de la sección 3 contra los `resumen.md` de cada corrida) |

### Contribución individual

| Integrante | Contribución al avance |
|---|---|
| Javier Emmanuel García Escobedo | _Por completar_ |
| Mario Raúl García Ramírez | Ejecutó en Colab las corridas 1 a 5 con la máscara de El Meslouhi, variando el video de validación, la tasa de aprendizaje, el lote, las iteraciones y la semilla, y subió sus resultados. |
| Juan Miguel Molina Reyes | Evaluó la máscara de El Meslouhi en C3VDv2 y ejecutó la prueba de entrenamiento con la máscara de Arnold et al. (2010) para compararla con El Meslouhi. |

### Decisiones principales

<small><b>Decisión 1:</b> Usar la máscara de El Meslouhi et al. (2011) y no la de Arnold et al. (2010) para las corridas</small>

- <small>Responsables: Juan (evaluó El Meslouhi y entrenó con Arnold), Mario (corridas con El Meslouhi), Javier (_por completar_).</small>
- <small>Justificación: es el método que usaron los autores de Endo-STTN, y en C3VDv2 marca reflejos sin invadir el tejido. En la prueba de Juan, Arnold marcó en promedio 7.6 % de la imagen contra 3.9 % de El Meslouhi, y en `c1_ascending` llegó a 23.5 % contra 2.4 %, pintando como reflejo pared clara que no lo es.</small>
- <small>Alternativa descartada y motivo: Arnold. Su modelo reentrenado sí mejoró al original (+1.24 dB de PSNR), pero aprende a borrar tejido normal, y sus huecos de prueba son otros, así que sus métricas no se pueden comparar directamente con las de El Meslouhi. Queda como siguiente paso compararlas con los mismos huecos (sección 6).</small>

<small><b>Decisión 2:</b> Dejar `c1_ascending_t3_v1` en entrenamiento y validar con `c2_sigmoid_t3_v2`</small>

- <small>Responsable: Mario, a partir de las corridas 1 a 5.</small>
- <small>Justificación: con `c1_ascending` fuera del entrenamiento (corridas 1 y 3) el modelo reentrenado fue peor que el original en prueba; con él dentro (corridas 2, 4 y 5) fue mejor, con diferencias significativas.</small>
- <small>Alternativa descartada y motivo: validar con `c1_ascending_t3_v1`, porque deja fuera el único video del colon c1 y el reentrenamiento empeora el relleno.</small>

<small><b>Decisión 3:</b> Usar una tasa de aprendizaje de 1e-5 en lugar de 1e-4</small>

- <small>Responsable: Mario, a partir de las corridas 1 a 5.</small>
- <small>Justificación: con 1e-5 la validación es estable entre checkpoints y la prueba mejora (32.52 dB contra 31.67 dB de PSNR, corrida 4 contra 2).</small>
- <small>Alternativa descartada y motivo: 1e-4, la del entrenamiento original de Endo-STTN, porque al partir de pesos ya entrenados la validación oscila y el resultado en prueba es menor.</small>

<small><b>Decisión 4:</b> Entrenar 2000 iteraciones y comprobar el resultado con una segunda semilla</small>

- <small>Responsable: Mario, a partir de las corridas 1 a 5.</small>
- <small>Justificación: el mejor checkpoint aparece antes de las 2000 iteraciones y la corrida 5 (semilla 2021) reproduce casi igual la corrida 4 (32.43 contra 32.52 dB).</small>
- <small>Alternativa descartada y motivo: entrenamientos largos (10 000 iteraciones en la corrida 1), porque la validación empeora casi sin parar desde la iteración 1000.</small>
