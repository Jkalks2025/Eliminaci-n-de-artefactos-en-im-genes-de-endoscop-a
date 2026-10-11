# Avance 2 · Endo-STTN guiado por un estimador de profundidad

**Equipo 19** · Eliminación de reflejos especulares en video endoscópico

> **Cómo revisar este avance en 5 minutos**
> 1. Lee el [Resumen](#resumen) y mira la [figura del método](#4-metodología).
> 2. La idea central está en [3.4 Una red congelada como función de pérdida](#34-una-red-congelada-como-función-de-pérdida).
> 3. Revisa el estado de los [experimentos](#5-experimentos-y-resultados).
> 4. El código nuevo son dos archivos: [`depth_loss.py`](../Endo-STTN/core/depth_loss.py) y ≈20 líneas en [`trainer.py`](../Endo-STTN/core/trainer.py).
> 5. Para verificar sin datos ni GPU: `cd Endo-STTN ; python test_depth_loss.py`.

**Contenido:** [Resumen](#resumen) · [1. Objetivo e hipótesis](#1-objetivo-e-hipótesis) · [2. Contexto](#2-contexto) ·
[3. Marco teórico](#3-marco-teórico) · [4. Metodología](#4-metodología) · [5. Experimentos y resultados](#5-experimentos-y-resultados) ·
[6. Discusión y limitaciones](#6-discusión-y-limitaciones) · [7. Siguientes pasos](#7-siguientes-pasos) ·
[8. Cómo reproducir](#8-cómo-reproducir) · [9. Referencias](#9-referencias)

## Resumen

Los reflejos especulares ocultan el tejido en video endoscópico. Endo-STTN [1] los elimina con
inpainting temporal, pero nada garantiza que el relleno sea **geométricamente** coherente con el
tejido. En este avance añadimos al entrenamiento de Endo-STTN una pérdida de **consistencia de
profundidad**: un estimador de profundidad monocular congelado (AF-SfMLearner [4]) compara la
disparidad del frame reconstruido con la del frame limpio, y su gradiente entrena a Endo-STTN.
Implementamos la pérdida, convertimos C3VDv2 al formato de Endo-STTN y comprobamos que el flujo
completo entrena en GPU. La comparación cuantitativa con y sin la pérdida está pendiente.

## 1. Objetivo e hipótesis

**Objetivo.** Que Endo-STTN rellene los reflejos especulares con contenido que conserve la forma 3D
del tejido, usando un estimador de profundidad como supervisión adicional durante el entrenamiento.

**Hipótesis.** Añadir $`\mathcal{L}_{depth}`$ reduce el error de profundidad en la zona rellenada sin
empeorar la calidad de imagen (PSNR similar al de Endo-STTN sin la pérdida).

## 2. Contexto

En el proyecto, Endo-STTN se usa como paso previo a la estimación de profundidad y la reconstrucción 3D.
El Equipo 52 [9] mostró que el preprocesamiento de la imagen cambia la profundidad estimada en SCARED [5].
Aquí invertimos la relación: en lugar de solo medir cómo afecta el inpainting a la profundidad, usamos la
profundidad para **entrenar** el inpainting.

## 3. Marco teórico

### 3.1 Reflejos especulares en endoscopía
El endoscopio ilumina con una fuente casi coaxial a la cámara, y la mucosa es húmeda y brillante.
Por eso aparecen **reflejos especulares**: zonas saturadas, casi blancas, que cambian de posición
con cada movimiento de la cámara. Ocultan la textura del tejido y no corresponden a su superficie,
lo que afecta a la inspección visual y a los algoritmos que dependen de la apariencia: búsqueda de
correspondencias, flujo óptico, estimación de movimiento de la cámara y de profundidad [1].

### 3.2 Inpainting de video: STTN y Endo-STTN
El *inpainting* rellena una región marcada por una máscara con contenido plausible. En video, la
información que falta en un frame suele verse en otros frames, porque el reflejo se mueve y el tejido no.
**STTN** [2] lo aprovecha con un transformer espacio-temporal: divide los frames en parches de varias
escalas y usa atención para buscar, en frames vecinos y lejanos, los parches que mejor completan la máscara.
Se entrena con pérdidas L1 dentro (*hole*) y fuera (*valid*) de la máscara, más una pérdida adversarial
con un discriminador espacio-temporal.

**Endo-STTN** [1] adapta STTN a endoscopía. El problema es que no existe un ground truth de "cómo se
vería el tejido sin el reflejo". Su solución es una **pseudo ground truth**: la máscara de reflejos detectada
se desplaza a una zona sana del mismo frame, de modo que el modelo aprende a rellenar tejido cuyo
aspecto real sí se conoce. Al usar el modelo, se aplica sobre la máscara original para borrar los reflejos.

### 3.3 Estimación de profundidad monocular autosupervisada
Una red de profundidad monocular predice, a partir de una sola imagen, un mapa de **disparidad**
(inversa de la profundidad). En endoscopía casi no hay profundidad real para entrenar, así que se usa
**aprendizaje autosupervisado** [3]: una red de profundidad y una de movimiento de cámara se entrenan
juntas para reconstruir un frame a partir del siguiente (síntesis de vista). El error fotométrico de esa
reconstrucción es la señal de entrenamiento. La profundidad resultante tiene **escala ambigua**: se
conoce la forma relativa, no los milímetros.

La síntesis de vista supone que un punto se ve igual desde dos posiciones, algo que la iluminación
del endoscopio rompe. **AF-SfMLearner** [4] añade un módulo de *appearance flow* que modela esos
cambios de brillo entre frames y mejora la profundidad en el dataset SCARED [5]. Es el estimador que
usamos; el Equipo 52 comparó varios de estos modelos en SCARED [9].

### 3.4 Una red congelada como función de pérdida
Una red preentrenada y congelada puede usarse como **pérdida**: se compara su respuesta ante la imagen
generada con su respuesta ante la imagen de referencia. Es la idea de las *perceptual losses* [6], que
usan las características de una red de clasificación para medir la similitud visual. Aquí la red es un
estimador de profundidad, así que la pérdida mide la **consistencia geométrica**.

Sea $G$ Endo-STTN con pesos $\theta$, $D$ el estimador congelado, $I$ el frame limpio y $M$ la máscara.
El frame reconstruido es $\hat I = I (1-M) + M G_\theta(I (1-M))$ y

```math
\mathcal{L}_{depth} = \frac{\sum_p w_p  \big| \mathcal{N}(D(\hat I))_p - \mathcal{N}(D(I))_p\big|}{\sum_p w_p},
\qquad
\mathcal{N}(d) = \frac{d - \mathrm{med}(d)}{\mathrm{mean} |d - \mathrm{med}(d)|},
\qquad w = 1 + 4M
```

$\mathcal{N}$ normaliza cada mapa por su mediana y su desviación absoluta, para que la comparación no
dependa de la escala ambigua (como en las pérdidas invariantes a escala [7, 8]). El peso $w$ da 5 veces más
importancia a la zona rellenada. La pérdida total del generador es
$`\mathcal{L}_G = \mathcal{L}_{GAN} + \mathcal{L}_{hole} + \mathcal{L}_{valid} + \lambda \mathcal{L}_{depth}`$, con $\lambda = 0.1$.

Los pesos de $D$ no se actualizan, pero el gradiente **sí pasa a través de** $D$ por la regla de la cadena:

```math
\frac{\partial \mathcal{L}_{depth}}{\partial \theta}
= \frac{\partial \mathcal{L}_{depth}}{\partial D(\hat I)}\cdot
  \frac{\partial D(\hat I)}{\partial \hat I}\cdot
  \frac{\partial \hat I}{\partial G_\theta}\cdot
  \frac{\partial G_\theta}{\partial \theta},
\qquad \frac{\partial \hat I}{\partial G_\theta} = M
```

El término $\partial D/\partial \hat I$ lo aporta el estimador: le indica a Endo-STTN en qué dirección
cambiar cada píxel rellenado para que la geometría estimada se parezca a la del tejido real. El factor $M$
hace que solo se corrija la zona inventada.


## 4. Metodología

![Flujo de una iteración de entrenamiento](figuras/diagrama_metodo.png)

*Flechas continuas: datos. Flechas discontinuas: gradiente. Solo Endo-STTN (verde) actualiza sus pesos.*

| Paso | Qué pasa | Código |
|---|---|---|
| 1 | El loader tapa una zona **sana** del frame (máscara de reflejos desplazada 50 px, sin los reflejos reales) | `Endo-STTN/core/dataset.py` (original) |
| 2 | Endo-STTN rellena la zona → `comp_img` | `Endo-STTN/core/trainer.py` |
| 3 | AF-SfMLearner estima la disparidad de `comp_img` (con gradiente) y del frame limpio (sin gradiente) | `Endo-STTN/core/depth_loss.py` |
| 4 | $`\mathcal{L}_{depth}`$: L1 invariante a escala, con 5× peso dentro de la máscara | `scale_invariant_l1` |
| 5 | `gen_loss += depth_weight · L_depth`, después de `warmup_iters` | `trainer.py` |
| 6 | `backward()` + `optimG.step()`: solo cambian los pesos de Endo-STTN | `trainer.py` |

**Datos.** C3VDv2 [11]: 19 videos convertidos a 288×288 con máscaras de reflejos por umbral de brillo
(`Endo-STTN/preparar_todos_c3vd.py`). Entrenamiento: `c2_descending_t4_v2`. Evaluación: `c2_descending_t4_v3`
(no se usa para entrenar).

**Configuración** (`Endo-STTN/configs/EndoSTTN_depth_prueba_afsfm.json`):

| Parámetro | Valor | Efecto |
|---|---|---|
| Punto de partida | `gen_00009.pth` (Endo-STTN preentrenado) | Ajuste fino, no desde cero |
| Iteraciones | 2000 | Prueba corta |
| `depth_weight` (λ) | 0.1 (0 en `EndoSTTN_prueba_sin_depth.json`) | Peso de $`\mathcal{L}_{depth}`$ |
| `warmup_iters` | 200 | Iteraciones iniciales sin $`\mathcal{L}_{depth}`$ |
| `mask_weight` | 4.0 | Peso extra dentro de la máscara |
| `max_frames` | 2 | Frames por paso que pasan por el estimador (memoria) |
| Estimador | AF-SfMLearner, pesos *End-to-end* (SCARED), entrada 256×320 | Congelado |

**Evaluación.** Tres modelos sobre el video de evaluación: Endo-STTN preentrenado, ajustado con
$`\mathcal{L}_{depth}`$ y ajustado sin ella. Dentro de la máscara se mide el **PSNR** contra el frame limpio
y el **error de profundidad** invariante a escala. Notebook: [`notebooks/Equipo19_EndoSTTN_Profundidad_local.ipynb`](notebooks/Equipo19_EndoSTTN_Profundidad_local.ipynb), sección 7.

## 5. Experimentos y resultados

| | Experimento | Resultado |
|---|---|---|
| ✅ | Prueba de la pérdida sin pesos (`test_depth_loss.py`) | El gradiente llega a Endo-STTN; el estimador no cambia |
| ✅ | Pesos reales de AF-SfMLearner en GPU (`check_depth_weights.py`) | Carga a 256×320; usa 0.32 GB de GPU |
| ✅ | Prueba de arranque (5 iteraciones, GPU de laptop) | ≈2.1 s/iteración; `L_depth` = 0 durante el warm-up (esperado) |
| ⬜ | Ajuste fino **con** profundidad (2000 iteraciones) | *pendiente* |
| ⬜ | Ajuste fino **sin** profundidad (2000 iteraciones) | *pendiente* |
| ⬜ | Evaluación en el video no visto | *pendiente* → [`resultados/`](resultados/README.md) |

**Tabla de evaluación** *(se completa con la sección 7 del notebook)*

| Modelo | PSNR en máscara (dB) ↑ | Error de profundidad en máscara ↓ |
|---|---|---|
| Endo-STTN preentrenado | — | — |
| Ajustado sin $`\mathcal{L}_{depth}`$ | — | — |
| Ajustado con $`\mathcal{L}_{depth}`$ | — | — |

## 6. Discusión y limitaciones

- **Prueba acotada:** un solo video de entrenamiento y 2000 iteraciones. Sirve para validar el método,
  no como resultado final.
- **Dominio del estimador:** AF-SfMLearner se entrenó en SCARED (tejido porcino ex vivo) y se aplica a C3VDv2
  (modelos de silicona). Sus errores se convierten en la señal que recibe Endo-STTN.
- **Objetivo indirecto:** la profundidad de referencia es la que el estimador predice en el frame limpio,
  no la profundidad real del dataset.
- **Máscaras:** se obtienen por umbral de brillo; pueden incluir zonas claras que no son reflejos.

## 7. Siguientes pasos

1. Completar los entrenamientos y la tabla de evaluación.
2. Entrenar con varios videos de C3VDv2 y evaluar en otros no vistos.
3. Usar la profundidad real de C3VDv2 como objetivo en lugar de la del estimador.
4. Probar otros estimadores (MonoIIT, mejor en el estudio del Equipo 52 [9]).

## 8. Cómo reproducir

Instalación y pasos: [README principal](../README.md#cómo-correrlo). En Google Colab:
[`notebooks/Equipo19_EndoSTTN_Profundidad_colab.ipynb`](notebooks/Equipo19_EndoSTTN_Profundidad_colab.ipynb).

| Carpeta | Contenido |
|---|---|
| [`notebooks/`](notebooks/) | Explicación paso a paso y evaluación (local y Colab) |
| [`figuras/`](figuras/) | Diagrama del método |
| [`resultados/`](resultados/README.md) | Tabla de métricas y figuras de la evaluación |

## 9. Referencias

1. R. Daher, F. Vasconcelos, D. Stoyanov. *A Temporal Learning Approach to Inpainting Endoscopic Specularities and Its Effect on Image Correspondence.* Medical Image Analysis, 2023. doi:10.1016/j.media.2023.102994
2. Y. Zeng, J. Fu, H. Chao. *Learning Joint Spatial-Temporal Transformations for Video Inpainting.* ECCV, 2020.
3. C. Godard, O. Mac Aodha, M. Firman, G. Brostow. *Digging Into Self-Supervised Monocular Depth Estimation* (Monodepth2). ICCV, 2019.
4. S. Shao et al. *Self-Supervised Monocular Depth and Ego-Motion Estimation in Endoscopy: Appearance Flow to the Rescue.* Medical Image Analysis, 2022.
5. M. Allan et al. *Stereo Correspondence and Reconstruction of Endoscopic Data Challenge* (SCARED). arXiv:2101.01133, 2021.
6. J. Johnson, A. Alahi, L. Fei-Fei. *Perceptual Losses for Real-Time Style Transfer and Super-Resolution.* ECCV, 2016.
7. D. Eigen, C. Puhrsch, R. Fergus. *Depth Map Prediction from a Single Image using a Multi-Scale Deep Network.* NeurIPS, 2014.
8. R. Ranftl et al. *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer* (MiDaS). IEEE TPAMI, 2022.
9. Equipo 52, *La luz que opaca*: evaluación de estimadores de profundidad en SCARED. [github.com/jmtoral/proyecto_integrador_52](https://github.com/jmtoral/proyecto_integrador_52)
10. T. L. Bobrow et al. *Colonoscopy 3D Video Dataset with Paired Depth from 2D-3D Registration* (C3VD). Medical Image Analysis, 2023.
11. Golhar et al. *C3VDv2*, 2025.
