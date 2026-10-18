# Avance 3 · Re-entrenar Endo-STTN con C3VDv2

**Equipo 19** · Continúa el [Avance 2](../Avance2_DeepEstimator_EndoSTTN/README.md). El
[Avance 4](../Avance4_EntrenamientoConjunto/README.md) (entrenamiento conjunto) queda en pausa hasta terminar este.

> **Estado:** implementado y probado de punta a punta con datos sintéticos (preparar, entrenar, continuar un
> entrenamiento interrumpido, evaluar y el notebook). Falta correrlo con C3VDv2 en Colab con GPU.
> Sigue el [checklist del equipo](#checklist-del-equipo) acordado con el Dr. Ricardo (14/10/2026).

## 1. Idea

En el Avance 2, Endo-STTN se usa tal como se publicó (`gen_00009.pth`), entrenado con colonoscopías reales de
HyperKvasir. C3VDv2 son modelos de silicona, con otra textura, otro color y otros reflejos. Aquí Endo-STTN se
**reentrena con frames de C3VDv2** usando su mismo procedimiento de entrenamiento, para que rellene mejor los
reflejos de este dominio.

A diferencia del Avance 4 (entrenamiento conjunto), aquí Endo-STTN aprende **solo a rellenar**: no recibe ninguna señal del estimador de
profundidad. Así se puede medir por separado si un mejor relleno ayuda después a la profundidad (paso 6 del checklist).

**Hipótesis.** El Endo-STTN reentrenado rellena el tejido de C3VDv2 con menos error que el original, en videos que
no vio al entrenar.

## 2. Las máscaras

Para entrenar un modelo de inpainting normalmente se necesita la imagen "sin reflejo" como respuesta correcta, y no
existe. Endo-STTN lo resuelve con dos máscaras, y las dos se generan automáticamente:

| Máscara | Cómo se obtiene | Para qué |
|---|---|---|
| **Reflejos** | Por defecto, **El Meslouhi et al. (2011)**, el método que usaron los autores de Endo-STTN (ver abajo). Opcional: el umbral de brillo del Avance 2 (`--mascara umbral`) | Indica dónde están los reflejos reales |
| **Hueco desplazado** | La máscara de reflejos movida 50 px a la derecha (a 288×288), sin los píxeles que caen sobre otro reflejo | Cae sobre tejido sano, cuyo color sí se conoce: es la respuesta correcta (*pseudo ground truth*) |

```
frame ──► máscara de reflejos ──► desplazar 50 px ──► hueco sobre tejido sano
                                                         │
                       Endo-STTN rellena el hueco ◄──────┘   L1 contra el color real del hueco
```

Al entrenar, el propio código de Endo-STTN (`Endo-STTN/core/dataset.py`) desplaza la máscara; al evaluar lo hace
`evaluar_endosttn.py` de la misma forma. Es el método con el que los autores de Endo-STTN entrenaron y evaluaron su
modelo (Daher et al., 2023).

**Detección de reflejos de El Meslouhi (2011).** Traducción a Python de `Endo-STTN/dataset_prep/SpecularDetectionMeslouhi2011.m`
y `createSpecularMasks288.m` (`endosttn_c3vd.mascara_meslouhi`), calculada sobre el frame de 288×288:

1. **Realce:** cada canal se multiplica por $`s = \min(R,G,B)/\max(R,G,B)`$. Los colores saturados (el rojo del
   tejido, el amarillo de los residuos) se oscurecen y los píxeles blancos casi no cambian.
2. **Color a CIE-XYZ** (sRGB, D65, como `rgb2xyz` de MATLAB).
3. **Reflejo** donde la luminancia es mayor o igual que la luminancia cromática: $`Y \ge Y/(X+Y+Z)`$.
4. **Dilatación** de 1 px con un rombo.

A diferencia del umbral de brillo, no marca los residuos amarillos muy claros, pero puede marcar tejido muy claro
y poco saturado. Por eso el notebook compara los dos métodos y muestra **50 máscaras al azar** para revisarlas a ojo
(`resultados/revision_mascaras.png`) antes de entrenar. El método final se decide con el Dr. Ricardo.

## 3. Configuración

Se usa el entrenamiento original (`Endo-STTN/core/trainer.py`), **sin modificarlo**. Solo cambia la configuración:

| Parámetro | Original (HyperKvasir) | Aquí (C3VDv2) |
|---|---|---|
| Datos | 2 GPUs, videos de HyperKvasir | 1 GPU (Colab), videos de entrenamiento de C3VDv2 |
| Punto de partida | STTN preentrenado | Generador `gen_00009.pth`; discriminador desde cero (el modelo publicado no lo trae) |
| Pérdidas | L1 en el hueco + L1 fuera del hueco + adversarial (hinge, 0.01) | Igual |
| Optimizador | Adam, lr 1e-4, betas (0, 0.99) | Igual |
| Clips | 5 frames a 288×288 | Igual |
| Lote | 8 | 4 (`--lote`) |
| Iteraciones | 500 000 | 10 000 por defecto (`--iteraciones`); un checkpoint cada 1 000 |
| Máscaras | El Meslouhi (2011) en MATLAB, dilatadas 1 px | El Meslouhi en Python, dilatadas 1 px (o umbral del Avance 2) |
| Dilatación extra al cargar (`Dil`) | 8 | 0: las máscaras ya vienen dilatadas |
| División | Por video | **Por segmento** (ver abajo) |

Los frames pasan por 320×256 (como en el Avance 2) y luego a 288×288, igual que en la inferencia. Solo se lee la
carpeta `rgb/` de cada ZIP, así que también sirven videos de C3VDv2 sin profundidad (deformación y cribado). Por defecto se usan **todos los frames** de cada video (`--paso 1`), no 1 de cada 5 como en el Avance 2,
porque Endo-STTN necesita más imágenes y aprovecha la continuidad entre frames. Cada video de entrenamiento se parte
en trozos de 30 frames, porque Endo-STTN toma un clip por "video" en cada paso y con solo 3 o 4 videos una época
tendría un único lote.

**División por segmento.** El nombre de cada video indica fantoma, segmento, textura y versión
(`c2_descending_t4_v3`). Todas las versiones de un mismo segmento (`c2_descending_t4`: v1, v2 y v3) van al mismo grupo,
porque recorren el mismo tejido; si una fuera a prueba y otra a entrenamiento, la prueba se parecería demasiado al
entrenamiento (el problema detectado en el Avance 2). Por defecto se aparta un segmento para prueba y otro para
validación (los de `VIDEOS_PRUEBA` y `VIDEOS_VALIDACION` de `codigo/c3vd.py` si están disponibles) y el resto va a
entrenamiento. Conviene tener al menos 3 segmentos distintos en `raw/`, idealmente de varias partes del colon.

**Elección del modelo.** Se evalúan el original y todos los checkpoints en el video de validación. El de menor MAE
en el hueco es el "reentrenado" (`gen_mejor.pth`), y se compara contra el original en los videos de prueba.

**Métricas.** Las de `quantifyResults.ipynb` de Endo-STTN, dentro del hueco desplazado: **PSNR** sobre los píxeles del
hueco (su *PSNRCrop*) y **SSIM**, el mapa SSIM de la imagen promediado en el hueco (su *SSIMCropFull*). Además, MAE.
No se usa ese notebook directamente porque sus funciones de scikit-image (`compare_ssim`, `compare_psnr`) ya no existen.

**Bitácora.** `modelos/config.json` guarda la configuración, la GPU, la fecha, la máscara y la división, y
`resultados/resumen.md` las repite al inicio.

## 4. Cómo correrlo

**En Google Colab (recomendado):** [`notebooks/Equipo19_Avance3_ReEntrenar_Endo_colab.ipynb`](notebooks/Equipo19_Avance3_ReEntrenar_Endo_colab.ipynb).
Usa `MyDrive/Equipo19/raw/` (los mismos ZIPs del Avance 2) y toma la división de `c3vd_preparado.zip`. Todo lo
que produce queda en `MyDrive/Equipo19/re_entrenar_endo/`, y si Colab se desconecta, al volver a correrlo continúa desde el
último checkpoint.

**En una computadora con GPU:**

```bash
pip install tensorboardX                                              # lo usa el entrenamiento de Endo-STTN
python Avance3_ReEntrenar_Endo/preparar_endosttn.py         # datos/endosttn_c3vd/
python Avance3_ReEntrenar_Endo/entrenar_endosttn.py         # modelos/endosttn_c3vd/gen_XXXXX.pth
python Avance3_ReEntrenar_Endo/evaluar_endosttn.py          # resultados/ y gen_mejor.pth
```

Opciones útiles: `--mascara umbral` (método del Avance 2), `--iteraciones 20000`, `--lote 2` (si falta memoria),
`--paso 2` (menos frames), `--division mi_division.json` (otra división de videos).

## 5. Experimentos

| | Experimento | Resultado |
|---|---|---|
| ✅ | Prueba de punta a punta con datos sintéticos (scripts y notebook) | Corre; reanuda tras una interrupción |
| ⬜ | Revisión de las 50 máscaras y elección del método con el Dr. Ricardo | *pendiente* |
| ⬜ | Subir a `raw/` videos de más segmentos del colon | *pendiente* |
| ⬜ | Reentrenamiento con C3VDv2 (10 000 iteraciones) y evaluación en el video de prueba | *pendiente* |
| ⬜ | Inspección visual del relleno original vs reentrenado | *pendiente* |
| ⬜ | Paso 6 del checklist: limpiar los frames con `gen_mejor.pth` y ajustar el estimador de profundidad (comparar con D) | *pendiente* |

## 6. Limitaciones

- **Pocos videos.** Con 3 o 4 videos de entrenamiento, el modelo puede aprender texturas específicas de esos
  segmentos. La validación por checkpoint ayuda a detectar el sobreajuste.
- **Máscaras por detección automática.** Ningún método es perfecto; la revisión a ojo de 50 máscaras es parte del
  proceso, no un extra.
- **El hueco desplazado no es un reflejo.** El error se mide sobre tejido sano con la forma de un reflejo; debajo de
  un reflejo real el tejido puede ser distinto (más cerca de la cámara, de frente a la luz).
- **Discriminador nuevo.** Al inicio el discriminador no sabe nada, y la pérdida adversarial puede ser inestable.
  Tiene peso 0.01, como en el original, así que domina la L1.
- **Espacio en Drive.** Cada checkpoint del generador ocupa unos 66 MB; además se conservan el último discriminador y
  optimizador (unos 350 MB).

## Archivos

| Archivo | Contenido |
|---|---|
| [`preparar_endosttn.py`](preparar_endosttn.py) | Paso 1: frames y máscaras de C3VDv2 en el formato de Endo-STTN |
| [`entrenar_endosttn.py`](entrenar_endosttn.py) | Paso 2: entrenamiento original de Endo-STTN con la configuración de C3VDv2 |
| [`evaluar_endosttn.py`](evaluar_endosttn.py) | Paso 3: elección del checkpoint y comparación original vs reentrenado |
| [`endosttn_c3vd.py`](endosttn_c3vd.py) | Máscaras (El Meslouhi y umbral), división por segmento, hueco desplazado y métricas |
| [`notebooks/Equipo19_Avance3_ReEntrenar_Endo_colab.ipynb`](notebooks/Equipo19_Avance3_ReEntrenar_Endo_colab.ipynb) | Flujo completo en Colab con figuras |
| [`resultados/`](resultados/) | Resultados de cada corrida |

## Checklist del equipo

Correspondencia con los pasos 2 a 5 del *Checklist del equipo: re-entrenar Endo-STTN* (v1.0, 14/10/2026):

| Paso | Qué pide | Dónde se hace |
|---|---|---|
| 2 · Máscaras | Método de reflejos (El Meslouhi u otro), script propio sin tocar Endo-STTN, revisar 50 máscaras | `endosttn_c3vd.mascara_meslouhi`, `preparar_endosttn.py --mascara`; celda de revisión del notebook |
| 3 · Datos | Más videos, separación por segmento, 288×288 en el formato de `dataset_prep` | `division_por_segmento`, `preparar_endosttn.py` (`JPEGImages/`, `Annotations/`, `train.json`) |
| 4 · Fine-tuning | Partir de `gen_00009.pth`, prueba corta, guardar en Drive, anotar configuración | `entrenar_endosttn.py` (`ITERACIONES` bajo para la prueba corta), `config.json` |
| 5 · Evaluar | Original vs nuevo en los mismos videos, PSNR y SSIM, figura lado a lado | `evaluar_endosttn.py`, figuras del notebook |

## Referencias

- R. Daher, F. Vasconcelos y D. Stoyanov, *A Temporal Learning Approach to Inpainting Endoscopic Specularities and
  Its Effect on Image Correspondence*, Medical Image Analysis, 2023.
- A. El Meslouhi, M. Kardouchi, H. Allali, T. Gadi y Y. A. Benkaddour, *Automatic detection and inpainting of specular
  reflections for colposcopic images*, Central European Journal of Computer Science, 2011.

---

*Este avance se elaboró con apoyo de Claude (Anthropic) para código y documentación.
Todos los integrantes revisaron y validaron el contenido, y el equipo es responsable de él.*
