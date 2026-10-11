# Eliminación de artefactos en imágenes de endoscopía · Equipo 19

Eliminamos reflejos especulares en video endoscópico con **Endo-STTN** (inpainting temporal) y
guiamos su entrenamiento con un **estimador de profundidad congelado** (AF-SfMLearner), para que
el relleno respete la geometría 3D del tejido. Datos: **C3VDv2** (colonoscopía 3D).

```
frames ─► [Endo-STTN] ─► frame reconstruido ─► [AF-SfMLearner congelado] ─► disparidad ─┐
   └─────────────────────────────────────────► [AF-SfMLearner, sin grad] ─► disparidad ─┴─► L_depth
pérdida = GAN + hole + valid + depth_weight · L_depth  →  el gradiente solo entrena a Endo-STTN
```

Explicación completa y evaluación: [`Avance2_DeepEstimator_EndoSTTN/`](Avance2_DeepEstimator_EndoSTTN/README.md).

**Contenido:** [Marco teórico](#marco-teórico) · [Cómo correrlo](#cómo-correrlo) · [Estado](#estado) ·
[Qué hay en el repositorio](#qué-hay-en-el-repositorio) · [Créditos y licencias](#créditos-y-licencias) · [Referencias](#referencias)

## Marco teórico

### 1. Reflejos especulares en endoscopía
El endoscopio ilumina con una fuente casi coaxial a la cámara, y la mucosa es húmeda y brillante.
Por eso aparecen **reflejos especulares**: zonas saturadas, casi blancas, que cambian de posición
con cada movimiento de la cámara. Ocultan la textura del tejido y no corresponden a su superficie,
lo que afecta a la inspección visual y a los algoritmos que dependen de la apariencia: búsqueda de
correspondencias, flujo óptico, estimación de movimiento de la cámara y de profundidad [1].

### 2. Inpainting de video: STTN y Endo-STTN
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

### 3. Estimación de profundidad monocular autosupervisada
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

### 4. Una red congelada como función de pérdida
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

### 5. Hipótesis y evaluación
**Hipótesis:** añadir $`\mathcal{L}_{depth}`$ hace que Endo-STTN rellene los reflejos con contenido
geométricamente coherente (menor error de profundidad en la zona rellenada) sin perder calidad de
imagen (PSNR similar).

**Evaluación:** en un video de C3VDv2 que no se usa para entrenar, comparamos tres modelos (Endo-STTN
preentrenado, ajustado con $`\mathcal{L}_{depth}`$ y ajustado sin ella), midiendo dentro de la máscara el
PSNR contra el frame limpio y el error de profundidad invariante a escala.

**Datos:** C3VD [10] y su extensión C3VDv2 [11] son videos de colonoscopía sobre modelos anatómicos de
silicona, con profundidad de referencia obtenida por registro 2D-3D. Las máscaras de reflejos se obtienen
por umbral de brillo.

## Cómo correrlo

Requiere Python 3.8+ y, para entrenar, una GPU NVIDIA con CUDA. También se puede usar
Google Colab: [`Equipo19_EndoSTTN_Profundidad_colab.ipynb`](Avance2_DeepEstimator_EndoSTTN/notebooks/Equipo19_EndoSTTN_Profundidad_colab.ipynb).

**1. Clonar e instalar**
```bash
git clone https://github.com/Jkalks2025/Eliminaci-n-de-artefactos-en-im-genes-de-endoscop-a.git
cd Eliminaci-n-de-artefactos-en-im-genes-de-endoscop-a
python -m venv venv            # activar: venv\Scripts\activate (Windows) o source venv/bin/activate
pip install torch torchvision  # o el comando de pytorch.org para tu versión de CUDA
pip install -r requirements.txt
```

**2. Descargar lo que no va en git** (AF-SfMLearner, sus pesos y el preentrenado de Endo-STTN)
```bash
python preparar_entorno.py
```
Al final indica si falta algo y cómo conseguirlo.

**3. Verificar** (desde `Endo-STTN/`)
```bash
cd Endo-STTN
python test_depth_loss.py       # lógica de la pérdida; no necesita pesos, datos ni GPU
python check_depth_weights.py   # pesos reales: carga el modelo y comprueba que pasa el gradiente
```

**4. Datos.** Descarga los videos de C3VDv2 (ZIPs con carpeta `rgb/`) en `Endo-STTN/datasets/raw/` y conviértelos:
```bash
python preparar_todos_c3vd.py
```
Crea `datasets/C3VD/` con `train.json` (todos menos `c2_descending_t4_v3`, reservado para evaluar)
y `test.json` (todos). Para una prueba rápida con un solo video, deja en `train.json` solo
`{"c2_descending_t4_v2": <frames>}`.

**5. Entrenar** (con profundidad y, para comparar, sin ella)
```bash
python train.py --model sttn --config configs/EndoSTTN_depth_prueba_afsfm.json -e    # prueba: 5 iteraciones
python train.py --model sttn --config configs/EndoSTTN_depth_prueba_afsfm.json -c release_model/pretrained_model/ -cn 9
python train.py --model sttn --config configs/EndoSTTN_prueba_sin_depth.json  -c release_model/pretrained_model/ -cn 9
```
Borra `release_model/sttn_EndoSTTN_depth_prueba_afsfm/` después de la prueba de 5 iteraciones.

**6. Evaluar**: notebook [`Equipo19_EndoSTTN_Profundidad_local.ipynb`](Avance2_DeepEstimator_EndoSTTN/notebooks/Equipo19_EndoSTTN_Profundidad_local.ipynb).

## Estado

| | Tarea |
|---|---|
| ✅ | Pérdida de profundidad integrada en Endo-STTN (AF-SfMLearner congelado) |
| ✅ | C3VDv2 convertido al formato de Endo-STTN (19 videos) |
| ✅ | Pesos de AF-SfMLearner cargan en GPU; prueba de arranque: 5 iteraciones a ≈2.1 s/iteración |
| ⬜ | Entrenamiento corto con y sin profundidad (2000 iteraciones, 1 video) |
| ⬜ | Evaluación en un video no visto → `Avance2_DeepEstimator_EndoSTTN/resultados/` |

## Qué hay en el repositorio

```
README.md                         este archivo
requirements.txt                  dependencias de Python
preparar_entorno.py               descarga AF-SfMLearner, pesos y preentrenado
Endo-STTN/                        Endo-STTN original + cambios del Equipo 19
├── core/depth_loss.py            ★ pérdida de profundidad (nuevo)
├── core/trainer.py               ★ trainer original + pérdida de profundidad (≈20 líneas)
├── configs/EndoSTTN_depth_prueba_afsfm.json   ★ con profundidad
├── configs/EndoSTTN_prueba_sin_depth.json     ★ sin profundidad (comparación)
├── preparar_c3vd.py, preparar_todos_c3vd.py   ★ conversión de C3VD
└── test_depth_loss.py, check_depth_weights.py ★ verificación
Avance2_DeepEstimator_EndoSTTN/   método, notebooks (local y Colab) y resultados
externos/                         (no está en git) AF-SfMLearner y pesos
```
★ = archivos del Equipo 19. El resto de `Endo-STTN/` es el código original sin cambios: el
primer commit lo agrega tal cual, así que `git log -p -- Endo-STTN` muestra solo nuestras modificaciones.

## Créditos y licencias

- **Endo-STTN** — R. Daher, F. Vasconcelos, D. Stoyanov, *A Temporal Learning Approach to Inpainting
  Endoscopic Specularities and Its Effect on Image Correspondence*, Medical Image Analysis, 2023.
  [github.com/endomapper/Endo-STTN](https://github.com/endomapper/Endo-STTN) (commit `9419a6c`), licencia MIT
  (`Endo-STTN/LICENCE`). Basado en STTN (Zeng et al., ECCV 2020).
- **AF-SfMLearner** — S. Shao et al., Medical Image Analysis, 2022.
  [github.com/ShuweiShao/AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner) (commit `53d01a4`), licencia MIT;
  partes de su red de profundidad provienen de Monodepth2 (Niantic, licencia de uso no comercial).
  No se incluye en el repositorio: lo descarga `preparar_entorno.py`.
- **C3VDv2** — Colonoscopy 3D Video Dataset (Johns Hopkins University). No se incluye en el repositorio.
- Evaluación previa de estimadores de profundidad: [proyecto_integrador_52](https://github.com/jmtoral/proyecto_integrador_52) (Equipo 52).

## Referencias

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
