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
