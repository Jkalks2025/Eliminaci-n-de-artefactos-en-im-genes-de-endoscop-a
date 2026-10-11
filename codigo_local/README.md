# Código para probar localmente

Aquí se centraliza el código que se corre en la laptop. Cada subcarpeta replica la
estructura del repositorio original, para copiar sus archivos encima.

| Carpeta | Qué contiene | Se copia sobre |
|---|---|---|
| `Endoscopy_Equipo19/` | Endo-STTN entrenado con pérdida de profundidad (MonoIIT o Monodepth2 congelado) | tu clon de [Endo-STTN](https://github.com/endomapper/Endo-STTN) (`E:/Endo-STTN`) |

---

## `Endoscopy_Equipo19/` — Endo-STTN guiado por profundidad

Durante el entrenamiento, el frame reconstruido por Endo-STTN pasa por un modelo de
profundidad **congelado**. La pérdida obliga a que su disparidad coincida con la del
frame limpio. Los pesos del modelo de profundidad no cambian; los gradientes solo
entrenan a Endo-STTN.

```
masked_frame ─► netG ─► comp_img ─► MonoIIT (congelado) ─► disp_pred ─┐
frames (limpios) ───────────────► MonoIIT (sin grad)    ─► disp_gt  ──┴─► L_depth
gen_loss = gan + hole + valid + depth_weight · L_depth   ──► backward ──► solo netG
```

### Archivos

| Archivo | Qué es |
|---|---|
| `core/depth_loss.py` | **Nuevo.** Carga MonoIIT o Monodepth2 y define `DepthLoss`. |
| `core/trainer.py` | `trainer.py` oficial + pérdida de profundidad (las líneas marcadas `depth`). |
| `train.py` | Sin cambios respecto al oficial (referencia). |
| `configs/EndoSTTN_depth_local.json` | Config del repo oficial ajustada a una laptop + bloque `depth`. |
| `prepare_one_video.py` | Crea un dataset de un solo video (carpeta de frames o `.mp4`) para pruebas. |
| `check_depth_weights.py` | Verifica que tus pesos reales cargan y que el gradiente pasa. |
| `Equipo19_EndoSTTN_Profundidad_local.ipynb` | Notebook que explica y reproduce la prueba en la laptop (`D:\Git\Endo-STTN`). |
| `Equipo19_EndoSTTN_Profundidad_colab.ipynb` | El mismo notebook para Google Colab: clona, descarga pesos y lee los datos de Drive. |
| `test_depth_loss.py` | Prueba sin pesos: el gradiente llega a `netG` y el modelo de profundidad no cambia. |

### Pasos (Windows)

1. **Clona Endo-STTN y respalda el trainer**
   ```powershell
   cd E:\
   git clone https://github.com/endomapper/Endo-STTN.git
   copy E:\Endo-STTN\core\trainer.py E:\Endo-STTN\core\trainer_original.py
   ```
2. **Copia los archivos de `codigo_local/Endoscopy_Equipo19/` sobre `E:\Endo-STTN\`**, respetando
   las carpetas (`core/`, `configs/`).
3. **Entorno.** Además de `requirements.txt` de Endo-STTN y PyTorch con CUDA:
   - Para MonoIIT: `pip install timm einops matplotlib` y `mmcv`/`mmseg`, que importa `MonoViT/networks/mpvit.py`.
     Usa el mismo entorno donde ya corre el Avance 7.
   - Para Monodepth2: no necesita nada extra.
4. **Dataset.** Descarga el de Endo-STTN (link en `dataset_prep/README.md` del repo oficial)
   y descomprímelo en `E:\Endo-STTN\datasets\EndoSTTN_dataset\`.
5. **Revisa las rutas** en `configs/EndoSTTN_depth_local.json` (`monovit_path`, `weights_dir`).
6. **Pruebas**
   ```powershell
   cd E:\Endo-STTN
   python test_depth_loss.py                                     # lógica, sin pesos
   python check_depth_weights.py configs/EndoSTTN_depth_local.json   # tus pesos reales
   ```
7. **Entrena** (arranca desde el modelo preentrenado `gen_00009.pth`):
   ```powershell
   python train.py --model sttn --config configs/EndoSTTN_depth_local.json -c release_model/pretrained_model/ -cn 9
   ```
   Monitorea con `tensorboard --logdir release_model` (`loss/depth_loss`).

### Prueba rápida con 1 solo video

Config: `configs/EndoSTTN_depth_prueba.json` (2000 iteraciones, checkpoint cada 500,
warm-up de 200, dataset `prueba_1video`).

1. **Dataset de un video**: genéralo con `prepare_one_video.py` (carpeta de frames o `.mp4`):
   ```powershell
   python prepare_one_video.py --input "<carpeta_de_la_secuencia>" --pattern "*color*.png"
   ```
   Crea `datasets/prueba_1video/` con este formato (máscaras de brillo por percentil 97):
   ```
   datasets/prueba_1video/
   ├── JPEGImages/video01.zip    (00000.jpg, 00001.jpg, ...)
   ├── Annotations/video01.zip   (máscaras con los mismos nombres)
   └── train.json                {"video01": <número de frames>}
   ```
2. **Humo (segundos)**: `-e` fuerza 5 iteraciones; solo comprueba que todo corre.
   ```powershell
   python train.py --model sttn --config configs/EndoSTTN_depth_prueba.json -e
   ```
   Borra después `release_model/sttn_EndoSTTN_depth_prueba/` para que el paso 3 no lo retome.
3. **Entrenamiento corto** desde el preentrenado:
   ```powershell
   python train.py --model sttn --config configs/EndoSTTN_depth_prueba.json -c release_model/pretrained_model/ -cn 9
   ```
   Si la barra de progreso no arranca en 0 (el preentrenado trae su contador de
   iteraciones en `opt_00009.pth`), el entrenamiento se detiene en cuanto supera
   `iterations`: súbelo a *iteración inicial + 2000*.
4. **Inferencia** con el checkpoint nuevo (frames y máscaras en carpetas, sin `--zip`):
   ```powershell
   python test.py --gpu 0 --overlaid --output results/prueba/ --frame <carpeta_frames_video01> --mask <carpeta_mascaras_video01> -c release_model/sttn_EndoSTTN_depth_prueba/ -cn <número en latest.ckpt>
   ```
5. **Comparación**: copia el config como `EndoSTTN_prueba_sin_depth.json` con
   `"depth_weight": 0` y repite 3–4. La carpeta de checkpoints sale del nombre del
   config, así que no se mezclan.

### Usar Monodepth2 (Endo-Depth-and-Motion) en vez de MonoIIT

Es más ligero (ResNet-18) y útil si la GPU de la laptop no alcanza. Cambia el bloque `depth`:

```json
"depth": {
    "model": "monodepth2",
    "edam_path": "E:/Endo-Depth-and-Motion",
    "weights_dir": "E:/scared_wights_complete/scared weights/monodepth2_weights/weights_19",
    "warmup_iters": 5000, "mask_weight": 4.0, "max_frames": 2
}
```

El código se toma de `Endo-Depth-and-Motion/apps/depth_estimate/`, igual que en los Avances 3 y 5.

### Usar AF-SfMLearner (pesos públicos entrenados en SCARED)

No necesita acceso al Drive del proyecto ni `mmcv`/`mmseg`. Config lista:
`configs/EndoSTTN_depth_prueba_afsfm.json`.

```powershell
git clone https://github.com/ShuweiShao/AF-SfMLearner D:\Git\AF-SfMLearner
```
Descarga los pesos *End-to-end* desde la sección **Model zoo** del README de
AF-SfMLearner y descomprímelos en `D:\Git\pesos\Model_trained_end_to_end\`
(deben quedar `encoder.pth` y `depth.pth`).

```powershell
python check_depth_weights.py configs/EndoSTTN_depth_prueba_afsfm.json
```

Su encoder no normaliza la imagen (así se entrenó); el cargador usa su propio código
para respetarlo.

### Parámetros del bloque `depth`

| Parámetro | Efecto |
|---|---|
| `model` | `monoiit` (por defecto), `monodepth2` o `afsfmlearner`. |
| `warmup_iters` | Iteraciones iniciales sin pérdida de profundidad. |
| `mask_weight` | Peso extra dentro de la máscara (peso total = 1 + `mask_weight`). |
| `max_frames` | Máximo de frames por paso que pasan por el modelo de profundidad. `0` = todos. |

`losses.depth_weight` controla el peso de la pérdida; `0` o ausente = Endo-STTN original.

### Si algo falla

| Síntoma | Causa / solución |
|---|---|
| El entrenamiento termina sin hacer nada | `torch.cuda.device_count()` es 0: falta GPU NVIDIA o PyTorch con CUDA. |
| `CUDA out of memory` | Baja `max_frames` a 1, o cambia a `monodepth2`. |
| `No module named 'mmcv'` / `'mmseg'` | Lo pide MonoViT; instálalo o usa `monodepth2`. |
| `Missing key(s) in state_dict` | `weights_dir` no apunta a la carpeta con `encoder.pth` y `depth.pth` correctos. |

### Qué modelos de profundidad sirven

Hace falta una red que vaya de una imagen a profundidad, para poder derivar respecto a la imagen.
MonoIIT, MonoViT, Monodepth2, AF-SfMLearner y EndoSfMLearner cumplen eso. EndoGaussian y NeXF
no: ajustan una escena 3D por video. Sirven para evaluar la geometría después, no como pérdida.
