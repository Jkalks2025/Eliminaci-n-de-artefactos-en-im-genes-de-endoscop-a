# Código para probar localmente

Aquí se centraliza el código que se corre en la laptop. Cada subcarpeta replica la
estructura del repositorio original, para copiar sus archivos encima.

| Carpeta | Qué contiene | Se copia sobre |
|---|---|---|
| `endo_sttn/` | Endo-STTN entrenado con pérdida de profundidad (MonoIIT o Monodepth2 congelado) | tu clon de [Endo-STTN](https://github.com/endomapper/Endo-STTN) (`E:/Endo-STTN`) |

---

## `endo_sttn/` — Endo-STTN guiado por profundidad

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
| `check_depth_weights.py` | Verifica que tus pesos reales cargan y que el gradiente pasa. |
| `test_depth_loss.py` | Prueba sin pesos: el gradiente llega a `netG` y el modelo de profundidad no cambia. |

### Pasos (Windows)

1. **Clona Endo-STTN y respalda el trainer**
   ```powershell
   cd E:\
   git clone https://github.com/endomapper/Endo-STTN.git
   copy E:\Endo-STTN\core\trainer.py E:\Endo-STTN\core\trainer_original.py
   ```
2. **Copia los archivos de `codigo_local/endo_sttn/` sobre `E:\Endo-STTN\`**, respetando
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

### Parámetros del bloque `depth`

| Parámetro | Efecto |
|---|---|
| `model` | `monoiit` (por defecto) o `monodepth2`. |
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
