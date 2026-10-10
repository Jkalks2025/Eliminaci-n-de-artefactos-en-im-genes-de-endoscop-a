# Endo-STTN guiado por profundidad (MonoIIT)

Entrena Endo-STTN con una pérdida extra: el frame reconstruido debe producir, en
MonoIIT, la misma disparidad que el frame limpio original. MonoIIT está **congelado**
(no se entrena), pero sus gradientes llegan al generador de Endo-STTN.

```
masked_frame ─► netG ─► comp_img ─► MonoIIT (congelado) ─► disp_pred ─┐
frames (limpios) ───────────────► MonoIIT (sin grad)    ─► disp_gt  ──┴─► L_depth
gen_loss = gan + hole + valid + depth_weight · L_depth   ──► backward ──► solo netG
```

## Archivos

| Archivo | Qué es |
|---|---|
| `core/depth_loss.py` | **Nuevo.** Carga MonoIIT (igual que el Avance 7) y define `DepthLoss`. |
| `core/trainer.py` | `trainer.py` de Endo-STTN con la pérdida de profundidad añadida. |
| `train.py` | Sin cambios (copia de referencia). |
| `test_depth_loss.py` | Prueba que el gradiente llega a `netG` y que MonoIIT no cambia. |

## Cómo usarlo

1. Copia `core/depth_loss.py` y `core/trainer.py` a la carpeta `core/` de tu Endo-STTN
   (reemplaza `trainer.py`).
2. Añade al `.json` de configuración:

```json
"losses": {
    "...": "lo que ya tienes (GAN_LOSS, hole_weight, valid_weight, adversarial_weight)",
    "depth_weight": 0.1
},
"depth": {
    "monovit_path": "/content/drive/MyDrive/proyecto_integrador/MonoViT",
    "weights_dir": "/content/drive/MyDrive/proyecto_integrador/scared weights/monoIIT_weights/trained-winner-weights",
    "warmup_iters": 5000,
    "mask_weight": 4.0,
    "max_frames": 4
}
```

3. Entrena igual que siempre: `python train.py -c <ckpt_inicial> -cn <epoch> --config configs/<tu_config>.json`

Para usar **Endo-Depth-and-Motion** (red estilo Monodepth2) en lugar de MonoIIT:

```json
"depth": {
    "model": "monodepth2",
    "networks_root": "/content/drive/MyDrive/proyecto_integrador/Endo-Depth-and-Motion/<carpeta con networks/ y layers.py>",
    "weights_dir": "<carpeta con encoder.pth y depth.pth>",
    "num_layers": 18,
    "warmup_iters": 5000, "mask_weight": 4.0, "max_frames": 4
}
```

| Parámetro | Efecto |
|---|---|
| `model` | `monoiit` (por defecto) o `monodepth2` (Endo-Depth-and-Motion). |
| `depth_weight` | Peso de la pérdida. `0` (o ausente) = Endo-STTN original. Empieza con 0.05–0.1. |
| `warmup_iters` | Iteraciones sin pérdida de profundidad, para que el inpainting se estabilice primero. |
| `mask_weight` | Peso extra dentro de la máscara (peso total = 1 + `mask_weight`). |
| `max_frames` | Máximo de frames por batch (de `b·t`) que pasan por MonoIIT. `0` = todos. Bájalo si falta memoria GPU. |

En TensorBoard aparece `loss/depth_loss` junto a las demás pérdidas.

## ¿Qué modelos de profundidad sirven?

La pérdida necesita una red que vaya **de una imagen a profundidad** en una sola pasada,
para poder derivarla respecto a la imagen.

| Modelo | ¿Sirve como pérdida? | Por qué |
|---|---|---|
| MonoIIT | Sí (por defecto) | Red monocular feed-forward, la de menor error del estudio. |
| Endo-Depth-and-Motion | Sí (`model: monodepth2`) | Su módulo de profundidad es Monodepth2: imagen → disparidad. |
| EndoGaussian | No directamente | Optimiza una escena 3D (Gaussian Splatting) por video; no es una función imagen → profundidad. |
| NeXF / NeRF | No directamente | Igual: campo neural ajustado por escena, no un predictor reutilizable. |

EndoGaussian y NeXF sirven para **evaluar** después: reconstruyes la escena con los frames
de Endo-STTN y comparas la geometría con el ground truth.

## Detalles

- La pérdida usa `comp_img`, así que el gradiente solo actúa en la región enmascarada.
- MonoIIT recibe la imagen en [0,1] y redimensionada a su tamaño de entrenamiento
  (`height`/`width` de `encoder.pth`, por defecto 192×640), igual que `predict_depth`
  del Avance 7, pero todo en torch para no romper el grafo.
- La comparación es L1 invariante a escala (mediana + desviación media por imagen),
  porque la profundidad monocular no tiene escala absoluta.
- MonoIIT se queda siempre en `eval()` (BatchNorm fijo) y sus pesos no están en `optimG`.
  No se guarda en los checkpoints: `gen_*.pth` sigue siendo un Endo-STTN normal.

## Prueba

```bash
cd avances/endo_sttn_depth
python test_depth_loss.py   # OK  loss=...  |grad netG|=...
```
