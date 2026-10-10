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

| Parámetro | Efecto |
|---|---|
| `depth_weight` | Peso de la pérdida. `0` (o ausente) = Endo-STTN original. Empieza con 0.05–0.1. |
| `warmup_iters` | Iteraciones sin pérdida de profundidad, para que el inpainting se estabilice primero. |
| `mask_weight` | Peso extra dentro de la máscara (peso total = 1 + `mask_weight`). |
| `max_frames` | Máximo de frames por batch (de `b·t`) que pasan por MonoIIT. `0` = todos. Bájalo si falta memoria GPU. |

En TensorBoard aparece `loss/depth_loss` junto a las demás pérdidas.

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
