# Avance 2 · DeepEstimator + Endo-STTN

Endo-STTN (inpainting de reflejos especulares) se entrena con una pérdida extra de
profundidad: el frame que reconstruye pasa por **AF-SfMLearner congelado** y su
disparidad debe coincidir con la del frame limpio. El gradiente atraviesa el modelo de
profundidad (sin cambiar sus pesos) y entrena solo a Endo-STTN.

```
frames ─► [Endo-STTN] ─► comp_img ─► [AF-SfMLearner congelado] ─► disp_pred ─┐
   └────────────────────────────────► [AF-SfMLearner, sin grad] ─► disp_gt ──┴─► L_depth
gen_loss = GAN + hole + valid + depth_weight · L_depth   →  backward  →  solo Endo-STTN
```

## Contenido

| Carpeta / archivo | Qué es | Dónde va |
|---|---|---|
| `endo_sttn/core/depth_loss.py` | Pérdida de profundidad (AF-SfMLearner, Monodepth2 o MonoIIT) | `Endo-STTN\core\` |
| `endo_sttn/core/trainer.py` | Trainer oficial + pérdida de profundidad | `Endo-STTN\core\` (reemplaza) |
| `endo_sttn/configs/EndoSTTN_depth_prueba_afsfm.json` | Entrenamiento **con** profundidad | `Endo-STTN\configs\` |
| `endo_sttn/configs/EndoSTTN_prueba_sin_depth.json` | Igual, **sin** profundidad (comparación) | `Endo-STTN\configs\` |
| `scripts/preparar_c3vd.py`, `scripts/preparar_todos_c3vd.py` | Convierten C3VD al formato de Endo-STTN | raíz de `Endo-STTN\` |
| `scripts/test_depth_loss.py`, `scripts/check_depth_weights.py` | Pruebas | raíz de `Endo-STTN\` |
| `notebooks/…_local.ipynb` | Explica y reproduce la prueba en la laptop | raíz de `Endo-STTN\` |
| `notebooks/…_colab.ipynb` | Lo mismo en Google Colab (se prepara solo) | se sube a Colab |
| `instalar.ps1` | Copia todo lo anterior a `Endo-STTN` | — |
| `resultados/` | Métricas y figuras de la evaluación | — |

## Pasos en la laptop

Todo en `D:\Git`, desde PowerShell con el `venv` activado.

**1. Clonar lo necesario** (una sola vez)
```powershell
git clone https://github.com/endomapper/Endo-STTN.git D:\Git\Endo-STTN
git clone https://github.com/ShuweiShao/AF-SfMLearner D:\Git\AF-SfMLearner
git clone https://github.com/Jkalks2025/Eliminaci-n-de-artefactos-en-im-genes-de-endoscop-a.git D:\Git\Equipo19
```
Pesos de AF-SfMLearner: README de AF-SfMLearner → **Model zoo** → fila *End-to-end* →
descomprimir en `D:\Git\pesos\Model_trained_end_to_end\` (`encoder.pth`, `depth.pth`).
Preentrenado de Endo-STTN: `D:\Git\Endo-STTN\release_model\pretrained_model\gen_00009.pth`.

**2. Instalar el código del Avance 2 en Endo-STTN** (y cada vez que hagas `git pull`)
```powershell
git -C D:\Git\Equipo19 pull
powershell -ExecutionPolicy Bypass -File D:\Git\Equipo19\Avance2_DeepEstimator_EndoSTTN\instalar.ps1
cd D:\Git\Endo-STTN
```

**3. Preparar C3VD** (ZIPs originales en `datasets\raw\`)
```powershell
python preparar_todos_c3vd.py
```
Crea `datasets\EndoSTTN_dataset\` con `train.json` (todos menos `VIDEOS_EVAL`) y `test.json` (todos).
Para la prueba rápida con un video, deja `train.json` con uno solo:
```powershell
python -c "import json; t=json.load(open('datasets/EndoSTTN_dataset/test.json')); v='c2_descending_t4_v2'; json.dump({v: t[v]}, open('datasets/EndoSTTN_dataset/train.json','w'))"
```

**4. Comprobar**
```powershell
python test_depth_loss.py                                              # lógica (sin pesos)
python check_depth_weights.py configs/EndoSTTN_depth_prueba_afsfm.json # pesos reales + GPU
python train.py --model sttn --config configs/EndoSTTN_depth_prueba_afsfm.json -e
Remove-Item -Recurse -Force release_model\sttn_EndoSTTN_depth_prueba_afsfm
```

**5. Entrenar con y sin profundidad** (≈70 min cada uno en la laptop)
```powershell
python train.py --model sttn --config configs/EndoSTTN_depth_prueba_afsfm.json -c release_model/pretrained_model/ -cn 9
python train.py --model sttn --config configs/EndoSTTN_prueba_sin_depth.json  -c release_model/pretrained_model/ -cn 9
```
Si la barra de progreso no arranca en 0, sube `"iterations"` a *iteración inicial + 2000*.

**6. Evaluar y documentar**: abre `Equipo19_EndoSTTN_Profundidad_local.ipynb` (secciones 6 y 7)
y guarda la tabla y las figuras en `resultados/`.

## En Google Colab

Sube `notebooks/Equipo19_EndoSTTN_Profundidad_colab.ipynb` a Colab, activa la GPU T4 y sigue
la sección 0 del notebook. En `MyDrive/Equipo19/` necesitas: `datasets/EndoSTTN_dataset/test.json`,
los ZIPs de los dos videos en `JPEGImages/` y `Annotations/`, y `pretrained_model/gen_00009.pth`.

## Parámetros del bloque `depth` del config

| Parámetro | Efecto |
|---|---|
| `model` | `afsfmlearner` (en uso), `monodepth2` o `monoiit` |
| `warmup_iters` | Iteraciones iniciales sin pérdida de profundidad (200) |
| `mask_weight` | Peso extra dentro de la máscara (peso total = 1 + `mask_weight`) |
| `max_frames` | Frames por paso que pasan por el modelo de profundidad; bájalo si falta memoria |

`losses.depth_weight` controla el peso de la pérdida; `0` = Endo-STTN original.

## Si algo falla

| Síntoma | Causa / solución |
|---|---|
| `train.py` termina sin hacer nada | No hay GPU visible (`torch.cuda.device_count()` = 0) |
| `No module named 'networks'` | `afsfm_path` no apunta a `D:\Git\AF-SfMLearner` |
| No encuentra `encoder.pth` | `weights_dir` no apunta a la carpeta de los pesos |
| `CUDA out of memory` | `"max_frames": 1` en el config |
| `instalar.ps1` no se ejecuta | Usa `powershell -ExecutionPolicy Bypass -File ...` |
