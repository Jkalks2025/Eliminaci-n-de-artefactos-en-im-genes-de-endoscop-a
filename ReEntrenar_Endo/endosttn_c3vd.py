"""Re-entrenar Endo: rutas y utilidades compartidas para reentrenar Endo-STTN con C3VDv2.

Endo-STTN se entrena sin imágenes "limpias" de referencia: la máscara de reflejos de cada frame se
desplaza 50 px a la derecha (a 288x288) y se le quitan los píxeles que caen sobre reflejos reales.
Ese hueco cae sobre tejido sano, cuyo color sí se conoce, así que el modelo aprende a rellenarlo y
se le puede medir el error (pseudo ground truth). Así lo hace Endo-STTN/core/dataset.py al entrenar.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "codigo"))
import c3vd  # noqa: E402

ENDO_STTN = c3vd.RAIZ / "Endo-STTN"
PESOS_ORIGINALES = ENDO_STTN / "release_model" / "pretrained_model" / "gen_00009.pth"

NOMBRE = "endosttn_c3vd"                         # nombre del dataset dentro de datos/
DIR_DATASET = c3vd.DIR_DATOS.parent / NOMBRE     # datos/endosttn_c3vd/ (formato de Endo-STTN)
DIR_EVAL = DIR_DATASET / "eval"                  # videos de validación y prueba en .npy
DIR_ENTRENAMIENTO = c3vd.DIR_MODELOS / NOMBRE    # checkpoints gen_/dis_/opt_ de Endo-STTN
DIR_RESULTADOS = Path(os.environ.get("EQUIPO19_RESULTADOS_31", AQUI / "resultados"))

LADO = 288            # resolución de Endo-STTN
DESPLAZAMIENTO = 50   # px a 288x288, como Endo-STTN/core/dataset.py


def a_288(rgb_320, mask_320):
    """Frame y máscara de 320x256 (Avance 2) a 288x288, igual que limpiar_endosttn.limpiar_bloque."""
    rgb = cv2.resize(rgb_320, (LADO, LADO), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask_320.astype(np.uint8), (LADO, LADO), interpolation=cv2.INTER_NEAREST)
    return rgb, mask.astype(bool)


def hueco_pseudo(mask):
    """Máscara desplazada DESPLAZAMIENTO px a la derecha sin los reflejos reales. mask (..., H, W) bool."""
    hueco = np.zeros_like(mask)
    hueco[..., DESPLAZAMIENTO:] = mask[..., :-DESPLAZAMIENTO]
    return hueco & ~mask


def metricas_hueco(pred, real, hueco):
    """Error del relleno dentro del hueco (píxeles con color conocido). pred, real (H,W,3) uint8."""
    if hueco.sum() < 10:
        return None
    d = pred[hueco].astype(np.float64) - real[hueco].astype(np.float64)
    mse = float(np.mean(d ** 2))
    return {"mae": float(np.mean(np.abs(d))), "psnr": float(10 * np.log10(255 ** 2 / max(mse, 1e-10))),
            "pixeles": int(hueco.sum())}
