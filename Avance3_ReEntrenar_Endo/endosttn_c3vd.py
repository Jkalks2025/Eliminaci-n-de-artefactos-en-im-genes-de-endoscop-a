"""Avance 3 (re-entrenar Endo-STTN): rutas y utilidades compartidas para reentrenar Endo-STTN con C3VDv2.

Endo-STTN se entrena sin imágenes "limpias" de referencia: la máscara de reflejos de cada frame se
desplaza 50 px a la derecha (a 288x288) y se le quitan los píxeles que caen sobre reflejos reales.
Ese hueco cae sobre tejido sano, cuyo color sí se conoce, así que el modelo aprende a rellenarlo y
se le puede medir el error (pseudo ground truth). Así lo hace Endo-STTN/core/dataset.py al entrenar.
"""
import os
import re
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
MIN_PIXELES = 30      # frames con un hueco más chico no se evalúan (como quantifyResults.ipynb)
METODOS_MASCARA = ["meslouhi", "hsv", "arnold", "umbral"]


# ---------------------------------------------------------------- máscaras de reflejos

def _srgb_a_xyz(img):
    """RGB en [0,1] (sRGB, D65) a CIE-XYZ, como rgb2xyz de MATLAB."""
    lin = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return lin @ m.T


def mascara_meslouhi(rgb):
    """Detección de reflejos de El Meslouhi et al. (2011), la que usaron los autores de Endo-STTN.

    Traducción de Endo-STTN/dataset_prep/SpecularDetectionMeslouhi2011.m y createSpecularMasks288.m:
    1. realce: cada canal se multiplica por min(RGB)/max(RGB) (apaga los colores saturados);
    2. se pasa a CIE-XYZ;
    3. reflejo donde la luminancia Y >= la luminancia cromática y = Y / (X + Y + Z);
    4. dilatación de 1 px con un rombo.
    rgb (H,W,3) uint8 -> (H,W) bool.
    """
    img = rgb.astype(np.float64) / 255
    mn, mx = img.min(-1), img.max(-1)
    s = np.divide(mn, mx, out=np.zeros_like(mn), where=mx > 0)
    xyz = _srgb_a_xyz(s[..., None] * img)
    suma = xyz.sum(-1)
    y = np.divide(xyz[..., 1], suma, out=np.full_like(suma, np.inf), where=suma > 0)
    m = (xyz[..., 1] >= y).astype(np.uint8)
    rombo = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    return cv2.dilate(m, rombo).astype(bool)


def _dilatar_1px(m):
    return cv2.dilate(m.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)).astype(bool)


def mascara_hsv(rgb, v_min=235, s_max=40):
    """Muy brillante Y casi sin color: V >= v_min y S <= s_max (HSV de OpenCV, 0-255), dilatada 1 px.

    Es el criterio de 'frac_especular' del Avance 1. Los residuos amarillos o marrones tienen color
    (S alta), así que no se marcan aunque sean claros.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return _dilatar_1px((hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max))


def mascara_arnold(rgb, t_abs=240, t_cand=200, t_rel=1.2, mediana=31):
    """Versión simplificada del método de Arnold et al. (2010) para endoscopía. Dos detectores:

    1. Absoluto: píxeles muy brillantes en gris o en los canales G y B, con umbrales corregidos por el
       balance de color de la imagen (cociente entre percentiles 95 de cada canal y del gris).
    2. Relativo: se estima una imagen "sin reflejos" rellenando los candidatos (umbral más bajo) y
       aplicando un filtro de mediana grande; es reflejo lo que es t_rel veces más brillante que esa
       imagen en algún canal. Detecta reflejos tenues que no llegan al umbral absoluto.
    El resultado se dilata 1 px. Los parámetros son los de uso común, no los del artículo exactamente.
    """
    img = rgb.astype(np.float32)
    gris = img @ np.array([0.2989, 0.5870, 0.1140], np.float32)
    p95 = lambda c: max(float(np.percentile(c, 95)), 1.0)
    r_g, r_b = p95(img[..., 1]) / p95(gris), p95(img[..., 2]) / p95(gris)
    sobre = lambda t: (img[..., 1] > r_g * t) | (img[..., 2] > r_b * t) | (gris > t)
    absoluto = sobre(t_abs)
    candidatos = cv2.dilate(sobre(t_cand).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    sin_reflejos = cv2.medianBlur(cv2.inpaint(rgb, candidatos, 5, cv2.INPAINT_TELEA), mediana).astype(np.float32)
    relativo = ((img / (sin_reflejos + 1)).max(-1) > t_rel) & (gris > t_cand * 0.75)
    return _dilatar_1px(absoluto | relativo)


def frame_288(rgb):
    """Frame original de C3VDv2 a 288x288, pasando por 320x256 como en el Avance 2 y en la inferencia."""
    rgb_320 = cv2.resize(rgb, (c3vd.ANCHO, c3vd.ALTO), interpolation=cv2.INTER_AREA)
    return cv2.resize(rgb_320, (LADO, LADO), interpolation=cv2.INTER_AREA)


def frame_y_mascara(rgb, metodo):
    """Frame original de C3VDv2 -> frame y máscara a 288x288 (uint8, bool).

    El frame pasa primero a 320x256 y luego a 288x288, igual que en el Avance 2 y en la inferencia.
    - meslouhi, hsv, arnold: la máscara se calcula a 288x288 (meslouhi, como hicieron los autores).
    - umbral: la del Avance 2 (brillo >= 230, dilatada 3 px a 320x256), llevada a 288x288.
    """
    rgb_320 = cv2.resize(rgb, (c3vd.ANCHO, c3vd.ALTO), interpolation=cv2.INTER_AREA)
    rgb_288 = cv2.resize(rgb_320, (LADO, LADO), interpolation=cv2.INTER_AREA)
    if metodo in ("meslouhi", "hsv", "arnold"):
        return rgb_288, {"meslouhi": mascara_meslouhi, "hsv": mascara_hsv, "arnold": mascara_arnold}[metodo](rgb_288)
    if metodo == "umbral":
        m = cv2.resize(c3vd.mascara_reflejos(rgb_320).astype(np.uint8), (LADO, LADO),
                       interpolation=cv2.INTER_NEAREST)
        return rgb_288, m.astype(bool)
    raise ValueError(f"Método de máscara desconocido: {metodo} (opciones: {METODOS_MASCARA})")


# ---------------------------------------------------------------- videos y división

def frames_rgb(zf):
    """Rutas de los frames de color de un ZIP de C3VDv2, en orden. Solo se necesita rgb/: sirven
    también los videos sin profundidad (deformación y cribado)."""
    rgb = [n for n in zf.namelist() if n.startswith("rgb/") and n.lower().endswith((".png", ".jpg"))]
    if not rgb:
        carpetas = sorted({n.split("/")[0] for n in zf.namelist() if "/" in n})
        raise ValueError(f"El ZIP no tiene rgb/ (carpetas encontradas: {carpetas})")
    return sorted(rgb, key=lambda n: (c3vd._indice(n) is None, c3vd._indice(n) or 0, n))


def segmento(video):
    """Segmento del fantoma: el nombre sin la versión ('c2_descending_t4_v3' -> 'c2_descending_t4')."""
    return re.sub(r"_v\d+$", "", video)


def division_por_segmento(videos):
    """Entrenamiento / validación / prueba por SEGMENTO: todas las versiones (v1, v2, v3...) de un
    segmento quedan en el mismo grupo, para que el video de prueba no se parezca a los de entrenamiento.

    Un segmento para prueba y uno para validación: el primero disponible de c3vd.VIDEOS_PRUEBA y de
    c3vd.VIDEOS_VALIDACION; si no hay, el último segmento en orden alfabético. El resto, a entrenamiento.
    Con menos de 3 segmentos no hay validación. Para otra división, usa --division.
    """
    videos = sorted(v for v in videos if v not in c3vd.EXCLUIR)
    libres = sorted({segmento(v) for v in videos})

    def elegir(preferidos):
        if len(libres) < 2:          # siempre queda al menos un segmento para entrenar
            return []
        s = next((segmento(v) for v in preferidos if segmento(v) in libres), libres[-1])
        libres.remove(s)
        return [s]

    prueba = elegir(c3vd.VIDEOS_PRUEBA)
    val = elegir(c3vd.VIDEOS_VALIDACION)
    grupo = lambda ss: [v for v in videos if segmento(v) in ss]
    return {"entrenamiento": grupo(libres), "validacion": grupo(val), "prueba": grupo(prueba)}


def segmentos_compartidos(div):
    """Segmentos que aparecen en más de un grupo (fuga de información entre entrenamiento y evaluación)."""
    por_grupo = {g: {segmento(v) for v in vs} for g, vs in div.items()}
    return sorted((por_grupo["entrenamiento"] & (por_grupo["validacion"] | por_grupo["prueba"]))
                  | (por_grupo["validacion"] & por_grupo["prueba"]))


# ---------------------------------------------------------------- pseudo ground truth y métricas

def hueco_pseudo(mask):
    """Máscara desplazada DESPLAZAMIENTO px a la derecha sin los reflejos reales. mask (..., H, W) bool."""
    hueco = np.zeros_like(mask)
    hueco[..., DESPLAZAMIENTO:] = mask[..., :-DESPLAZAMIENTO]
    return hueco & ~mask


def metricas_hueco(pred, real, hueco):
    """Error del relleno dentro del hueco (píxeles con color conocido). pred, real (H,W,3) uint8.

    mae y psnr: sobre los píxeles del hueco (PSNRCrop de quantifyResults.ipynb).
    ssim: mapa SSIM de la imagen completa promediado dentro del hueco (SSIMCropFull de quantifyResults).
    """
    if hueco.sum() < MIN_PIXELES:
        return None
    d = pred[hueco].astype(np.float64) - real[hueco].astype(np.float64)
    mse = float(np.mean(d ** 2))
    from skimage.metrics import structural_similarity
    _, mapa = structural_similarity(real, pred, channel_axis=2, data_range=255, full=True)
    return {"mae": float(np.mean(np.abs(d))), "psnr": float(10 * np.log10(255 ** 2 / max(mse, 1e-10))),
            "ssim": float(mapa.mean(-1)[hueco].mean()), "pixeles": int(hueco.sum())}
