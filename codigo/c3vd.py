"""Lectura de C3VDv2 y división por video.

Cada ZIP de C3VDv2 trae (entre otras) las carpetas rgb/ (frames de color) y depth/
(profundidad real alineada a cada frame). Aquí se emparejan por número de frame,
se convierte la profundidad a milímetros y se calcula la máscara de reflejos.
"""
import os
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
# Carpeta de trabajo (datos y modelos). prueba_rapida.py la cambia a una carpeta temporal.
_TRABAJO = Path(os.environ.get("EQUIPO19_TRABAJO", RAIZ))
DIR_RAW = _TRABAJO / "datos" / "raw"        # ZIPs originales de C3VDv2
DIR_DATOS = _TRABAJO / "datos" / "c3vd"     # frames preparados (salida de preparar_datos.py)
DIR_MODELOS = _TRABAJO / "modelos"          # estimadores ajustados (salida de ajustar_profundidad.py)
DIR_RESULTADOS = Path(os.environ.get("EQUIPO19_RESULTADOS",
                                     RAIZ / "Avance2_DeepEstimator_EndoSTTN" / "resultados"))

# Resolución de trabajo = entrada de AF-SfMLearner (C3VD es 5:4, igual que 320x256)
ANCHO, ALTO = 320, 256

# Profundidad: en C3VD un TIFF de 16 bits va de 0 a 65535 = 0 a 100 mm.
# Si probar_formato.py muestra otra escala, cámbiala aquí.
PROF_MAX_MM = 100.0

# Máscara de reflejos: brillo >= UMBRAL, dilatada DILATACION píxeles (a 320x256)
UMBRAL = 230
DILATACION = 3

# División por video (nunca por frame). Se usan solo los videos que existan.
VIDEOS_PRUEBA = ["c1_ascending_t4_v3", "c2_descending_t4_v3", "c2_rectum_t1_v3", "c1_ascending_t2_v3"]
VIDEOS_VALIDACION = ["c1_ascending_t3_v1", "c2_descending_t3_v1"]
EXCLUIR = ["c1_full_t1_v1", "c1_full_t3_v1"]   # muy grandes (17-19 GB); se pueden incluir quitándolos de aquí


def _indice(nombre):
    """Número de frame dentro del nombre de archivo ('0042_color.png' -> 42)."""
    m = re.search(r"(\d+)", Path(nombre).stem)
    return int(m.group(1)) if m else None


def listar_pares(zf):
    """Pares (indice, ruta_rgb, ruta_depth) presentes en ambas carpetas, ordenados."""
    nombres = zf.namelist()
    rgb = {_indice(n): n for n in nombres if n.startswith("rgb/") and n.lower().endswith((".png", ".jpg"))}
    dep = {_indice(n): n for n in nombres if n.startswith("depth/") and n.lower().endswith((".tiff", ".tif", ".png"))}
    if not rgb or not dep:
        carpetas = sorted({n.split("/")[0] for n in nombres if "/" in n})
        raise ValueError(f"El ZIP no tiene rgb/ y depth/ (carpetas encontradas: {carpetas})")
    comunes = sorted(set(rgb) & set(dep) - {None})
    return [(i, rgb[i], dep[i]) for i in comunes]


def leer_rgb(zf, ruta):
    img = cv2.imdecode(np.frombuffer(zf.read(ruta), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def leer_profundidad(zf, ruta):
    """Profundidad en mm (float32). 0 = sin dato (incluye píxeles saturados)."""
    d = cv2.imdecode(np.frombuffer(zf.read(ruta), np.uint8), cv2.IMREAD_UNCHANGED)
    if d.ndim == 3:
        d = d[..., 0]
    if d.dtype == np.uint16:
        mm = d.astype(np.float32) * (PROF_MAX_MM / 65535.0)
        mm[d == 65535] = 0          # saturado: "más lejos que PROF_MAX_MM", no es una medida
        return mm
    return d.astype(np.float32)


def mascara_reflejos(rgb):
    """Máscara booleana de reflejos especulares (píxeles casi blancos, dilatados)."""
    gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = (gris >= UMBRAL).astype(np.uint8)
    if DILATACION:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATACION + 1, 2 * DILATACION + 1))
        m = cv2.dilate(m, k)
    return m.astype(bool)


def redimensionar(rgb, prof):
    """A la resolución de trabajo. La profundidad usa vecino más cercano para no mezclar píxeles sin dato."""
    rgb = cv2.resize(rgb, (ANCHO, ALTO), interpolation=cv2.INTER_AREA)
    prof = cv2.resize(prof, (ANCHO, ALTO), interpolation=cv2.INTER_NEAREST)
    return rgb, prof


def division(videos):
    """Reparte los videos disponibles en entrenamiento / validación / prueba."""
    videos = [v for v in sorted(videos) if v not in EXCLUIR]
    prueba = [v for v in VIDEOS_PRUEBA if v in videos]
    val = [v for v in VIDEOS_VALIDACION if v in videos]
    train = [v for v in videos if v not in prueba and v not in val]
    return {"entrenamiento": train, "validacion": val, "prueba": prueba}


def videos_preparados():
    return sorted(p.name for p in DIR_DATOS.iterdir() if (p / "rgb.npy").exists()) if DIR_DATOS.exists() else []


def cargar_video(video, entrada="rgb"):
    """Arrays del video preparado (memmap, no se carga todo en RAM)."""
    d = DIR_DATOS / video
    return (np.load(d / f"{entrada}.npy", mmap_mode="r"),
            np.load(d / "depth.npy", mmap_mode="r"),
            np.load(d / "mask.npy", mmap_mode="r"))
