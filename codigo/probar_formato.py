"""Muestra cómo vienen los datos de un ZIP de C3VDv2, para confirmar la lectura de la profundidad.

    python codigo/probar_formato.py                       # primer ZIP de datos/raw
    python codigo/probar_formato.py datos/raw/<video>.zip
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import cv2

import c3vd

ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else next(iter(sorted(c3vd.DIR_RAW.glob("*.zip"))), None)
if ruta is None:
    sys.exit(f"No hay ZIPs en {c3vd.DIR_RAW}")

with zipfile.ZipFile(ruta) as zf:
    nombres = zf.namelist()
    print("ZIP:", ruta.name)
    print("Carpetas:", sorted({n.split("/")[0] for n in nombres if "/" in n}))
    pares = c3vd.listar_pares(zf)
    print(f"Pares rgb/depth: {len(pares)}  (ej.: {pares[0][1]}  <->  {pares[0][2]})")

    rgb = c3vd.leer_rgb(zf, pares[0][1])
    crudo = cv2.imdecode(np.frombuffer(zf.read(pares[0][2]), np.uint8), cv2.IMREAD_UNCHANGED)
    mm = c3vd.leer_profundidad(zf, pares[0][2])
    print(f"RGB: {rgb.shape} {rgb.dtype}")
    print(f"Profundidad en el archivo: {crudo.shape} {crudo.dtype}  min={crudo.min()}  max={crudo.max()}")
    v = mm[mm > 0]
    print(f"Profundidad interpretada: {v.min():.1f} - {v.max():.1f} mm (mediana {np.median(v):.1f} mm), "
          f"{100 * (mm > 0).mean():.0f}% de píxeles con dato")
    print(f"Reflejos (umbral {c3vd.UMBRAL}): {100 * c3vd.mascara_reflejos(rgb).mean():.1f}% del frame")
    print("\nSi la profundidad interpretada no está en milímetros razonables (unos 5-100 mm en colon),"
          "\najusta PROF_MAX_MM en codigo/c3vd.py.")
