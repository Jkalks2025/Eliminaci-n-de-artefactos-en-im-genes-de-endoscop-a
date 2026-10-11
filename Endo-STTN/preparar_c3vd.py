# -*- coding: utf-8 -*-
# Convierte un zip de C3VD al formato de Endo-STTN (entrenamiento y test.py --zip):
#   - Toma solo los frames de la carpeta rgb/
#   - Los redimensiona a 288x288 (tamano de entrenamiento del modelo)
#   - Genera mascaras marcando los reflejos (pixeles muy brillantes)
#   - Crea train.json y test.json
# Uso (desde la carpeta de Endo-STTN):  python preparar_c3vd.py
# Para convertir todos los zips de datasets/raw:  python preparar_todos_c3vd.py
import zipfile
import json
import os
import numpy as np
import cv2

# ---------- Ajusta estos valores si hace falta ----------
ORIGEN = "datasets/raw/c1_ascending_t1_v2.zip"
NOMBRE = "c1_ascending_t1_v2"
RAW = "datasets/raw"                              # zips originales de C3VD
RAIZ = "datasets/C3VD"                           # carpeta del dataset convertido
TAMANO = (288, 288)                               # ancho, alto
UMBRAL = 230   # brillo minimo (0-255) para marcar un pixel como reflejo
VIDEOS_EVAL = ["c2_descending_t4_v3"]   # se excluyen de train.json (para evaluar)
# --------------------------------------------------------


def preparar(origen, nombre):
    """Convierte un zip de C3VD. Devuelve el numero de frames (0 si no tiene rgb/)."""
    os.makedirs(os.path.join(RAIZ, "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join(RAIZ, "Annotations"), exist_ok=True)
    zip_frames = os.path.join(RAIZ, "JPEGImages", nombre + ".zip")
    zip_mascaras = os.path.join(RAIZ, "Annotations", nombre + ".zip")

    with zipfile.ZipFile(origen) as z:
        nombres = sorted(n for n in z.namelist()
                         if n.startswith("rgb/") and n.lower().endswith(".png"))
        if not nombres:
            return 0

        porcentajes = []
        with zipfile.ZipFile(zip_frames, "w") as zf, zipfile.ZipFile(zip_mascaras, "w") as zm:
            for i, n in enumerate(nombres):
                datos = np.frombuffer(z.read(n), np.uint8)
                img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
                img = cv2.resize(img, TAMANO, interpolation=cv2.INTER_AREA)

                gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mascara = (gris >= UMBRAL).astype(np.uint8) * 255
                porcentajes.append(100.0 * (mascara > 0).mean())

                archivo = f"{i:05d}.png"   # mismo nombre en ambos zips
                zf.writestr(archivo, cv2.imencode(".png", img)[1].tobytes())
                zm.writestr(archivo, cv2.imencode(".png", mascara)[1].tobytes())

    print(f"  {nombre}: reflejo promedio {np.mean(porcentajes):.2f}% por frame")
    return len(nombres)


if __name__ == "__main__":
    n = preparar(ORIGEN, NOMBRE)
    print(f"Frames encontrados en rgb/: {n}")
    # train.json lo usa el entrenamiento; test.json lo usa test.py
    for archivo in ("train.json", "test.json"):
        with open(os.path.join(RAIZ, archivo), "w") as f:
            json.dump({NOMBRE: n}, f)
    print(f"Frames:   {os.path.join(RAIZ, 'JPEGImages', NOMBRE + '.zip')}")
    print(f"Mascaras: {os.path.join(RAIZ, 'Annotations', NOMBRE + '.zip')}")
    print(f"train.json y test.json creados en {RAIZ}")
