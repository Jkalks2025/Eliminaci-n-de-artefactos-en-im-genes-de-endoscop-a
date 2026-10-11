"""Paso 1: extrae de cada ZIP de C3VDv2 los frames, la profundidad real y la máscara de reflejos.

    python codigo/preparar_datos.py                 # todos los ZIPs de datos/raw
    python codigo/preparar_datos.py --paso 5 --max-frames 300

Salida por video en datos/c3vd/<video>/ (todo a 320x256):
    rgb.npy    (N,256,320,3) uint8    frames originales
    depth.npy  (N,256,320)   float16  profundidad real en mm (0 = sin dato)
    mask.npy   (N,256,320)   bool     reflejos especulares
y datos/c3vd/division.json con la división entrenamiento / validación / prueba.
"""
import argparse
import json
import zipfile

import numpy as np

import c3vd

ap = argparse.ArgumentParser()
ap.add_argument("--paso", type=int, default=5, help="tomar 1 de cada N frames (frames consecutivos son casi iguales)")
ap.add_argument("--max-frames", type=int, default=300, help="máximo de frames por video (0 = sin límite)")
args = ap.parse_args()

zips = sorted(c3vd.DIR_RAW.glob("*.zip"))
if not zips:
    raise SystemExit(f"No hay ZIPs de C3VDv2 en {c3vd.DIR_RAW}. Descárgalos ahí y vuelve a correr.")

for k, ruta in enumerate(zips, 1):
    video = ruta.stem
    salida = c3vd.DIR_DATOS / video
    if video in c3vd.EXCLUIR:
        print(f"[{k}/{len(zips)}] {video}: excluido (ver EXCLUIR en c3vd.py)")
        continue
    if (salida / "mask.npy").exists():
        print(f"[{k}/{len(zips)}] {video}: ya preparado")
        continue
    with zipfile.ZipFile(ruta) as zf:
        try:
            pares = c3vd.listar_pares(zf)[::args.paso]
        except ValueError as e:
            print(f"[{k}/{len(zips)}] {video}: saltado ({e})")
            continue
        if args.max_frames:
            pares = pares[:args.max_frames]
        rgbs, profs, masks = [], [], []
        for _, r, d in pares:
            rgb, prof = c3vd.redimensionar(c3vd.leer_rgb(zf, r), c3vd.leer_profundidad(zf, d))
            rgbs.append(rgb)
            profs.append(prof.astype(np.float16))
            masks.append(c3vd.mascara_reflejos(rgb))
    salida.mkdir(parents=True, exist_ok=True)
    np.save(salida / "rgb.npy", np.stack(rgbs))
    np.save(salida / "depth.npy", np.stack(profs))
    np.save(salida / "mask.npy", np.stack(masks))   # se guarda al final: marca el video como completo
    print(f"[{k}/{len(zips)}] {video}: {len(pares)} frames, reflejos {100 * np.mean(masks):.1f}%")

div = c3vd.division(c3vd.videos_preparados())
json.dump(div, open(c3vd.DIR_DATOS / "division.json", "w"), indent=2)
print(f"\nDivisión -> {c3vd.DIR_DATOS / 'division.json'}")
for k, v in div.items():
    print(f"  {k}: {len(v)} videos {v}")
