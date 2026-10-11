"""Crea un dataset de Endo-STTN con UN solo video, para pruebas rápidas.

Entrada: una carpeta con los frames del video (p. ej. una secuencia de C3VDv2,
donde los frames son *_color.png) o un archivo de video (.mp4/.avi).
Salida (formato que lee core/dataset.py):

    datasets/<nombre>/JPEGImages/<video>.zip   00000.jpg, 00001.jpg, ...
    datasets/<nombre>/Annotations/<video>.zip  máscaras con los mismos nombres
    datasets/<nombre>/train.json               {"<video>": <número de frames>}

Las máscaras marcan los brillos especulares (luminancia >= percentil 97, igual que
en el Avance 7). No se dilatan ni desplazan: eso lo hace el dataloader ("Dil" y el
desplazamiento de 50 px).

Ejemplos (desde la carpeta de Endo-STTN):
    python prepare_one_video.py --input "D:/datos/C3VDv2/cecum_t1_a" --pattern "*color*.png"
    python prepare_one_video.py --input "D:/videos/colon.mp4" --max-frames 300
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np


def read_frames(src, pattern, max_frames):
    src = Path(src)
    if src.is_dir():
        files = sorted(src.glob(pattern))
        if not files:
            raise SystemExit(f"No hay imágenes '{pattern}' en {src}")
        for f in files[:max_frames or None]:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"No se pudo leer {f}")
            yield img
    else:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise SystemExit(f"No se pudo abrir el video {src}")
        n = 0
        while not max_frames or n < max_frames:
            ok, img = cap.read()
            if not ok:
                break
            n += 1
            yield img
        cap.release()


def specular_mask(img_bgr, percentile):
    lum = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    return ((lum >= np.percentile(lum, percentile)) * 255).astype(np.uint8)


def jpg_bytes(img, quality=95):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="carpeta de frames o archivo de video")
    ap.add_argument("--pattern", default="*.png", help="filtro de frames si --input es carpeta")
    ap.add_argument("--name", default="prueba_1video", help="nombre del dataset (\"name\" en el config)")
    ap.add_argument("--video", default="video01", help="id del video dentro del dataset")
    ap.add_argument("--root", default="datasets", help="\"data_root\" del config")
    ap.add_argument("--size", type=int, default=288, help="lado de los frames (w = h del config)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = todos")
    ap.add_argument("--percentile", type=float, default=97.0, help="umbral de brillo para la máscara")
    args = ap.parse_args()

    out = Path(args.root) / args.name
    (out / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (out / "Annotations").mkdir(parents=True, exist_ok=True)

    n, mask_frac = 0, []
    with zipfile.ZipFile(out / "JPEGImages" / f"{args.video}.zip", "w") as zf, \
         zipfile.ZipFile(out / "Annotations" / f"{args.video}.zip", "w") as zm:
        for img in read_frames(args.input, args.pattern, args.max_frames):
            img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
            mask = specular_mask(img, args.percentile)
            name = f"{n:05d}.jpg"          # el orden en el zip es el orden temporal
            zf.writestr(name, jpg_bytes(img))
            zm.writestr(name, jpg_bytes(mask, quality=100))
            mask_frac.append((mask > 0).mean())
            n += 1

    if n < 5:
        raise SystemExit(f"Solo {n} frames; Endo-STTN necesita al menos sample_length (5)")
    json.dump({args.video: n}, open(out / "train.json", "w"))
    print(f"OK  {n} frames -> {out}")
    print(f"    máscara media: {100 * np.mean(mask_frac):.1f}% de cada frame")
    print(f"    en el config: \"data_root\": \"{args.root}/\", \"name\": \"{args.name}\"")
    if n > 927:
        print(f"    AVISO: {n} frames > frame_limit (927); súbelo en el config si quieres usarlos todos")


if __name__ == "__main__":
    main()
