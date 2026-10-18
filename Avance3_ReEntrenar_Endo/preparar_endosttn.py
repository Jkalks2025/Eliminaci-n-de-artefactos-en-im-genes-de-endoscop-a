"""Avance 3 (re-entrenar Endo-STTN), paso 1: arma con C3VDv2 un dataset en el formato que espera Endo-STTN para entrenar.

    python Avance3_ReEntrenar_Endo/preparar_endosttn.py
    python Avance3_ReEntrenar_Endo/preparar_endosttn.py --mascara umbral --paso 2 --division mi_division.json

Lee los ZIPs de datos/raw/ (solo hace falta rgb/, así que sirven también videos sin profundidad),
lleva cada frame a 288x288 como en la inferencia de Endo-STTN y calcula su máscara de reflejos:
    --mascara meslouhi  El Meslouhi et al. (2011), el método de los autores de Endo-STTN (por defecto)
    --mascara umbral    brillo >= 230, el del Avance 2 (también marca residuos muy claros)

Salida en datos/endosttn_c3vd/:
    JPEGImages/<video>__<k>.zip   frames (PNG) de los videos de entrenamiento, en trozos de --trozo frames
    Annotations/<video>__<k>.zip  máscaras de reflejos (PNG 0/255) con los mismos nombres
    train.json                    {trozo: número de frames}, como lo lee Endo-STTN/core/dataset.py
    eval/<video>/rgb.npy, mask.npy   videos de validación y prueba (288x288), para evaluar_endosttn.py
    division.json, preparacion.json

La división es por SEGMENTO del fantoma (endosttn_c3vd.division_por_segmento): todas las versiones
v1, v2, v3 de un segmento van al mismo grupo, para que el video de prueba no se parezca a los de
entrenamiento. Endo-STTN toma clips de 5 frames de un mismo "video" del dataset; los videos de
entrenamiento se parten en trozos para que cada época tenga varios lotes.
"""
import argparse
import json
import shutil
import zipfile

import cv2
import numpy as np

import endosttn_c3vd as ec
from endosttn_c3vd import c3vd


def leer_video(ruta, paso, max_frames, metodo):
    """Frames y máscaras de un ZIP de C3VDv2, a 288x288 (uint8, bool)."""
    rgbs, masks = [], []
    with zipfile.ZipFile(ruta) as zf:
        rutas = ec.frames_rgb(zf)[::paso]
        if max_frames:
            rutas = rutas[:max_frames]
        for r in rutas:
            rgb, mask = ec.frame_y_mascara(c3vd.leer_rgb(zf, r), metodo)
            rgbs.append(rgb)
            masks.append(mask)
    return np.stack(rgbs), np.stack(masks)


def escribir_trozo(nombre, rgb, mask):
    for carpeta, imgs in [("JPEGImages", [x[..., ::-1] for x in rgb]), ("Annotations", [m.astype(np.uint8) * 255 for m in mask])]:
        (ec.DIR_DATASET / carpeta).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ec.DIR_DATASET / carpeta / f"{nombre}.zip", "w") as z:
            for i, img in enumerate(imgs):
                z.writestr(f"{i:05d}.png", cv2.imencode(".png", img)[1].tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paso", type=int, default=1, help="tomar 1 de cada N frames (1 = todos)")
    ap.add_argument("--max-frames", type=int, default=0, help="máximo de frames por video (0 = sin límite)")
    ap.add_argument("--trozo", type=int, default=30, help="frames por trozo de entrenamiento")
    ap.add_argument("--mascara", choices=ec.METODOS_MASCARA, default="meslouhi", help="detección de reflejos")
    ap.add_argument("--division", help="JSON con entrenamiento/validacion/prueba (por defecto, por segmento)")
    ap.add_argument("--rehacer", action="store_true", help="borra lo preparado y vuelve a empezar")
    args = ap.parse_args()

    zips = {z.stem: z for z in sorted(c3vd.DIR_RAW.glob("*.zip"))}
    if not zips:
        raise SystemExit(f"No hay ZIPs de C3VDv2 en {c3vd.DIR_RAW}.")
    div = json.load(open(args.division)) if args.division else ec.division_por_segmento(list(zips))
    for grupo, vs in div.items():
        print(f"{grupo:14s}: {len(vs)} videos, segmentos {sorted({ec.segmento(v) for v in vs})}")
    compartidos = ec.segmentos_compartidos(div)
    if compartidos:
        print(f"AVISO: estos segmentos están en más de un grupo (la prueba se parecerá al entrenamiento): {compartidos}")
    faltan = [v for g in div.values() for v in g if v not in zips]
    if faltan:
        raise SystemExit(f"Estos videos de la división no están en {c3vd.DIR_RAW}: {faltan}")
    if not div["entrenamiento"] or not (div["validacion"] or div["prueba"]):
        raise SystemExit(f"Hacen falta videos de entrenamiento y de validación o prueba: {div}\n"
                         "Con la división por segmento se necesitan al menos 2 segmentos distintos en datos/raw/ "
                         "(mejor 3 o más); o usa --division.")

    if args.rehacer and ec.DIR_DATASET.exists():
        shutil.rmtree(ec.DIR_DATASET)
    ec.DIR_DATASET.mkdir(parents=True, exist_ok=True)
    if (ec.DIR_DATASET / "preparacion.json").exists():
        previo = json.load(open(ec.DIR_DATASET / "preparacion.json"))
        if (previo["division"], previo["paso"], previo["max_frames"], previo.get("mascara")) == \
                (div, args.paso, args.max_frames, args.mascara):
            print(f"Ya preparado en {ec.DIR_DATASET} (usa --rehacer para repetirlo)")
            return
        raise SystemExit(f"{ec.DIR_DATASET} se preparó con otra división u otros parámetros: usa --rehacer")

    trozos, resumen = {}, {}
    for grupo in ["entrenamiento", "validacion", "prueba"]:
        for v in div[grupo]:
            rgb, mask = leer_video(zips[v], args.paso, args.max_frames, args.mascara)
            resumen[v] = {"grupo": grupo, "frames": len(rgb), "reflejos_pct": round(100 * float(mask.mean()), 2)}
            if grupo == "entrenamiento":
                inicios = list(range(0, len(rgb), args.trozo))
                if len(inicios) > 1 and len(rgb) - inicios[-1] < 5:   # un clip de Endo-STTN necesita 5 frames
                    inicios.pop()
                for k, i in enumerate(inicios):
                    fin = inicios[k + 1] if k + 1 < len(inicios) else len(rgb)
                    escribir_trozo(f"{v}__{k:03d}", rgb[i:fin], mask[i:fin])
                    trozos[f"{v}__{k:03d}"] = fin - i
                resumen[v]["trozos"] = len(inicios)
            else:
                d = ec.DIR_EVAL / v
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / "rgb.npy", rgb)
                np.save(d / "mask.npy", mask)
            print(f"{grupo:14s} {v}: {len(rgb)} frames, reflejos {resumen[v]['reflejos_pct']:.1f}%", flush=True)

    json.dump(trozos, open(ec.DIR_DATASET / "train.json", "w"), indent=1)
    json.dump(div, open(ec.DIR_DATASET / "division.json", "w"), indent=2)
    json.dump({"mascara": args.mascara, "paso": args.paso, "max_frames": args.max_frames, "trozo": args.trozo,
               "division": div, "segmentos_compartidos": compartidos,
               "videos": resumen}, open(ec.DIR_DATASET / "preparacion.json", "w"), indent=2)
    print(f"\n{len(trozos)} trozos de entrenamiento ({sum(trozos.values())} frames) -> {ec.DIR_DATASET}")


if __name__ == "__main__":
    main()
