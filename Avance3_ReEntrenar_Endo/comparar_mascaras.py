"""Avance 3 (re-entrenar Endo-STTN): compara métodos de máscara de reflejos contra una referencia de C3VDv2.

    python Avance3_ReEntrenar_Endo/comparar_mascaras.py
    python Avance3_ReEntrenar_Endo/comparar_mascaras.py --frames-por-video 50 --k 1.4 --y-min 0.65

Los videos registrados de C3VDv2 traen, por frame, una imagen diffuse/ renderizada desde el modelo 3D
con reflexión lambertiana: el tejido sin reflejos. Un reflejo es un píxel mucho más brillante de lo
que predice esa imagen. La referencia se construye así (todo a 288x288):

    Y   = brillo del frame real (0-1)        D = imagen difusa (0-1)
    s   = mediana local (31 px) de Y / D     (ajusta textura, color y escala entre las dos imágenes)
    reflejo de referencia  <=>  Y >= y_min  y  Y > k * s * D

Después se mide cada método (endosttn_c3vd.METODOS_MASCARA) contra esa referencia, píxel por píxel:
precisión (de lo marcado, cuánto es reflejo), sensibilidad (de los reflejos, cuánto se marcó), F1 e IoU.
Se reporta estricto y con tolerancia de 2 px en los bordes (las máscaras difieren sobre todo ahí).

La referencia no es perfecta: depende del registro 3D y de los parámetros k e y_min. Hay que
revisarla visualmente (el notebook de comparación lo hace) antes de confiar en los números. En v3
(con residuos) el render no tiene residuos, así que un residuo claro también puede caer en la referencia;
por eso los resultados se separan entre videos limpios (v1, v2) y con residuos (v3).

Salida en resultados/mascaras/: comparacion_por_frame.csv, comparacion_resumen.csv y .md.
"""
import argparse
import re
import zipfile

import cv2
import numpy as np
import pandas as pd

import endosttn_c3vd as ec
from endosttn_c3vd import c3vd

DIR_SALIDA = ec.DIR_RESULTADOS / "mascaras"
K, Y_MIN, MEDIANA, TOLERANCIA = 1.3, 0.6, 31, 2


def pares_difusa(zf):
    """(índice, ruta rgb, ruta diffuse) de los frames que tienen las dos imágenes."""
    nombres = zf.namelist()
    rgb = {c3vd._indice(n): n for n in nombres if n.startswith("rgb/") and n.lower().endswith((".png", ".jpg"))}
    dif = {c3vd._indice(n): n for n in nombres if n.startswith("diffuse/") and n.lower().endswith((".png", ".tif", ".tiff"))}
    return [(i, rgb[i], dif[i]) for i in sorted(set(rgb) & set(dif) - {None})]


def leer_difusa(zf, ruta):
    """Imagen difusa en gris, 0-1, a 288x288 (acepta 8 o 16 bits, 1 o 3 canales)."""
    d = cv2.imdecode(np.frombuffer(zf.read(ruta), np.uint8), cv2.IMREAD_UNCHANGED)
    maximo = float(np.iinfo(d.dtype).max) if np.issubdtype(d.dtype, np.integer) else max(float(d.max()), 1.0)
    d = d.astype(np.float32) / maximo
    if d.ndim == 3:
        d = d[..., :3].mean(-1)
    d = cv2.resize(d, (c3vd.ANCHO, c3vd.ALTO), interpolation=cv2.INTER_AREA)
    return cv2.resize(d, (ec.LADO, ec.LADO), interpolation=cv2.INTER_AREA)


def referencia(rgb, difusa, k=K, y_min=Y_MIN, mediana=MEDIANA):
    """Máscara de referencia y mapas intermedios (para revisarla). rgb (288,288,3) uint8, difusa 0-1."""
    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    valido = difusa > 0.02
    cociente = np.where(valido, y / np.maximum(difusa, 0.02), 0)
    escala = cv2.medianBlur(np.clip(cociente * 50, 0, 255).astype(np.uint8), mediana).astype(np.float32) / 50
    prediccion = escala * difusa
    ref = valido & (y >= y_min) & (y > k * prediccion)
    return ref, {"brillo": y, "prediccion": prediccion, "exceso": y / np.maximum(prediccion, 1e-3)}


def comparar(pred, ref, tol=TOLERANCIA):
    """Conteos estrictos y con tolerancia de tol px (para sumarlos entre frames)."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    dil = lambda m: cv2.dilate(m.astype(np.uint8), ker).astype(bool)
    return {"tp": int((pred & ref).sum()), "fp": int((pred & ~ref).sum()), "fn": int((~pred & ref).sum()),
            "pred_ok_tol": int((pred & dil(ref)).sum()), "ref_ok_tol": int((ref & dil(pred)).sum()),
            "pred": int(pred.sum()), "ref": int(ref.sum()), "pixeles": int(pred.size)}


def resumir(df):
    """Precisión, sensibilidad, F1 e IoU sumando los píxeles de todos los frames del grupo."""
    s = df.sum(numeric_only=True)
    div = lambda a, b: a / b if b else float("nan")
    p, r = div(s.tp, s.tp + s.fp), div(s.tp, s.tp + s.fn)
    pt, rt = div(s.pred_ok_tol, s.pred), div(s.ref_ok_tol, s.ref)
    f1 = lambda a, b: div(2 * a * b, a + b)
    return pd.Series({"precision": p, "sensibilidad": r, "f1": f1(p, r), "iou": div(s.tp, s.tp + s.fp + s.fn),
                      "precision_tol": pt, "sensibilidad_tol": rt, "f1_tol": f1(pt, rt),
                      "marcado_pct": 100 * div(s.pred, s.pixeles), "referencia_pct": 100 * div(s.ref, s.pixeles),
                      "frames": len(df) // max(df.metodo.nunique(), 1)})


def videos_con_difusa():
    """{video: ruta del ZIP} de los ZIPs de datos/raw/ que traen diffuse/."""
    out = {}
    for z in sorted(c3vd.DIR_RAW.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            if any(n.startswith("diffuse/") for n in zf.namelist()):
                out[z.stem] = z
    return out


def muestra(ruta, n):
    """n frames equiespaciados de un video: lista de (índice, rgb 288, difusa 288)."""
    with zipfile.ZipFile(ruta) as zf:
        pares = pares_difusa(zf)
        elegidos = [pares[i] for i in np.linspace(0, len(pares) - 1, min(n, len(pares))).astype(int)] if pares else []
        return [(i, ec.frame_288(c3vd.leer_rgb(zf, r)), leer_difusa(zf, d)) for i, r, d in elegidos]


def mascaras(rgb_288):
    """Máscara de cada método sobre un frame ya a 288x288 (umbral: brillo >= UMBRAL y dilatación de 3 px)."""
    out = {m: getattr(ec, f"mascara_{m}")(rgb_288) for m in ec.METODOS_MASCARA if m != "umbral"}
    gris = cv2.cvtColor(rgb_288, cv2.COLOR_RGB2GRAY)
    out["umbral"] = cv2.dilate((gris >= c3vd.UMBRAL).astype(np.uint8),
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)
    return out


def evaluar(frames_por_video=30, k=K, y_min=Y_MIN, videos=None):
    """DataFrame con los conteos por video, frame y método."""
    disponibles = videos_con_difusa()
    filas = []
    for v, ruta in disponibles.items():
        if videos and v not in videos:
            continue
        version = (re.search(r"_v(\d+)$", v) or [None, "?"])[1]
        for i, rgb, dif in muestra(ruta, frames_por_video):
            ref, _ = referencia(rgb, dif, k, y_min)
            for metodo, pred in mascaras(rgb).items():
                filas.append({"video": v, "version": f"v{version}", "frame": i, "metodo": metodo, **comparar(pred, ref)})
        print(f"{v}: {len({f['frame'] for f in filas if f['video'] == v})} frames", flush=True)
    return pd.DataFrame(filas)


def tabla_resumen(df):
    grupos = {"todos": df, "limpios (v1, v2)": df[df.version.isin(["v1", "v2"])], "con residuos (v3)": df[df.version == "v3"]}
    filas = []
    for nombre, d in grupos.items():
        for metodo, dm in d.groupby("metodo"):
            filas.append({"videos": nombre, "metodo": metodo, **resumir(dm)})
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-por-video", type=int, default=30)
    ap.add_argument("--k", type=float, default=K, help="cuántas veces más brillante que lo predicho por diffuse/")
    ap.add_argument("--y-min", type=float, default=Y_MIN, help="brillo mínimo (0-1) para ser reflejo de referencia")
    args = ap.parse_args()

    if not videos_con_difusa():
        raise SystemExit(f"Ningún ZIP de {c3vd.DIR_RAW} trae diffuse/ (solo los videos registrados la traen).")
    df = evaluar(args.frames_por_video, args.k, args.y_min)
    res = tabla_resumen(df)
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIR_SALIDA / "comparacion_por_frame.csv", index=False)
    res.to_csv(DIR_SALIDA / "comparacion_resumen.csv", index=False)
    with open(DIR_SALIDA / "comparacion_resumen.md", "w") as f:
        f.write("# Comparación de máscaras de reflejos contra la referencia de diffuse/\n\n")
        f.write(f"Referencia: Y >= {args.y_min} y Y > {args.k} x (mediana local de Y/D) x D. "
                f"Tolerancia en bordes: {TOLERANCIA} px. {args.frames_por_video} frames por video.\n\n")
        f.write(res.round(3).to_markdown(index=False) + "\n")
    print(res[res.videos == "todos"].round(3).to_string(index=False))
    print(f"Resultados -> {DIR_SALIDA}")


if __name__ == "__main__":
    main()
