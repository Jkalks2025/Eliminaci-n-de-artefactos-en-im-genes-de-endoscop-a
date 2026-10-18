"""Paso 4: compara la profundidad de los modelos contra la profundidad real de C3VDv2.

    python codigo/evaluar_profundidad.py
    python codigo/evaluar_profundidad.py --semilla 0

| Modelo | Pesos                         | Frames de prueba |
| A      | AF-SfMLearner (SCARED)        | originales       |
| B      | AF-SfMLearner (SCARED)        | limpios          |
| C      | ajustado con frames originales| originales       |
| D      | ajustado con frames limpios   | limpios          |
| E      | Endo-STTN + estimador ajustados juntos (Entrenamiento_end_to_end/ajustar_conjunto.py) | limpios por su Endo-STTN |

Mide en videos de prueba (no usados para entrenar), en tres regiones: toda la imagen, dentro de
los reflejos y un anillo de 10 px alrededor. Guarda en Avance2_DeepEstimator_EndoSTTN/resultados/:
evaluacion_por_frame.csv, evaluacion_resumen.csv y evaluacion_resumen.md.
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd
import torch

import c3vd
import modelo_profundidad as mp

RESULTADOS = c3vd.DIR_RESULTADOS
METRICAS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1"]


def regiones(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    anillo = cv2.dilate(mask.astype(np.uint8), k).astype(bool) & ~mask
    return {"imagen": None, "reflejos": mask, "anillo": anillo}


def wilcoxon(df, a, b, region="reflejos", metrica="abs_rel"):
    """Prueba pareada por frame: ¿el modelo b tiene menor error que a?"""
    try:
        from scipy.stats import wilcoxon as w
    except ImportError:
        return None
    t = df[df.region == region].pivot_table(index=["video", "frame"], columns="modelo", values=metrica).dropna()
    if a not in t or b not in t or len(t) < 10:
        return None
    r = w(t[a], t[b], alternative="greater")
    return {"comparacion": f"{a} vs {b}", "region": region, "metrica": metrica, "n_frames": len(t),
            "mediana_dif": float((t[a] - t[b]).median()), "p_valor": float(r.pvalue)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semilla", type=int, default=0, help="qué ajuste fino evaluar (modelos/<entrada>_s<semilla>)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prueba = json.load(open(c3vd.DIR_DATOS / "division.json"))["prueba"]
    modelos = {
        "A": (mp.PESOS_SCARED, "rgb"),
        "B": (mp.PESOS_SCARED, "rgb_limpio"),
        "C": (c3vd.DIR_MODELOS / f"rgb_s{args.semilla}", "rgb"),
        "D": (c3vd.DIR_MODELOS / f"rgb_limpio_s{args.semilla}", "rgb_limpio"),
        "E": (c3vd.DIR_MODELOS / f"conjunto_s{args.semilla}", f"rgb_conjunto_s{args.semilla}"),
    }
    filas = []
    for nombre, (pesos, entrada) in modelos.items():
        if not (pesos / "encoder.pth").exists():
            print(f"[omitido] {nombre}: no existe {pesos}")
            continue
        if not all((c3vd.DIR_DATOS / v / f"{entrada}.npy").exists() for v in prueba):
            print(f"[omitido] {nombre}: faltan los frames {entrada}.npy en los videos de prueba")
            continue
        enc, dec = mp.cargar(pesos, device)
        enc.eval(); dec.eval()
        for v in prueba:
            rgb, prof, mask = c3vd.cargar_video(v, entrada)
            with torch.no_grad():
                pred = np.concatenate([mp.predecir(enc, dec, mp.a_tensor(rgb[i:i + 16], device))[:, 0].cpu().numpy()
                                       for i in range(0, len(rgb), 16)])
            for i in range(len(rgb)):
                g = np.asarray(prof[i], np.float32)
                for reg, m in regiones(np.asarray(mask[i])).items():
                    r = mp.metricas(pred[i], g, m)
                    if r:
                        filas.append({"modelo": nombre, "video": v, "frame": i, "region": reg, **r})
        print(f"{nombre}: listo")

    if not filas:
        raise SystemExit("No hubo nada que evaluar (¿faltan modelos o videos de prueba?)")
    df = pd.DataFrame(filas)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTADOS / "evaluacion_por_frame.csv", index=False)
    resumen = df.groupby(["region", "modelo"])[METRICAS].mean().round(4)
    resumen.to_csv(RESULTADOS / "evaluacion_resumen.csv")

    pruebas = [p for p in (wilcoxon(df, "C", "D"), wilcoxon(df, "A", "B"),
                           wilcoxon(df, "C", "D", "imagen"),
                           wilcoxon(df, "D", "E"), wilcoxon(df, "D", "E", "imagen")) if p]
    with open(RESULTADOS / "evaluacion_resumen.md", "w") as f:
        f.write(f"Videos de prueba: {', '.join(prueba)}  \nAjuste fino: semilla {args.semilla}\n\n")
        for reg in ["imagen", "reflejos", "anillo"]:
            if reg in resumen.index.get_level_values(0):
                f.write(f"### Región: {reg}\n\n{resumen.loc[reg].to_markdown()}\n\n")
        if pruebas:
            f.write("### Wilcoxon pareado (H1: el segundo modelo tiene menor AbsRel)\n\n")
            f.write(pd.DataFrame(pruebas).to_markdown(index=False) + "\n")
    print(open(RESULTADOS / "evaluacion_resumen.md").read())
    print(f"Resultados -> {RESULTADOS}")


if __name__ == "__main__":
    main()
