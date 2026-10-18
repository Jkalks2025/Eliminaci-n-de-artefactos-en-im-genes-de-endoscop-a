"""Re-entrenar Endo, paso 3: compara el Endo-STTN original contra el reentrenado con C3VDv2.

    python ReEntrenar_Endo/evaluar_endosttn.py

Como no existen frames sin reflejos de referencia, se mide igual que en el artículo de Endo-STTN:
la máscara de reflejos se desplaza 50 px sobre tejido sano (pseudo ground truth), el modelo rellena
ese hueco y se compara contra el color real. Métricas dentro del hueco: MAE (0-255, menor es mejor)
y PSNR (dB, mayor es mejor).

1. Validación: evalúa el original y cada checkpoint gen_XXXXX.pth; el de menor MAE es el "reentrenado".
   Lo copia a modelos/endosttn_c3vd/gen_mejor.pth (mismo formato que gen_00009.pth).
2. Prueba: original vs reentrenado, por frame, con prueba de Wilcoxon pareada.

Escribe en ReEntrenar_Endo/resultados/: validacion_por_checkpoint.csv,
prueba_por_frame.csv y resumen.md.
"""
import argparse
import json
import shutil

import numpy as np
import pandas as pd
import torch

import endosttn_c3vd as ec
import limpiar_endosttn as le


def evaluar_video(modelo, video, device, bloque):
    """Métricas por frame en el hueco desplazado y frames limpios de los reflejos reales."""
    rgb = np.load(ec.DIR_EVAL / video / "rgb.npy")
    mask = np.load(ec.DIR_EVAL / video / "mask.npy")
    hueco = ec.hueco_pseudo(mask)
    relleno = le.limpiar_video(modelo, rgb, hueco, device, bloque)
    filas = []
    for i in range(len(rgb)):
        m = ec.metricas_hueco(relleno[i], rgb[i], hueco[i])
        if m:
            filas.append({"video": video, "frame": i, **m})
    return filas


def media(filas, clave):
    return float(np.mean([f[clave] for f in filas])) if filas else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos-originales", default=str(ec.PESOS_ORIGINALES), help="Endo-STTN publicado")
    ap.add_argument("--bloque", type=int, default=60, help="frames por bloque (bájalo si falta memoria)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    div = json.load(open(ec.DIR_DATASET / "division.json"))
    guardar_cada = json.load(open(ec.DIR_ENTRENAMIENTO / "config.json"))["trainer"]["save_freq"]
    checkpoints = sorted(ec.DIR_ENTRENAMIENTO.glob("gen_[0-9]*.pth"))
    if not checkpoints:
        raise SystemExit(f"No hay checkpoints en {ec.DIR_ENTRENAMIENTO}: corre primero entrenar_endosttn.py")
    modelos = [("original", 0, args.pesos_originales)] + [
        (p.stem, int(p.stem.split("_")[1]) * guardar_cada, p) for p in checkpoints]
    ec.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # 1. Validación: elegir checkpoint
    val = div["validacion"]
    filas_val = []
    if val:
        for nombre, it, pesos in modelos:
            modelo = le.cargar(pesos, device)
            filas = [f for v in val for f in evaluar_video(modelo, v, device, args.bloque)]
            filas_val.append({"modelo": nombre, "iteracion": it, "mae": media(filas, "mae"),
                              "psnr": media(filas, "psnr"), "frames": len(filas)})
            print(f"validación {nombre:10s} (iteración {it:6d}): MAE {filas_val[-1]['mae']:.2f} | "
                  f"PSNR {filas_val[-1]['psnr']:.2f} dB", flush=True)
        tabla_val = pd.DataFrame(filas_val)
        tabla_val.to_csv(ec.DIR_RESULTADOS / "validacion_por_checkpoint.csv", index=False)
        reent = tabla_val[tabla_val.modelo != "original"].sort_values("mae").iloc[0]
        nombre_mejor = reent.modelo
    else:
        nombre_mejor = checkpoints[-1].stem
        print("Sin videos de validación: se usa el último checkpoint.")
    mejor = ec.DIR_ENTRENAMIENTO / f"{nombre_mejor}.pth"
    shutil.copy(mejor, ec.DIR_ENTRENAMIENTO / "gen_mejor.pth")
    it_mejor = int(nombre_mejor.split("_")[1]) * guardar_cada
    json.dump({"checkpoint": nombre_mejor, "iteracion": it_mejor},
              open(ec.DIR_ENTRENAMIENTO / "mejor.json", "w"), indent=2)
    print(f"Reentrenado = {nombre_mejor} (iteración {it_mejor}) -> {ec.DIR_ENTRENAMIENTO / 'gen_mejor.pth'}")

    # 2. Prueba: original vs reentrenado
    prueba = div["prueba"]
    if not prueba:
        raise SystemExit("No hay videos de prueba en la división.")
    filas = []
    for nombre, pesos in [("original", args.pesos_originales), ("reentrenado", mejor)]:
        modelo = le.cargar(pesos, device)
        for v in prueba:
            filas += [{"modelo": nombre, **f} for f in evaluar_video(modelo, v, device, args.bloque)]
    df = pd.DataFrame(filas)
    df.to_csv(ec.DIR_RESULTADOS / "prueba_por_frame.csv", index=False)
    resumen = df.groupby("modelo")[["mae", "psnr"]].mean()

    t = df.pivot_table(index=["video", "frame"], columns="modelo", values="mae").dropna()
    wil = None
    if len(t) >= 10:
        from scipy.stats import wilcoxon
        wil = {"frames": len(t), "mediana_dif_mae": float((t.original - t.reentrenado).median()),
               "p_valor": float(wilcoxon(t.original, t.reentrenado, alternative="greater").pvalue)}

    with open(ec.DIR_RESULTADOS / "resumen.md", "w") as f:
        f.write("# Re-entrenar Endo · Endo-STTN original vs reentrenado con C3VDv2\n\n")
        f.write(f"Videos de validación: {', '.join(val) or '(ninguno)'}  \nVideos de prueba: {', '.join(prueba)}  \n")
        f.write(f"Reentrenado: `{nombre_mejor}.pth` (iteración {it_mejor})\n\n")
        f.write("## Prueba: error del relleno en el hueco desplazado (pseudo ground truth)\n\n")
        f.write("MAE en valores de color 0-255 (menor es mejor); PSNR en dB (mayor es mejor).\n\n")
        f.write(resumen.round(3).to_markdown() + "\n\n")
        if wil:
            f.write(f"Wilcoxon pareado por frame (H1: el reentrenado tiene menor MAE): {wil['frames']} frames, "
                    f"mediana de la diferencia {wil['mediana_dif_mae']:.3f}, p = {wil['p_valor']:.4g}\n\n")
        if filas_val:
            f.write("## Validación por checkpoint\n\n")
            f.write(tabla_val.round(3).to_markdown(index=False) + "\n")
    print("\n" + resumen.round(3).to_string())
    if wil:
        print(f"Wilcoxon: p = {wil['p_valor']:.4g} ({wil['frames']} frames)")
    print(f"Resultados -> {ec.DIR_RESULTADOS}")


if __name__ == "__main__":
    main()
