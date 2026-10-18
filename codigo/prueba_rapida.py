"""Prueba de humo: corre los 4 pasos con datos sintéticos, sin descargar C3VDv2 ni pesos.

    python codigo/prueba_rapida.py

Necesita solo el código de AF-SfMLearner en externos/ (lo clona preparar_entorno.py).
Crea en una carpeta temporal ZIPs con el formato de C3VDv2 (rgb/ + depth/), pesos aleatorios de
AF-SfMLearner y Endo-STTN, y ejecuta preparar -> limpiar -> ajustar (C, D y E, 1 época) -> evaluar.
No dice nada de la calidad de los modelos: solo comprueba que el flujo funciona de punta a punta.
"""
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import c3vd  # noqa: E402

VIDEOS = {c3vd.VIDEOS_PRUEBA[0]: "prueba", c3vd.VIDEOS_VALIDACION[0]: "validación",
          "c1_ascending_t1_v1": "entrenamiento", "c2_descending_t4_v2": "entrenamiento"}


def zip_sintetico(ruta, n=12, rng=np.random.default_rng(0)):
    """Frames de 'colon' (gradiente + ruido) con un reflejo blanco y profundidad uint16 como C3VD."""
    with zipfile.ZipFile(ruta, "w") as z:
        for i in range(n):
            yy, xx = np.mgrid[0:540, 0:675]
            prof_mm = 20 + 40 * (yy / 540) + 5 * np.sin(xx / 60 + i / 3)
            rgb = np.stack([180 - prof_mm * 1.5, 90 - prof_mm * 0.6, 80 - prof_mm * 0.5], -1)
            rgb = np.clip(rgb + rng.normal(0, 6, rgb.shape), 0, 255).astype(np.uint8)
            cv2.circle(rgb, (300 + 5 * i, 260), 14, (255, 255, 255), -1)
            z.writestr(f"rgb/{i:04d}_color.png", cv2.imencode(".png", rgb[..., ::-1])[1].tobytes())
            d16 = (prof_mm / c3vd.PROF_MAX_MM * 65535).astype(np.uint16)
            z.writestr(f"depth/{i:04d}_depth.tiff", cv2.imencode(".tiff", d16)[1].tobytes())


def main():
    tmp = Path(tempfile.mkdtemp(prefix="equipo19_prueba_"))
    print("Carpeta temporal:", tmp)
    raw = tmp / "datos" / "raw"
    raw.mkdir(parents=True)
    for v in VIDEOS:
        zip_sintetico(raw / f"{v}.zip")

    # pesos aleatorios con las arquitecturas reales
    import modelo_profundidad as mp
    nets = mp._networks()
    enc = nets.ResnetEncoder(18, False)
    dec = nets.DepthDecoder(num_ch_enc=enc.num_ch_enc, scales=range(4))
    mp.guardar(enc, dec, tmp / "pesos_af")
    sys.path.insert(0, str(c3vd.RAIZ / "Endo-STTN"))
    from model.sttn import InpaintGenerator
    torch.save({"netG": InpaintGenerator().state_dict()}, tmp / "gen_aleatorio.pth")

    env = dict(os.environ, EQUIPO19_TRABAJO=str(tmp), EQUIPO19_RESULTADOS=str(tmp / "resultados"),
               EQUIPO19_PESOS_AF=str(tmp / "pesos_af"), PYTHONWARNINGS="ignore")
    pasos = [
        ["preparar_datos.py", "--paso", "1"],
        ["limpiar_endosttn.py", "--pesos", str(tmp / "gen_aleatorio.pth")],
        ["ajustar_profundidad.py", "--entrada", "rgb", "--epocas", "1", "--lote", "4"],
        ["ajustar_profundidad.py", "--entrada", "rgb_limpio", "--epocas", "1", "--lote", "4"],
        ["../Entrenamiento_end_to_end/ajustar_conjunto.py", "--pesos-sttn", str(tmp / "gen_aleatorio.pth"), "--epocas", "1", "--lote", "1"],
        ["evaluar_profundidad.py"],
    ]
    for paso in pasos:
        print(f"\n=== {' '.join(paso)} ===", flush=True)
        r = subprocess.run([sys.executable, str(AQUI / paso[0]), *paso[1:]], env=env)
        if r.returncode != 0:
            raise SystemExit(f"FALLÓ: {paso[0]}")

    assert (tmp / "resultados" / "evaluacion_resumen.csv").exists()
    print(f"\nPRUEBA RÁPIDA OK: los pasos corrieron de punta a punta (archivos en {tmp}).")


if __name__ == "__main__":
    main()
