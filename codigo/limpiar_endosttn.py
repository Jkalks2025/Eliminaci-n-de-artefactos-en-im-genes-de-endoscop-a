"""Paso 2: borra los reflejos de cada video preparado con Endo-STTN (congelado, sin entrenar).

    python codigo/limpiar_endosttn.py
    python codigo/limpiar_endosttn.py --pesos Endo-STTN/release_model/pretrained_model/gen_00009.pth

Para cada datos/c3vd/<video>/ crea rgb_limpio.npy: igual a rgb.npy fuera de la máscara y con el
relleno de Endo-STTN dentro. Usa la misma inferencia que Endo-STTN/test.py (frames vecinos +
frames de referencia), por bloques de frames para no agotar la memoria de la GPU.
"""
import argparse
import sys

import cv2
import numpy as np
import torch

import c3vd

ENDO_STTN = c3vd.RAIZ / "Endo-STTN"
sys.path.insert(0, str(ENDO_STTN))
from model.sttn import InpaintGenerator  # noqa: E402

LADO = 288            # resolución de Endo-STTN
NEIGHBOR_STRIDE = 5   # como en Endo-STTN/test.py
REF_LENGTH = 10


def referencias(vecinos, n):
    return [i for i in range(0, n, REF_LENGTH) if i not in vecinos]


@torch.no_grad()
def limpiar_bloque(modelo, rgb, mask, device):
    """rgb (T,H,W,3) uint8, mask (T,H,W) bool -> (T,H,W,3) uint8 con los reflejos rellenados."""
    T, H, W = mask.shape
    f = np.stack([cv2.resize(x, (LADO, LADO), interpolation=cv2.INTER_AREA) for x in rgb])
    m = np.stack([cv2.resize(x.astype(np.uint8), (LADO, LADO), interpolation=cv2.INTER_NEAREST) for x in mask])
    feats = torch.from_numpy(f).permute(0, 3, 1, 2).float().div(255).mul(2).sub(1).to(device)   # [-1,1]
    masks = torch.from_numpy(m).float().unsqueeze(1).to(device)
    enc = modelo.encoder(feats * (1 - masks))
    _, c, fh, fw = enc.shape
    enc = enc.view(1, T, c, fh, fw)

    acum = np.zeros((T, LADO, LADO, 3), np.float32)
    cuenta = np.zeros(T, np.float32)
    for k in range(0, T, NEIGHBOR_STRIDE):
        vec = list(range(max(0, k - NEIGHBOR_STRIDE), min(T, k + NEIGHBOR_STRIDE + 1)))
        ids = vec + referencias(vec, T)
        pred = modelo.infer(enc[0, ids], masks[ids])
        pred = torch.tanh(modelo.decoder(pred[:len(vec)]))
        pred = ((pred + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255
        for i, idx in enumerate(vec):
            acum[idx] += pred[i]
            cuenta[idx] += 1
    pred = acum / cuenta[:, None, None, None]

    out = rgb.copy()
    for t in range(T):
        relleno = cv2.resize(pred[t], (W, H), interpolation=cv2.INTER_LINEAR)
        out[t][mask[t]] = np.clip(relleno[mask[t]], 0, 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default=str(ENDO_STTN / "release_model" / "pretrained_model" / "gen_00009.pth"))
    ap.add_argument("--bloque", type=int, default=60, help="frames por bloque (bájalo si falta memoria)")
    ap.add_argument("--rehacer", action="store_true", help="volver a limpiar videos ya limpiados")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = InpaintGenerator().to(device)
    modelo.load_state_dict(torch.load(args.pesos, map_location=device)["netG"])
    modelo.eval()
    print(f"Endo-STTN: {args.pesos} | {device}")

    videos = c3vd.videos_preparados()
    if not videos:
        raise SystemExit("No hay videos preparados: corre primero  python codigo/preparar_datos.py")
    for k, video in enumerate(videos, 1):
        destino = c3vd.DIR_DATOS / video / "rgb_limpio.npy"
        if destino.exists() and not args.rehacer:
            print(f"[{k}/{len(videos)}] {video}: ya limpio")
            continue
        rgb, _, mask = c3vd.cargar_video(video)
        rgb, mask = np.asarray(rgb), np.asarray(mask)
        out = np.concatenate([limpiar_bloque(modelo, rgb[i:i + args.bloque], mask[i:i + args.bloque], device)
                              for i in range(0, len(rgb), args.bloque)])
        np.save(destino, out)
        cambio = np.abs(out.astype(np.int16) - rgb).mean(-1)
        print(f"[{k}/{len(videos)}] {video}: {len(rgb)} frames | píxeles modificados fuera de la máscara: "
              f"{int((cambio[~mask] > 0).sum())}")


if __name__ == "__main__":
    main()
