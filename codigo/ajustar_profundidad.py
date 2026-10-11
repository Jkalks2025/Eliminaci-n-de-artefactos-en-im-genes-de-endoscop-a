"""Paso 3: ajuste fino supervisado de AF-SfMLearner con la profundidad real de C3VDv2.

    python codigo/ajustar_profundidad.py --entrada rgb           # modelo C: frames originales
    python codigo/ajustar_profundidad.py --entrada rgb_limpio    # modelo D: frames limpiados por Endo-STTN

Ambos parten de los pesos de AF-SfMLearner entrenados en SCARED y usan exactamente la misma
configuración; solo cambian los frames de entrada. Guarda el mejor modelo (menor AbsRel en
validación) en modelos/<entrada>_s<semilla>/ junto con historial.json.
"""
import argparse
import json
import random
import time

import numpy as np
import torch

import c3vd
import modelo_profundidad as mp


class Frames(torch.utils.data.Dataset):
    def __init__(self, videos, entrada, aumentar):
        self.datos = [c3vd.cargar_video(v, entrada)[:2] for v in videos]
        self.indices = [(k, i) for k, (rgb, _) in enumerate(self.datos) for i in range(len(rgb))]
        self.aumentar = aumentar

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, n):
        k, i = self.indices[n]
        rgb = np.array(self.datos[k][0][i], dtype=np.float32) / 255
        prof = np.array(self.datos[k][1][i], dtype=np.float32)
        if self.aumentar:
            if random.random() < 0.5:                                    # volteo horizontal
                rgb, prof = rgb[:, ::-1], prof[:, ::-1]
            rgb = rgb * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)   # brillo/contraste suaves
        x = torch.from_numpy(np.clip(rgb, 0, 1).transpose(2, 0, 1).copy())
        return x, torch.from_numpy(prof.copy())[None]


@torch.no_grad()
def validar(enc, dec, videos, entrada, device):
    enc.eval(); dec.eval()
    vals = []
    for v in videos:
        rgb, prof, _ = c3vd.cargar_video(v, entrada)
        for i in range(0, len(rgb), 16):
            pred = mp.predecir(enc, dec, mp.a_tensor(rgb[i:i + 16], device))[:, 0].cpu().numpy()
            for p, g in zip(pred, np.asarray(prof[i:i + 16], np.float32)):
                m = mp.metricas(p, g)
                if m:
                    vals.append(m["abs_rel"])
    enc.train(); dec.train()
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", choices=["rgb", "rgb_limpio"], required=True)
    ap.add_argument("--epocas", type=int, default=15)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4, help="decoder; el encoder usa lr/10")
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.semilla); np.random.seed(args.semilla); torch.manual_seed(args.semilla)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    div = json.load(open(c3vd.DIR_DATOS / "division.json"))
    if not div["entrenamiento"] or not div["validacion"]:
        raise SystemExit(f"Hacen falta videos de entrenamiento y de validación: {div}")
    salida = c3vd.DIR_MODELOS / f"{args.entrada}_s{args.semilla}"

    datos = Frames(div["entrenamiento"], args.entrada, aumentar=True)
    gen = torch.Generator().manual_seed(args.semilla)
    carga = torch.utils.data.DataLoader(datos, batch_size=args.lote, shuffle=True, num_workers=0,
                                        drop_last=True, generator=gen)
    enc, dec = mp.cargar(mp.PESOS_SCARED, device)
    opt = torch.optim.Adam([{"params": enc.parameters(), "lr": args.lr / 10},
                            {"params": dec.parameters(), "lr": args.lr}])
    print(f"Entrada: {args.entrada} | {len(datos)} frames de {len(div['entrenamiento'])} videos | {device}")

    historial = {"args": vars(args), "division": div, "epocas": []}
    mejor = validar(enc, dec, div["validacion"], args.entrada, device)
    print(f"Época 0 (pesos SCARED): AbsRel validación = {mejor:.4f}")
    historial["epocas"].append({"epoca": 0, "perdida": None, "abs_rel_val": mejor})
    mp.guardar(enc, dec, salida)

    for ep in range(1, args.epocas + 1):
        t0, perdidas = time.time(), []
        for x, gt in carga:
            x, gt = x.to(device), gt.to(device)
            loss = mp.perdida_si_log(mp.predecir(enc, dec, x), gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            perdidas.append(loss.item())
        val = validar(enc, dec, div["validacion"], args.entrada, device)
        historial["epocas"].append({"epoca": ep, "perdida": float(np.mean(perdidas)), "abs_rel_val": val})
        marca = ""
        if val < mejor:
            mejor, marca = val, "  <- mejor, guardado"
            mp.guardar(enc, dec, salida)
        print(f"Época {ep}: pérdida {np.mean(perdidas):.4f} | AbsRel validación {val:.4f} | {time.time() - t0:.0f} s{marca}")
        json.dump(historial, open(salida / "historial.json", "w"), indent=2)

    historial["mejor_abs_rel_val"] = mejor
    json.dump(historial, open(salida / "historial.json", "w"), indent=2)
    print(f"Mejor modelo -> {salida}")


if __name__ == "__main__":
    main()
