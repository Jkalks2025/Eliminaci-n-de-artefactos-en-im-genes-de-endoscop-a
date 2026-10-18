"""Avance 4: ajuste de punta a punta de Endo-STTN + AF-SfMLearner (modelo E).

    python Avance4_EntrenamientoConjunto/ajustar_conjunto.py
    python Avance4_EntrenamientoConjunto/ajustar_conjunto.py --sttn-capas transformer --peso-hueco 1 --peso-ancla 0.5

En D, Endo-STTN está congelado: limpia los frames una vez y el estimador aprende con ellos. En E la
pérdida de profundidad también entrena a Endo-STTN: el relleno de los reflejos se ajusta para que el
estimador acierte. Como así Endo-STTN podría "pintar" cualquier cosa que le sirva al estimador, se
añaden dos pérdidas de imagen para que el relleno siga pareciéndose a tejido:

- hueco: como en el entrenamiento de Endo-STTN, la máscara de reflejos desplazada DESPLAZAMIENTO px
  se pone sobre tejido sano y Endo-STTN debe reconstruir esos píxeles, que sí se conocen.
- ancla (opcional): L1 entre el relleno y el del Endo-STTN original (rgb_limpio.npy), en los reflejos.

    pérdida = SI-log(profundidad) + peso_hueco * L1_hueco + peso_ancla * L1_ancla

El estimador parte de los mismos pesos (SCARED) y con la misma configuración que C y D, de modo que
E vs D aísla el efecto de entrenar también a Endo-STTN. Endo-STTN trabaja con clips de --frames
frames (consecutivos o al azar, como en su entrenamiento). Guarda el mejor modelo (menor AbsRel en
validación) en modelos/conjunto_s<semilla>/ (encoder.pth, depth.pth, sttn.pth, historial.json) y
escribe rgb_conjunto_s<semilla>.npy (frames limpiados por el Endo-STTN ajustado) en los videos de
validación y prueba, para que evaluar_profundidad.py lo compare como modelo E.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codigo"))
import c3vd  # noqa: E402
import limpiar_endosttn as le  # noqa: E402
import modelo_profundidad as mp  # noqa: E402

DESPLAZAMIENTO = 56   # px a 320x256; equivale a los 50 px a 288x288 de Endo-STTN (core/dataset.py)


class Clips(torch.utils.data.Dataset):
    """Clips de T frames de un video: la mitad de las veces consecutivos y la otra mitad al azar
    (ordenados), como get_ref_index de Endo-STTN."""

    def __init__(self, videos, T):
        self.datos = [c3vd.cargar_video(v) + (c3vd.cargar_video(v, "rgb_limpio")[0],) for v in videos]
        self.T = T
        self.indices = [(k, i) for k, d in enumerate(self.datos) for i in range(0, len(d[0]) - T + 1, T)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, n):
        k, i = self.indices[n]
        rgb, prof, mask, limpio = self.datos[k]
        ids = sorted(random.sample(range(len(rgb)), self.T)) if random.random() < 0.5 else range(i, i + self.T)
        ids = list(ids)
        x = np.stack([rgb[j] for j in ids]).astype(np.float32) / 255
        lim = np.stack([limpio[j] for j in ids]).astype(np.float32) / 255
        m = np.stack([mask[j] for j in ids]).astype(np.float32)
        g = np.stack([prof[j] for j in ids]).astype(np.float32)
        if random.random() < 0.5:                                        # volteo horizontal del clip
            x, lim, m, g = x[:, :, ::-1], lim[:, :, ::-1], m[:, :, ::-1], g[:, :, ::-1]
        brillo = np.array([random.uniform(0.9, 1.1), random.uniform(-0.05, 0.05)], np.float32)
        a_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        return (a_t(x.transpose(0, 3, 1, 2)), a_t(lim.transpose(0, 3, 1, 2)), a_t(m[:, None]),
                a_t(g[:, None]), torch.from_numpy(brillo))


def desplazar(m, dx=DESPLAZAMIENTO):
    """Máscara desplazada dx px a la derecha; por la izquierda entra cero (como cv2.warpAffine)."""
    out = torch.zeros_like(m)
    out[..., dx:] = m[..., :-dx]
    return out


def rellenar(sttn, x, hueco):
    """Relleno de Endo-STTN, diferenciable. x (B,T,3,H,W) en [0,1], hueco (B,T,1,H,W) -> (B,T,3,H,W).

    Mismo pre y postproceso que limpiar_endosttn.limpiar_bloque, pero en torch: 288x288, [-1,1],
    máscara por vecino más cercano y regreso a HxW bilineal.
    """
    B, T, _, H, W = x.shape
    f = F.interpolate(x.flatten(0, 1), (le.LADO, le.LADO), mode="area") * 2 - 1
    m = F.interpolate(hueco.flatten(0, 1), (le.LADO, le.LADO), mode="nearest")
    out = sttn((f * (1 - m)).view(B, T, 3, le.LADO, le.LADO), m.view(B, T, 1, le.LADO, le.LADO))
    out = F.interpolate((out + 1) / 2, (H, W), mode="bilinear", align_corners=False)
    return out.view(B, T, 3, H, W)


def l1_en(a, b, region):
    """L1 media dentro de una región (B,T,1,H,W); 0 si la región está vacía."""
    return ((a - b).abs() * region).sum() / (region.sum() * a.shape[2]).clamp(min=1)


@torch.no_grad()
def validar(sttn, enc, dec, videos, device, bloque):
    """AbsRel de profundidad y cambio medio del relleno respecto al Endo-STTN original (0-255)."""
    sttn.eval(); enc.eval(); dec.eval()
    absrel, cambio = [], []
    for v in videos:
        rgb, prof, mask = c3vd.cargar_video(v)
        mask = np.asarray(mask)
        limpio = le.limpiar_video(sttn, rgb, mask, device, bloque)
        if mask.any():
            original = np.asarray(c3vd.cargar_video(v, "rgb_limpio")[0])
            cambio.append(np.abs(limpio.astype(np.int16) - original)[mask].mean())
        for i in range(0, len(limpio), 16):
            pred = mp.predecir(enc, dec, mp.a_tensor(limpio[i:i + 16], device))[:, 0].cpu().numpy()
            for p, g in zip(pred, np.asarray(prof[i:i + 16], np.float32)):
                m = mp.metricas(p, g)
                if m:
                    absrel.append(m["abs_rel"])
    sttn.train(); enc.train(); dec.train()
    media = lambda l: float(np.mean(l)) if l else float("nan")
    return media(absrel), media(cambio)


def guardar(sttn, enc, dec, carpeta):
    mp.guardar(enc, dec, carpeta)
    torch.save({"netG": sttn.state_dict()}, carpeta / "sttn.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos-sttn", default=str(le.ENDO_STTN / "release_model" / "pretrained_model" / "gen_00009.pth"))
    ap.add_argument("--sttn-capas", choices=["todo", "transformer"], default="todo",
                    help="qué partes de Endo-STTN se entrenan (transformer: encoder y decoder congelados)")
    ap.add_argument("--epocas", type=int, default=15)
    ap.add_argument("--lote", type=int, default=2, help="clips por lote (cada clip tiene --frames frames)")
    ap.add_argument("--frames", type=int, default=5, help="frames por clip (sample_length de Endo-STTN)")
    ap.add_argument("--lr", type=float, default=1e-4, help="decoder de profundidad; su encoder usa lr/10")
    ap.add_argument("--lr-sttn", type=float, default=1e-5, help="Endo-STTN (se entrenó con 1e-4)")
    ap.add_argument("--peso-hueco", type=float, default=1.0)
    ap.add_argument("--peso-ancla", type=float, default=0.0)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--bloque", type=int, default=60, help="frames por bloque al limpiar (validación)")
    args = ap.parse_args()

    random.seed(args.semilla); np.random.seed(args.semilla); torch.manual_seed(args.semilla)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    div = json.load(open(c3vd.DIR_DATOS / "division.json"))
    if not div["entrenamiento"] or not div["validacion"]:
        raise SystemExit(f"Hacen falta videos de entrenamiento y de validación: {div}")
    salida = c3vd.DIR_MODELOS / f"conjunto_s{args.semilla}"
    entrada_e = f"rgb_conjunto_s{args.semilla}"

    datos = Clips(div["entrenamiento"], args.frames)
    gen = torch.Generator().manual_seed(args.semilla)
    carga = torch.utils.data.DataLoader(datos, batch_size=args.lote, shuffle=True, num_workers=0,
                                        drop_last=True, generator=gen)
    sttn = le.cargar(args.pesos_sttn, device).train()
    if args.sttn_capas == "transformer":
        for p in [*sttn.encoder.parameters(), *sttn.decoder.parameters()]:
            p.requires_grad = False
    enc, dec = mp.cargar(mp.PESOS_SCARED, device)
    opt = torch.optim.Adam([
        {"params": enc.parameters(), "lr": args.lr / 10},
        {"params": dec.parameters(), "lr": args.lr},
        # betas de Endo-STTN (configs/EndoSTTN_dataset.json)
        {"params": [p for p in sttn.parameters() if p.requires_grad], "lr": args.lr_sttn, "betas": (0.0, 0.99)},
    ])
    print(f"E: {len(datos)} clips de {args.frames} frames de {len(div['entrenamiento'])} videos | "
          f"Endo-STTN entrena: {args.sttn_capas} | {device}")

    historial = {"args": vars(args), "division": div, "epocas": []}
    mejor, cambio = validar(sttn, enc, dec, div["validacion"], device, args.bloque)
    print(f"Época 0 (SCARED + Endo-STTN original): AbsRel validación = {mejor:.4f}")
    historial["epocas"].append({"epoca": 0, "abs_rel_val": mejor, "cambio_relleno_val": cambio})
    guardar(sttn, enc, dec, salida)

    for ep in range(1, args.epocas + 1):
        t0, suma = time.time(), {"prof": [], "hueco": [], "ancla": [], "total": []}
        for x, limpio, m, gt, brillo in carga:
            x, limpio, m, gt, brillo = (t.to(device) for t in (x, limpio, m, gt, brillo))
            pseudo = desplazar(m) * (1 - m)                     # hueco sobre tejido sano (verdad conocida)
            relleno = rellenar(sttn, x, (m + pseudo).clamp(max=1))
            x_limpio = x * (1 - m) + relleno * m                 # solo se pega dentro de los reflejos
            a, b = brillo[:, 0].view(-1, 1, 1, 1, 1), brillo[:, 1].view(-1, 1, 1, 1, 1)
            entrada = (x_limpio * a + b).clamp(0, 1).flatten(0, 1)
            l_prof = mp.perdida_si_log(mp.predecir(enc, dec, entrada), gt.flatten(0, 1))
            l_hueco = l1_en(relleno, x, pseudo)
            l_ancla = l1_en(relleno, limpio, m)
            loss = l_prof + args.peso_hueco * l_hueco + args.peso_ancla * l_ancla
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in zip(suma, (l_prof, l_hueco, l_ancla, loss)):
                suma[k].append(v.item())
        val, cambio = validar(sttn, enc, dec, div["validacion"], device, args.bloque)
        fila = {"epoca": ep, **{f"perdida_{k}": float(np.mean(v)) for k, v in suma.items()},
                "abs_rel_val": val, "cambio_relleno_val": cambio}
        historial["epocas"].append(fila)
        marca = ""
        if val < mejor:
            mejor, marca = val, "  <- mejor, guardado"
            guardar(sttn, enc, dec, salida)
        print(f"Época {ep}: prof {fila['perdida_prof']:.4f} | hueco {fila['perdida_hueco']:.4f} | "
              f"ancla {fila['perdida_ancla']:.4f} | AbsRel validación {val:.4f} | "
              f"cambio del relleno {cambio:.1f}/255 | {time.time() - t0:.0f} s{marca}")
        json.dump(historial, open(salida / "historial.json", "w"), indent=2)

    historial["mejor_abs_rel_val"] = mejor
    json.dump(historial, open(salida / "historial.json", "w"), indent=2)
    print(f"Mejor modelo -> {salida}")

    # Frames limpiados por el Endo-STTN ajustado, para evaluar E con evaluar_profundidad.py
    sttn = le.cargar(salida / "sttn.pth", device)
    for v in div["validacion"] + div["prueba"]:
        rgb, _, mask = c3vd.cargar_video(v)
        np.save(c3vd.DIR_DATOS / v / f"{entrada_e}.npy", le.limpiar_video(sttn, rgb, mask, device, args.bloque))
        print(f"{v}: {entrada_e}.npy")


if __name__ == "__main__":
    main()
