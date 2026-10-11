"""AF-SfMLearner: carga, predicción de profundidad, pérdida de entrenamiento y métricas."""
import os
import sys
from pathlib import Path

import numpy as np
import torch

import c3vd

AFSFM_DIR = c3vd.RAIZ / "externos" / "AF-SfMLearner"
PESOS_SCARED = Path(os.environ.get("EQUIPO19_PESOS_AF", c3vd.RAIZ / "externos" / "pesos" / "Model_trained_end_to_end"))
PROF_MIN, PROF_MAX = 0.1, c3vd.PROF_MAX_MM   # rango de profundidad en mm (como la profundidad real)


def _networks():
    if str(AFSFM_DIR) not in sys.path:
        sys.path.insert(0, str(AFSFM_DIR))
    import networks
    return networks


def cargar(pesos_dir=PESOS_SCARED, device="cpu"):
    """Encoder + decoder de AF-SfMLearner con los pesos de pesos_dir (encoder.pth, depth.pth)."""
    nets = _networks()
    pesos_dir = Path(pesos_dir)
    enc = nets.ResnetEncoder(18, False)
    sd = torch.load(pesos_dir / "encoder.pth", map_location=device)
    enc.load_state_dict({k: v for k, v in sd.items() if k in enc.state_dict()})
    dec = nets.DepthDecoder(num_ch_enc=enc.num_ch_enc, scales=range(4))
    dec.load_state_dict(torch.load(pesos_dir / "depth.pth", map_location=device))
    return enc.to(device), dec.to(device)


def guardar(enc, dec, carpeta):
    """Mismo formato que los pesos originales (incluye alto/ancho de entrada)."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    sd = enc.state_dict()
    sd.update(height=c3vd.ALTO, width=c3vd.ANCHO)
    torch.save(sd, carpeta / "encoder.pth")
    torch.save(dec.state_dict(), carpeta / "depth.pth")


def a_tensor(rgb_uint8, device):
    """(B,H,W,3) uint8 -> (B,3,H,W) en [0,1]. AF-SfMLearner no normaliza con media/desviación."""
    x = torch.as_tensor(np.ascontiguousarray(rgb_uint8)).permute(0, 3, 1, 2).float().div(255)
    return x.to(device)


def predecir(enc, dec, x):
    """Profundidad en mm, (B,1,H,W). Igual que disp_to_depth de Monodepth2/AF-SfMLearner."""
    disp = dec(enc(x))[("disp", 0)]
    min_disp, max_disp = 1 / PROF_MAX, 1 / PROF_MIN
    return 1 / (min_disp + (max_disp - min_disp) * disp)


def perdida_si_log(pred, gt, lam=1.0):
    """Pérdida scale-invariant log (Eigen et al., 2014), por imagen, sobre píxeles con profundidad real.

    Con lam=1 es totalmente invariante a la escala: solo importa la forma relativa, igual que en la
    evaluación con median scaling. (Con lam<1 también exigiría los milímetros reales, que una red
    monocular no puede conocer y que llevan la disparidad a la zona saturada de la sigmoide.)
    """
    perdidas = []
    for p, g in zip(pred, gt):
        valido = g > 0
        if valido.sum() < 10:
            continue
        d = torch.log(p[valido]) - torch.log(g[valido])
        perdidas.append((d ** 2).mean() - lam * d.mean() ** 2)
    return torch.stack(perdidas).mean()


def metricas(pred, gt, region=None):
    """Métricas estándar con median scaling. pred, gt: (H,W) en mm. region: máscara booleana opcional.

    La escala se calcula con toda la imagen y después se mide solo dentro de la región, para que
    medir en los reflejos no le dé a cada modelo una escala hecha a la medida de esa zona.
    """
    valido = (gt > 0) & (gt <= PROF_MAX)
    if valido.sum() < 10:
        return None
    escala = np.median(gt[valido]) / np.median(pred[valido])
    if region is not None:
        valido &= region
        if valido.sum() < 10:
            return None
    p, g = pred[valido].astype(np.float64), gt[valido].astype(np.float64)
    p = np.clip(p * escala, PROF_MIN, PROF_MAX)
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "sq_rel": float(np.mean((p - g) ** 2 / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "a1": float(np.mean(ratio < 1.25)),
    }
