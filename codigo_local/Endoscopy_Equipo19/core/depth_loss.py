"""Pérdida de profundidad para entrenar Endo-STTN guiado por MonoIIT.

MonoIIT (encoder mpvit_small + decoder HR-Depth de MonoViT) se usa CONGELADO:
sus pesos no se actualizan, pero el gradiente atraviesa sus activaciones y llega
al generador de Endo-STTN. La idea es que el frame reconstruido produzca la misma
disparidad que el frame limpio original (ground truth del entrenamiento).
"""
import os
import sys
import types
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_monoiit(monovit_path, weights_dir, device):
    """Carga MonoIIT igual que en el Avance 7 (paquete `networks` de MonoViT aislado).

    Devuelve (encoder, decoder, (alto, ancho) de entrada).
    """
    netdir = Path(monovit_path) / "networks"
    weights_dir = Path(weights_dir)

    # Registrar `networks` sin ejecutar su __init__ (evita dependencias extra)
    for k in list(sys.modules):
        if k == "networks" or k.startswith("networks."):
            del sys.modules[k]
    pkg = types.ModuleType("networks")
    pkg.__path__ = [str(netdir)]
    sys.modules["networks"] = pkg

    def _load(name, fn):
        spec = importlib.util.spec_from_file_location(f"networks.{name}", str(netdir / fn))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"networks.{name}"] = mod
        spec.loader.exec_module(mod)
        setattr(pkg, name, mod)
        return mod

    _load("hr_layers", "hr_layers.py")
    DepthDecoderHR = _load("hr_decoder", "hr_decoder.py").DepthDecoder
    mpvit = _load("mpvit", "mpvit.py")

    # mpvit_small() lee './ckpt/mpvit_small.pth' relativo al cwd: se construye desde MonoViT
    cwd = os.getcwd()
    os.chdir(monovit_path)
    try:
        enc = mpvit.mpvit_small()
    except FileNotFoundError:
        # Sin el checkpoint ImageNet: misma arquitectura; encoder.pth sobrescribe los pesos igual
        enc = mpvit.MPViT(num_stages=4, num_path=[2, 3, 3, 3], num_layers=[1, 3, 6, 3],
                          embed_dims=[64, 128, 216, 288], mlp_ratios=[4, 4, 4, 4], num_heads=[8, 8, 8, 8])
    finally:
        os.chdir(cwd)
    enc.num_ch_enc = [64, 128, 216, 288, 288]
    enc_sd = torch.load(weights_dir / "encoder.pth", map_location=device)
    in_h, in_w = enc_sd.get("height", 192), enc_sd.get("width", 640)
    enc.load_state_dict({k: v for k, v in enc_sd.items() if k in enc.state_dict()})

    dec = DepthDecoderHR()
    r = dec.load_state_dict(torch.load(weights_dir / "depth.pth", map_location=device), strict=False)
    if r.missing_keys:
        print(f"[DepthLoss] AVISO: faltan {len(r.missing_keys)} pesos en el decoder de MonoIIT")

    return enc.to(device), dec.to(device), (in_h, in_w)


def load_monodepth2(edam_path, weights_dir, device, num_layers=18):
    """Carga Monodepth2 con el código de Endo-Depth-and-Motion (igual que Avance 3/5).

    El código está en `<edam_path>/apps/depth_estimate/` (resnet_encoder.py,
    depth_decoder.py, layers.py). Devuelve (encoder, decoder, (alto, ancho) de entrada).
    """
    code_dir = str(Path(edam_path) / "apps" / "depth_estimate")
    weights_dir = Path(weights_dir)
    for k in ("resnet_encoder", "depth_decoder", "layers"):
        sys.modules.pop(k, None)
    sys.path.insert(0, code_dir)
    try:
        from resnet_encoder import ResnetEncoder
        from depth_decoder import DepthDecoder
    finally:
        sys.path.remove(code_dir)

    enc = ResnetEncoder(num_layers, False)
    enc_sd = torch.load(weights_dir / "encoder.pth", map_location=device)
    in_h, in_w = enc_sd.get("height", 192), enc_sd.get("width", 640)
    enc.load_state_dict({k: v for k, v in enc_sd.items() if k in enc.state_dict()})

    dec = DepthDecoder(num_ch_enc=enc.num_ch_enc, scales=range(4))
    dec.load_state_dict(torch.load(weights_dir / "depth.pth", map_location=device))

    return enc.to(device), dec.to(device), (in_h, in_w)


def scale_invariant_l1(pred, target, weight=None, eps=1e-6):
    """L1 tras normalizar cada mapa por mediana y desviación absoluta media.

    La profundidad monocular tiene escala/desplazamiento ambiguos; así la pérdida
    compara la forma relativa y no la escala.
    """
    def _norm(d):
        flat = d.flatten(1)
        med = flat.median(dim=1).values.view(-1, 1, 1, 1)
        scale = (d - med).abs().flatten(1).mean(dim=1).view(-1, 1, 1, 1) + eps
        return (d - med) / scale

    diff = (_norm(pred) - _norm(target)).abs()
    if weight is None:
        return diff.mean()
    return (diff * weight).sum() / weight.sum().clamp_min(eps)


class DepthLoss(nn.Module):
    """L_depth(comp_img, frames): disparidad de MonoIIT del frame reconstruido vs. del frame limpio.

    Entradas (como en core/trainer.py de Endo-STTN):
        pred   (N,3,H,W) en [-1,1], con gradiente (comp_img)
        target (N,3,H,W) en [-1,1], frames limpios
        masks  (N,1,H,W), 1 = región a rellenar
    """

    def __init__(self, encoder, decoder, in_size, mask_weight=4.0, max_frames=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.in_size = tuple(in_size)
        self.mask_weight = mask_weight
        self.max_frames = max_frames
        for p in self.parameters():
            p.requires_grad_(False)   # congelado: no se entrena
        self.eval()

    @classmethod
    def from_config(cls, cfg, device):
        model = cfg.get("model", "monoiit")
        if model == "monoiit":
            enc, dec, in_size = load_monoiit(cfg["monovit_path"], cfg["weights_dir"], device)
        elif model == "monodepth2":
            enc, dec, in_size = load_monodepth2(cfg["edam_path"], cfg["weights_dir"], device,
                                                num_layers=cfg.get("num_layers", 18))
        else:
            raise ValueError(f"depth.model desconocido: {model} (usa 'monoiit' o 'monodepth2')")
        return cls(enc, dec, in_size,
                   mask_weight=cfg.get("mask_weight", 4.0),
                   max_frames=cfg.get("max_frames", 0))

    def train(self, mode=True):
        # Siempre en eval (BatchNorm/Dropout fijos), aunque alguien llame .train()
        return super().train(False)

    def _disp(self, x):
        # [-1,1] -> [0,1] (MonoIIT y Monodepth2 reciben ToTensor; normalizan dentro del encoder si hace falta)
        x = (x + 1) / 2
        x = F.interpolate(x, size=self.in_size, mode="bilinear", align_corners=False)
        return self.decoder(self.encoder(x))[("disp", 0)]

    def forward(self, pred, target, masks):
        if self.max_frames and pred.size(0) > self.max_frames:
            # Submuestreo de frames para ahorrar memoria
            idx = torch.randperm(pred.size(0), device=pred.device)[:self.max_frames]
            pred, target, masks = pred[idx], target[idx], masks[idx]

        disp_pred = self._disp(pred)          # con gradiente hacia Endo-STTN
        with torch.no_grad():
            disp_target = self._disp(target)  # objetivo fijo

        m = F.interpolate(masks.float(), size=disp_pred.shape[-2:], mode="nearest")
        weight = 1.0 + self.mask_weight * m   # más peso dentro de la máscara
        return scale_invariant_l1(disp_pred, disp_target, weight)
