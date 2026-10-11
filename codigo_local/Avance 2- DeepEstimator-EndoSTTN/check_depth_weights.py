"""Comprueba que el modelo de profundidad del config carga con tus pesos reales.

Ejecutar desde la carpeta de Endo-STTN (la que tiene core/ y configs/):
    python check_depth_weights.py configs/EndoSTTN_depth_local.json
"""
import sys
import json

import torch

from core.depth_loss import DepthLoss


def main(config_path):
    cfg = json.load(open(config_path))
    h, w = cfg["data_loader"]["h"], cfg["data_loader"]["w"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    loss_fn = DepthLoss.from_config(cfg["depth"], device)
    print(f"Modelo '{cfg['depth'].get('model', 'monoiit')}' cargado; entrada {loss_fn.in_size}")

    frames = torch.rand(2, 3, h, w, device=device) * 2 - 1
    pred = (frames + 0.05 * torch.randn_like(frames)).requires_grad_()
    masks = (torch.rand(2, 1, h, w, device=device) > 0.9).float()
    loss = loss_fn(pred, frames, masks)
    loss.backward()
    print(f"OK  loss={loss.item():.4f}  |grad entrada|={pred.grad.abs().sum().item():.4f}")
    if device == "cuda":
        print(f"Memoria GPU máx.: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/EndoSTTN_depth_local.json")
