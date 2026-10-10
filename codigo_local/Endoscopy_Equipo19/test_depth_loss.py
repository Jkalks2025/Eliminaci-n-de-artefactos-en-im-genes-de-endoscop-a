"""Prueba rápida: el gradiente de DepthLoss llega al generador y MonoIIT queda congelado.

Usa redes pequeñas de reemplazo (no necesita pesos). Ejecutar desde esta carpeta:
    python test_depth_loss.py
"""
import torch
import torch.nn as nn

from core.depth_loss import DepthLoss


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x):
        return [torch.relu(self.bn(self.conv(x)))]


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 1, 3, padding=1)

    def forward(self, feats):
        return {("disp", 0): torch.sigmoid(self.conv(feats[-1]))}


def main():
    torch.manual_seed(0)
    b, t, h, w = 2, 3, 32, 32
    gen = nn.Conv2d(3, 3, 3, padding=1)   # hace de netG
    loss_fn = DepthLoss(FakeEncoder(), FakeDecoder(), in_size=(24, 40), max_frames=4)
    loss_fn.train()                       # debe seguir en eval
    assert not loss_fn.training and not loss_fn.encoder.bn.training

    bn_mean = loss_fn.encoder.bn.running_mean.clone()
    depth_params = [p.clone() for p in loss_fn.parameters()]

    frames = torch.rand(b * t, 3, h, w) * 2 - 1
    masks = (torch.rand(b * t, 1, h, w) > 0.8).float()
    pred_img = torch.tanh(gen(frames * (1 - masks)))
    comp_img = frames * (1. - masks) + masks * pred_img

    loss = loss_fn(comp_img, frames, masks)
    loss.backward()

    assert loss.item() > 0
    assert gen.weight.grad is not None and gen.weight.grad.abs().sum() > 0, "no llega gradiente a netG"
    assert all(p.grad is None for p in loss_fn.parameters()), "MonoIIT recibió gradiente"
    assert all(torch.equal(a, p) for a, p in zip(depth_params, loss_fn.parameters()))
    assert torch.equal(bn_mean, loss_fn.encoder.bn.running_mean), "BatchNorm de MonoIIT cambió"
    print(f"OK  loss={loss.item():.4f}  |grad netG|={gen.weight.grad.abs().sum():.4f}")


if __name__ == "__main__":
    main()
