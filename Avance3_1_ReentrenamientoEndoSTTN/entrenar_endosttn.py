"""Avance 3.1, paso 2: reentrena Endo-STTN con C3VDv2, partiendo del modelo publicado (gen_00009.pth).

    python Avance3_1_ReentrenamientoEndoSTTN/entrenar_endosttn.py
    python Avance3_1_ReentrenamientoEndoSTTN/entrenar_endosttn.py --iteraciones 20000 --lote 8

Usa el entrenamiento original de Endo-STTN sin modificarlo (Endo-STTN/core/trainer.py y
core/dataset.py): generador + discriminador con pérdida L1 en el hueco desplazado (pseudo ground
truth), L1 en el resto de la imagen y pérdida adversarial (hinge). Lo único que cambia es la
configuración que aquí se escribe (datos de C3VDv2, una GPU, número de iteraciones):

- El generador arranca con gen_00009.pth. El discriminador arranca desde cero, porque el modelo
  publicado no lo incluye (Endo-STTN carga solo el generador si encuentra un sttn.pth).
- Dil = 0: las máscaras de preparar_endosttn.py ya vienen dilatadas como en el Avance 2.

Guarda cada --guardar-cada iteraciones gen_XXXXX.pth en modelos/endosttn_c3vd/. Si el entrenamiento se
corta (p. ej. Colab se desconecta), al volver a correrlo continúa desde el último checkpoint. De los
dis_/opt_ (discriminador y optimizadores) solo se conserva el último, que es el que hace falta para continuar.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import endosttn_c3vd as ec


def config(args, device):
    """Igual que Endo-STTN/configs/EndoSTTN_dataset.json salvo datos, GPU, lote e iteraciones."""
    return {
        "seed": args.semilla,
        "save_dir": str(ec.DIR_ENTRENAMIENTO),
        "data_loader": {"name": ec.NOMBRE, "data_root": str(ec.DIR_DATASET.parent), "w": ec.LADO, "h": ec.LADO,
                        "sample_length": 5, "Dil": 0, "frame_limit": 100000},
        "losses": {"hole_weight": 1, "valid_weight": 1, "adversarial_weight": 0.01, "GAN_LOSS": "hinge"},
        "trainer": {"type": "Adam", "beta1": 0.0, "beta2": 0.99, "lr": args.lr, "d2glr": 1,
                    "batch_size": args.lote, "num_workers": args.workers, "verbosity": 2, "log_step": 100,
                    "save_freq": args.guardar_cada, "valid_freq": args.guardar_cada,
                    "iterations": args.iteraciones, "niter": 10 ** 9, "niter_steady": 10 ** 9},
        # lo que train.py de Endo-STTN agrega antes de crear el Trainer (aquí sin procesos distribuidos)
        "model": "sttn", "distributed": False, "world_size": 1, "global_rank": 0, "local_rank": 0,
        "device": device, "initialmodel": str(ec.DIR_ENTRENAMIENTO / "inicio"), "chosen_epoch": "0",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default=str(ec.PESOS_ORIGINALES), help="generador inicial ({'netG': ...})")
    ap.add_argument("--iteraciones", type=int, default=10000)
    ap.add_argument("--guardar-cada", type=int, default=1000)
    ap.add_argument("--lote", type=int, default=4, help="clips de 5 frames por lote (Endo-STTN usó 8 en 2 GPUs)")
    ap.add_argument("--lr", type=float, default=1e-4, help="como en el entrenamiento original")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--semilla", type=int, default=2020)
    args = ap.parse_args()

    if not (ec.DIR_DATASET / "train.json").exists():
        raise SystemExit("Falta el dataset: corre primero preparar_endosttn.py")
    if not Path(args.pesos).exists():
        raise SystemExit(f"No encuentro {args.pesos} (python preparar_entorno.py lo descarga)")

    # Barra de progreso de Endo-STTN: una línea cada 30 s en vez de una por iteración
    os.environ.setdefault("TQDM_MININTERVAL", "30")
    import random
    import numpy as np
    import torch
    sys.path.insert(0, str(ec.ENDO_STTN))
    from core.trainer import Trainer

    class Entrenador(Trainer):
        def save(self, it):
            super().save(it)
            ultimo = f"{int(it):05d}.pth"
            for p in ec.DIR_ENTRENAMIENTO.glob("*_*.pth"):
                if p.name.startswith(("dis_", "opt_")) and not p.name.endswith(ultimo):
                    p.unlink()

    random.seed(args.semilla); np.random.seed(args.semilla); torch.manual_seed(args.semilla)
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    cfg = config(args, device)

    inicio = ec.DIR_ENTRENAMIENTO / "inicio"
    inicio.mkdir(parents=True, exist_ok=True)
    if not (inicio / "sttn.pth").exists():
        (inicio / "sttn.pth").symlink_to(Path(args.pesos).resolve())
    json.dump({**cfg, "device": str(device), "pesos_iniciales": args.pesos},
              open(ec.DIR_ENTRENAMIENTO / "config.json", "w"), indent=2)

    trozos = json.load(open(ec.DIR_DATASET / "train.json"))
    print(f"Endo-STTN: {len(trozos)} trozos ({sum(trozos.values())} frames) | lote {args.lote} | "
          f"{args.iteraciones} iteraciones | {device}", flush=True)
    if str(device) == "cpu":
        print("AVISO: sin GPU el entrenamiento es muy lento; úsalo solo para probar.", flush=True)

    entrenador = Entrenador(cfg)
    if entrenador.iteration:
        print(f"Continuando desde la iteración {entrenador.iteration}", flush=True)
    entrenador.train()
    print(f"Checkpoints en {ec.DIR_ENTRENAMIENTO} (siguiente paso: evaluar_endosttn.py)")


if __name__ == "__main__":
    main()
