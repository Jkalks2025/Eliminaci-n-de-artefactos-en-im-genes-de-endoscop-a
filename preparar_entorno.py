"""Descarga lo que no va en el repositorio y revisa que todo esté en su lugar.

    python preparar_entorno.py

1. Clona AF-SfMLearner (estimador de profundidad) en externos/, fijado a la versión probada.
2. Descarga sus pesos End-to-end (entrenados en SCARED) en externos/pesos/.
3. Descarga el modelo preentrenado de Endo-STTN (gen_00009.pth) en Endo-STTN/release_model/.
4. Revisa que existan los datos de C3VD convertidos.

Lo que no se pueda descargar solo se indica al final, con el enlace para hacerlo a mano.
Se puede correr las veces que haga falta: lo que ya existe no se vuelve a descargar.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
EXTERNOS = RAIZ / "externos"
AFSFM_DIR = EXTERNOS / "AF-SfMLearner"
AFSFM_REPO = "https://github.com/ShuweiShao/AF-SfMLearner.git"
AFSFM_COMMIT = "53d01a4"                                   # versión probada
AFSFM_PESOS = EXTERNOS / "pesos" / "Model_trained_end_to_end"
AFSFM_PESOS_GDRIVE = "1kf7LjQ6a2ACKr6nX5Uyee3of3bXn1xWB"   # Model zoo de AF-SfMLearner, fila End-to-end
STTN_PRE = RAIZ / "Endo-STTN" / "release_model" / "pretrained_model" / "gen_00009.pth"
STTN_PRE_GDRIVE = "14sdaDejsxgRuzHBSuqH2xEpbxuqyWI-R"      # copia de gen_00009.pth en Google Drive
DATOS = RAIZ / "Endo-STTN" / "datasets" / "C3VD"

pendientes = []


def gdown(file_id, destino):
    """Descarga un archivo público de Google Drive. Devuelve True si funcionó."""
    try:
        import gdown as g
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)
        try:
            import gdown as g
        except ImportError:
            return False
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        return g.download(id=file_id, output=str(destino), quiet=False) is not None and destino.exists()
    except Exception as e:
        print(f"   No se pudo descargar: {e}")
        return False


print("1. AF-SfMLearner")
if (AFSFM_DIR / "networks" / "resnet_encoder.py").exists():
    print(f"   ya existe: {AFSFM_DIR}")
else:
    EXTERNOS.mkdir(exist_ok=True)
    ok = subprocess.run(["git", "clone", "-q", AFSFM_REPO, str(AFSFM_DIR)]).returncode == 0
    ok = ok and subprocess.run(["git", "-C", str(AFSFM_DIR), "checkout", "-q", AFSFM_COMMIT]).returncode == 0
    print(f"   {'clonado en ' + str(AFSFM_DIR) + ' @ ' + AFSFM_COMMIT if ok else 'FALLÓ el clone'}")
    if not ok:
        pendientes.append(f"Clonar {AFSFM_REPO} en {AFSFM_DIR}")

print("2. Pesos de AF-SfMLearner")
if (AFSFM_PESOS / "encoder.pth").exists() and (AFSFM_PESOS / "depth.pth").exists():
    print(f"   ya existen: {AFSFM_PESOS}")
else:
    zip_path = EXTERNOS / "Model_trained_end_to_end.zip"
    if zip_path.exists() or gdown(AFSFM_PESOS_GDRIVE, zip_path):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(AFSFM_PESOS.parent)
        # si el zip trae otra carpeta, mover encoder.pth/depth.pth a la ruta esperada
        if not (AFSFM_PESOS / "encoder.pth").exists():
            enc = next(AFSFM_PESOS.parent.rglob("encoder.pth"), None)
            if enc is not None:
                AFSFM_PESOS.mkdir(parents=True, exist_ok=True)
                for f in ("encoder.pth", "depth.pth"):
                    shutil.move(str(enc.parent / f), str(AFSFM_PESOS / f))
    if (AFSFM_PESOS / "encoder.pth").exists():
        print(f"   listos en {AFSFM_PESOS}")
    else:
        pendientes.append("Pesos de AF-SfMLearner: README de github.com/ShuweiShao/AF-SfMLearner -> Model zoo -> "
                          f"End-to-end -> guardar el zip como {zip_path} y volver a correr este script")

print("3. Preentrenado de Endo-STTN")
if STTN_PRE.exists():
    print(f"   ya existe: {STTN_PRE}")
elif gdown(STTN_PRE_GDRIVE, STTN_PRE):
    print(f"   listo en {STTN_PRE}")
else:
    pendientes.append("gen_00009.pth: enlace 'pretrained model' del README de github.com/endomapper/Endo-STTN -> "
                      f"descomprimir para que quede en {STTN_PRE}")

print("4. Datos C3VD convertidos")
if (DATOS / "train.json").exists() and any((DATOS / "JPEGImages").glob("*.zip")):
    print(f"   listos en {DATOS}")
else:
    pendientes.append("Datos: descargar videos de C3VDv2 (doi:10.7281/T1/JC64MK) en Endo-STTN/datasets/raw/ "
                      "y correr  cd Endo-STTN ; python preparar_todos_c3vd.py")

print()
if pendientes:
    print("PENDIENTE (hacer a mano):")
    for p in pendientes:
        print("  -", p)
else:
    print("Todo listo. Siguiente:  cd Endo-STTN ; python check_depth_weights.py")
