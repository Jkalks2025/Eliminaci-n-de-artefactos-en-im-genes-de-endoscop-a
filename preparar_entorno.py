"""Descarga lo que no va en el repositorio y revisa que todo esté en su lugar.

    python preparar_entorno.py

1. Clona AF-SfMLearner (estimador de profundidad) en externos/, fijado a la versión probada.
2. Descarga sus pesos End-to-end (entrenados en SCARED) en externos/pesos/.
3. Descarga el modelo preentrenado de Endo-STTN (gen_00009.pth) en Endo-STTN/release_model/.
4. Revisa que existan los videos de C3VDv2 en datos/raw/ y si ya se prepararon.

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
# Copias en la carpeta compartida del Equipo 19 (primero) y enlaces originales (respaldo)
CARPETA_EQUIPO = "https://drive.google.com/drive/folders/1_L36tu2Q4mr1VXogK7a7M3bfcj9-bMU3"
AFSFM_PESOS_GDRIVE = ["1ZogZ5nQsfLhFAQGG_UE9A5R_osypLZf1",   # Equipo 19
                      "1kf7LjQ6a2ACKr6nX5Uyee3of3bXn1xWB"]   # Model zoo de AF-SfMLearner, fila End-to-end
STTN_PRE = RAIZ / "Endo-STTN" / "release_model" / "pretrained_model" / "gen_00009.pth"
STTN_PRE_GDRIVE = ["1Evx0YSlb7YK0WWok862gddT62gDgv6ux",      # Equipo 19
                   "14sdaDejsxgRuzHBSuqH2xEpbxuqyWI-R"]      # copia del Equipo 52
RAW = RAIZ / "datos" / "raw"
DATOS = RAIZ / "datos" / "c3vd"

pendientes = []


def gdown(ids, destino):
    """Descarga un archivo público de Google Drive, probando cada id en orden. Devuelve True si funcionó."""
    try:
        import gdown as g
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)
        try:
            import gdown as g
        except ImportError:
            return False
    destino.parent.mkdir(parents=True, exist_ok=True)
    for file_id in ids:
        try:
            if g.download(id=file_id, output=str(destino), quiet=False) is not None and destino.exists():
                return True
        except Exception as e:
            print(f"   No se pudo descargar {file_id}: {e}")
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
        pendientes.append(f"Pesos de AF-SfMLearner: descargar Model_trained_end_to_end.zip de {CARPETA_EQUIPO} "
                          f"y guardarlo como {zip_path}; luego volver a correr este script")

print("3. Preentrenado de Endo-STTN")
if STTN_PRE.exists():
    print(f"   ya existe: {STTN_PRE}")
elif gdown(STTN_PRE_GDRIVE, STTN_PRE):
    print(f"   listo en {STTN_PRE}")
else:
    pendientes.append(f"gen_00009.pth: descargarlo de {CARPETA_EQUIPO} y guardarlo en {STTN_PRE}")

print("4. Datos C3VDv2")
zips = sorted(RAW.glob("*.zip"))
if not zips:
    pendientes.append(f"Datos: descargar los videos de C3VDv2 (ZIPs con rgb/ y depth/) en {RAW}")
elif (DATOS / "division.json").exists():
    print(f"   {len(zips)} ZIPs en datos/raw; ya preparados en {DATOS}")
else:
    print(f"   {len(zips)} ZIPs en datos/raw; falta prepararlos:  python codigo/preparar_datos.py")

print()
if pendientes:
    print("PENDIENTE (hacer a mano):")
    for p in pendientes:
        print("  -", p)
else:
    print("Todo listo. Siguiente:  python codigo/probar_formato.py")
