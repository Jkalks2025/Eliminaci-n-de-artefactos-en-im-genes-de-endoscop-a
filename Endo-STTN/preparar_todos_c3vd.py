# -*- coding: utf-8 -*-
# Solo PREPARA todos los zips de datasets/raw (frames + mascaras + train.json/test.json).
# No corre el modelo. Usa la funcion preparar() de preparar_c3vd.py.
# Uso (desde la carpeta de Endo-STTN):  python preparar_todos_c3vd.py
import os
import glob
import json
from preparar_c3vd import preparar, RAW, RAIZ, VIDEOS_EVAL

zips = sorted(glob.glob(os.path.join(RAW, "*.zip")))
print(f"Zips encontrados en {RAW}: {len(zips)}\n")

lista = {}
for k, origen in enumerate(zips, 1):
    nombre = os.path.splitext(os.path.basename(origen))[0]
    n = preparar(origen, nombre)
    if n == 0:
        print(f"[{k}/{len(zips)}] {nombre}: sin carpeta rgb/ -> saltado")
    else:
        print(f"[{k}/{len(zips)}] {nombre}: {n} frames listos")
        lista[nombre] = n

# test.json: todos los videos (test.py los procesa todos)
# train.json: todos menos VIDEOS_EVAL, que se reservan para evaluar
train = {v: n for v, n in lista.items() if v not in VIDEOS_EVAL} or lista
test = lista

for archivo, datos in (("train.json", train), ("test.json", test)):
    with open(os.path.join(RAIZ, archivo), "w") as f:
        json.dump(datos, f, indent=2)

print(f"\nListo: {len(lista)} videos preparados, {sum(lista.values())} frames en total.")
print(f"train.json: {len(train)} videos | test.json: {len(test)} videos  -> {RAIZ}")
print(f"Reservados para evaluar (fuera de train.json): {[v for v in VIDEOS_EVAL if v in lista]}")
