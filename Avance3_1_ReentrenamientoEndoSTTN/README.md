# Avance 3.1 · Reentrenamiento de Endo-STTN con C3VDv2

**Equipo 19** · Continúa el [Avance 2](../Avance2_DeepEstimator_EndoSTTN/README.md), en paralelo al
[Avance 3](../Avance3_EntrenamientoConjunto/README.md).

> **Estado:** implementado y probado de punta a punta con datos sintéticos (preparar, entrenar, continuar un
> entrenamiento interrumpido, evaluar y el notebook). Falta correrlo con C3VDv2 en Colab con GPU.

## 1. Idea

En el Avance 2, Endo-STTN se usa tal como se publicó (`gen_00009.pth`), entrenado con colonoscopías reales de
HyperKvasir. C3VDv2 son modelos de silicona, con otra textura, otro color y otros reflejos. Aquí Endo-STTN se
**reentrena con frames de C3VDv2** usando su mismo procedimiento de entrenamiento, para que rellene mejor los
reflejos de este dominio.

A diferencia del Avance 3, aquí Endo-STTN aprende **solo a rellenar**: no recibe ninguna señal del estimador de
profundidad. Así se puede medir por separado si un mejor relleno ayuda después a la profundidad (Avance 3.2).

**Hipótesis.** El Endo-STTN reentrenado rellena el tejido de C3VDv2 con menos error que el original, en videos que
no vio al entrenar.

## 2. Las máscaras

Para entrenar un modelo de inpainting normalmente se necesita la imagen "sin reflejo" como respuesta correcta, y no
existe. Endo-STTN lo resuelve con dos máscaras, y las dos se generan automáticamente:

| Máscara | Cómo se obtiene | Para qué |
|---|---|---|
| **Reflejos** | Píxeles casi blancos (brillo ≥ 230), dilatados 3 px, igual que en el Avance 2 (`codigo/c3vd.py`) | Indica dónde están los reflejos reales |
| **Hueco desplazado** | La máscara de reflejos movida 50 px a la derecha (a 288×288), sin los píxeles que caen sobre otro reflejo | Cae sobre tejido sano, cuyo color sí se conoce: es la respuesta correcta (*pseudo ground truth*) |

```
frame ──► máscara de reflejos ──► desplazar 50 px ──► hueco sobre tejido sano
                                                         │
                       Endo-STTN rellena el hueco ◄──────┘   L1 contra el color real del hueco
```

Al entrenar, el propio código de Endo-STTN (`Endo-STTN/core/dataset.py`) desplaza la máscara; al evaluar lo hace
`evaluar_endosttn.py` de la misma forma. Es el método con el que los autores de Endo-STTN entrenaron y evaluaron su
modelo (Daher et al., 2023).

## 3. Configuración

Se usa el entrenamiento original (`Endo-STTN/core/trainer.py`), **sin modificarlo**. Solo cambia la configuración:

| Parámetro | Original (HyperKvasir) | Aquí (C3VDv2) |
|---|---|---|
| Datos | 2 GPUs, videos de HyperKvasir | 1 GPU (Colab), videos de entrenamiento de C3VDv2 |
| Punto de partida | STTN preentrenado | Generador `gen_00009.pth`; discriminador desde cero (el modelo publicado no lo trae) |
| Pérdidas | L1 en el hueco + L1 fuera del hueco + adversarial (hinge, 0.01) | Igual |
| Optimizador | Adam, lr 1e-4, betas (0, 0.99) | Igual |
| Clips | 5 frames a 288×288 | Igual |
| Lote | 8 | 4 (`--lote`) |
| Iteraciones | 500 000 | 10 000 por defecto (`--iteraciones`); un checkpoint cada 1 000 |
| Dilatación de la máscara (`Dil`) | 8 | 0: las máscaras ya vienen dilatadas como en el Avance 2 |
| División | — | Por video, la misma del Avance 2 |

Los frames se preparan como en el Avance 2 (320×256, misma máscara) y luego se llevan a 288×288, igual que en la
inferencia. Por defecto se usan **todos los frames** de cada video (`--paso 1`), no 1 de cada 5 como en el Avance 2,
porque Endo-STTN necesita más imágenes y aprovecha la continuidad entre frames. Cada video de entrenamiento se parte
en trozos de 30 frames, porque Endo-STTN toma un clip por "video" en cada paso y con solo 3 o 4 videos una época
tendría un único lote.

**Elección del modelo.** Se evalúan el original y todos los checkpoints en el video de validación. El de menor MAE
en el hueco es el "reentrenado" (`gen_mejor.pth`), y se compara contra el original en el video de prueba.

## 4. Cómo correrlo

**En Google Colab (recomendado):** [`notebooks/Equipo19_Avance3_1_colab.ipynb`](notebooks/Equipo19_Avance3_1_colab.ipynb).
Usa `MyDrive/Equipo19/raw/` (los mismos ZIPs del Avance 2) y toma la división de `c3vd_preparado.zip`. Todo lo
que produce queda en `MyDrive/Equipo19/avance3_1/`, y si Colab se desconecta, al volver a correrlo continúa desde el
último checkpoint.

**En una computadora con GPU:**

```bash
pip install tensorboardX                                              # lo usa el entrenamiento de Endo-STTN
python Avance3_1_ReentrenamientoEndoSTTN/preparar_endosttn.py         # datos/endosttn_c3vd/
python Avance3_1_ReentrenamientoEndoSTTN/entrenar_endosttn.py         # modelos/endosttn_c3vd/gen_XXXXX.pth
python Avance3_1_ReentrenamientoEndoSTTN/evaluar_endosttn.py          # resultados/ y gen_mejor.pth
```

Opciones útiles: `--iteraciones 20000`, `--lote 2` (si falta memoria), `--paso 2` (menos frames),
`--division mi_division.json` (otra división de videos).

## 5. Experimentos

| | Experimento | Resultado |
|---|---|---|
| ✅ | Prueba de punta a punta con datos sintéticos (scripts y notebook) | Corre; reanuda tras una interrupción |
| ⬜ | Reentrenamiento con C3VDv2 (10 000 iteraciones) y evaluación en el video de prueba | *pendiente* |
| ⬜ | Inspección visual del relleno original vs reentrenado | *pendiente* |
| ⬜ | Avance 3.2: limpiar los frames con `gen_mejor.pth` y ajustar el estimador de profundidad (comparar con D) | *pendiente* |

## 6. Limitaciones

- **Pocos videos.** Con 3 o 4 videos de entrenamiento, el modelo puede aprender texturas específicas de esos
  segmentos. La validación por checkpoint ayuda a detectar el sobreajuste.
- **El hueco desplazado no es un reflejo.** El error se mide sobre tejido sano con la forma de un reflejo; debajo de
  un reflejo real el tejido puede ser distinto (más cerca de la cámara, de frente a la luz).
- **Discriminador nuevo.** Al inicio el discriminador no sabe nada, y la pérdida adversarial puede ser inestable.
  Tiene peso 0.01, como en el original, así que domina la L1.
- **Espacio en Drive.** Cada checkpoint del generador ocupa unos 66 MB; además se conservan el último discriminador y
  optimizador (unos 350 MB).

## Archivos

| Archivo | Contenido |
|---|---|
| [`preparar_endosttn.py`](preparar_endosttn.py) | Paso 1: frames y máscaras de C3VDv2 en el formato de Endo-STTN |
| [`entrenar_endosttn.py`](entrenar_endosttn.py) | Paso 2: entrenamiento original de Endo-STTN con la configuración de C3VDv2 |
| [`evaluar_endosttn.py`](evaluar_endosttn.py) | Paso 3: elección del checkpoint y comparación original vs reentrenado |
| [`endosttn_c3vd.py`](endosttn_c3vd.py) | Rutas, hueco desplazado y métricas compartidas |
| [`notebooks/Equipo19_Avance3_1_colab.ipynb`](notebooks/Equipo19_Avance3_1_colab.ipynb) | Flujo completo en Colab con figuras |
| [`resultados/`](resultados/) | Resultados de cada corrida |

## Referencia

R. Daher, F. Vasconcelos y D. Stoyanov, *A Temporal Learning Approach to Inpainting Endoscopic Specularities and
Its Effect on Image Correspondence*, Medical Image Analysis, 2023.

---

*Este avance se elaboró con apoyo de Claude (Anthropic) para código y documentación.
Todos los integrantes revisaron y validaron el contenido, y el equipo es responsable de él.*
