# Modelos finales

**Equipo 19** · Modelos end-to-end del [Avance 5](../README.md): Endo-STTN y AF-SfMLearner
entrenados juntos (modelo **E**), uno por cada máscara de reflejos.

| Modelo | Endo-STTN de partida | Máscara | Carpeta en Drive |
|---|---|---|---|
| **E · meslouhi** | Corrida 4 del Avance 4 (iteración 1750) | El Meslouhi (2011) | `MyDrive/Equipo19/end_to_end_mascaras/modelos/E_meslouhi_s0/` |
| **E · arnold** | Variante A1 del Avance 4 (iteración 1750) | Arnold (2010) | `MyDrive/Equipo19/end_to_end_mascaras/modelos/E_arnold_s0/` |

Cada carpeta es un **par**: las dos redes se entrenaron juntas y se usan juntas.

| Archivo | Qué es | Tamaño aprox. |
|---|---|---|
| `sttn.pth` | Endo-STTN ajustado (inpainting), mismo formato que `gen_00009.pth` | 64 MB |
| `encoder.pth`, `depth.pth` | AF-SfMLearner ajustado (estimación de profundidad) | 45 MB y 13 MB |
| `historial.json` | Configuración del entrenamiento y métricas de validación por época | — |

Los dos se entrenaron con la configuración del notebook
[`Equipo19_EndToEnd_Mascaras_colab.ipynb`](../notebooks/Equipo19_EndToEnd_Mascaras_colab.ipynb)
(semilla 0, 15 épocas). Sus corridas ejecutadas están en [`../notebooks/`](../notebooks/).

## Por qué los pesos no están en git

Cada modelo pesa unos 122 MB. En git, cada versión quedaría para siempre en el historial y se descargaría en cada
`git clone`, y el `.gitignore` del proyecto excluye los `*.pth`. Los pesos se guardan en Google Drive. Para
compartirlos con alguien sin acceso a ese Drive, se pueden subir a la
[carpeta compartida del Equipo 19](https://drive.google.com/drive/folders/1_L36tu2Q4mr1VXogK7a7M3bfcj9-bMU3) o
adjuntar a un *Release* de GitHub.

## Cómo usarlos en un video de C3VDv2

Para cada modelo: calcular la máscara con **su** método (El Meslouhi o Arnold), limpiar el video con su `sttn.pth`
(inferencia oficial de Endo-STTN, en orden de frames) y estimar la profundidad de los frames limpios con su
`encoder.pth` y `depth.pth`. Un video usado en entrenamiento o validación (o de su mismo segmento) no sirve para
evaluar.
