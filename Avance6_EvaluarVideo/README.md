# Avance 6 · Aplicar los modelos end-to-end a un video de C3VDv2

**Equipo 19** · Continúa el [Avance 5](../Avance5_EndtoEnd/README.md).

> **Estado:** notebook listo y probado de punta a punta solo con datos sintéticos y modelos de prueba. Falta
> correrlo en Colab con los modelos finales y videos de C3VDv2.

## Idea

En el Avance 5 se entrenaron los modelos **E** (Endo-STTN y AF-SfMLearner entrenados juntos), uno por máscara: E ·
meslouhi y E · arnold. Este avance **no entrena nada**: aplica esos modelos a **cualquier video** de C3VDv2, frame por
frame y en orden, y permite verlo como video.

```
video ─► máscara de reflejos ─► Endo-STTN ajustado (inpainting) ─► frame limpio ─► estimador ajustado ─► profundidad
         (la de cada modelo)       sttn.pth                                         encoder.pth + depth.pth
```

## Qué produce

| Salida | Contenido |
|---|---|
| Visor frame por frame | Control deslizante en Colab: original, frame limpio, profundidad y error de cada modelo |
| `<video>_end_to_end.mp4` | Todo el video con el mismo diseño, para presentaciones |
| `resumen.md`, `metricas_por_frame.csv` | Solo si el video trae `depth/`: AbsRel y δ<1.25 en la imagen y en los reflejos, y parpadeo |
| `absrel_por_frame.png` | AbsRel de cada modelo a lo largo del video |

Se comparan **E · meslouhi**, **E · arnold** y, como referencia, **A** (el estimador original sobre frames sin
limpiar). Todo se guarda en `MyDrive/Equipo19/avance6/<video>/`.

## Decisiones

- **Todos los frames** (`PASO = 1`) para que el video sea fluido. El entrenamiento usó 1 de cada 5 frames; el modelo
  funciona igual con todos.
- **Cada modelo usa su propia máscara** (El Meslouhi o Arnold), la misma con la que se entrenó su Endo-STTN.
- **Colores con un rango fijo para todo el video** y **una sola escala de profundidad por video y modelo**. Si cada
  frame tuviera su propia escala, el video parpadearía aunque la profundidad no cambiara.
- **Métricas con *median scaling* por frame**, igual que la evaluación del Avance 5, y en una región común a todos los
  modelos (la unión de las dos máscaras).
- **Parpadeo:** cuánto cambia la profundidad relativa entre frames consecutivos, comparado con el de la profundidad
  real. El estimador procesa cada frame por separado, así que puede temblar más que la realidad.
- **Aviso de fuga de datos:** el notebook lee de cada modelo qué videos usó para entrenar y validar. Si el video
  elegido es uno de ellos o comparte segmento (otra versión `v1/v2/v3`), lo avisa: sirve para verlo, pero sus métricas
  no son una evaluación válida.

## Cómo correrlo

En Google Colab (Colab Pro, A100 o L4, RAM alta):
[`notebooks/Equipo19_Avance6_Evaluar_Video_colab.ipynb`](notebooks/Equipo19_Avance6_Evaluar_Video_colab.ipynb).
En la primera celda se elige `VIDEO` (nombre del ZIP en `MyDrive/Equipo19/raw/`). Los modelos se leen de
`MyDrive/Equipo19/end_to_end_mascaras/modelos/` (ver [Modelos Finales](../Avance5_EndtoEnd/Modelos%20Finales/README.md)).

---

*Este avance se elaboró con apoyo de Claude (Anthropic) para código y documentación.
Todos los integrantes revisaron y validaron el contenido, y el equipo es responsable de él.*
