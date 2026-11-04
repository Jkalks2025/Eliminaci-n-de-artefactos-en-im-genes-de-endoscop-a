# Prueba de SAM para las máscaras de reflejos

**Equipo 19** · Prueba aparte, a partir del [Avance 3](../Avance3_ReEntrenar_Endo/README.md) (comparación de
máscaras) y del [Avance 4](../Avance4_ComparacionMascaras/README.md).

> **Estado:** notebook listo y probado de punta a punta solo con datos sintéticos y SAM con pesos aleatorios (sin
> sentido para los números). Falta correrlo en Colab con los videos de C3VDv2 y los pesos reales de SAM.

## Pregunta

¿**SAM** (*Segment Anything*, Kirillov et al., 2023) detecta los reflejos especulares mejor que **El Meslouhi
(2011)** y **Arnold (2010)**? Es solo el primer paso: medir las máscaras. Si SAM no gana claramente, no vale la pena
reentrenar Endo-STTN con ellas ni llegar al entrenamiento end-to-end.

## Métodos comparados

SAM segmenta lo que se le señala, pero no sabe qué es un reflejo. Por eso se prueban tres formas de usarlo:

| Método | Cómo funciona |
|---|---|
| `meslouhi`, `arnold` | Métodos actuales, sin SAM (línea base) |
| `sam_meslouhi`, `sam_arnold` | **SAM como refinador:** cada reflejo detectado por el método actual se le da a SAM como caja + punto en su píxel más brillante, y SAM ajusta el contorno. Si SAM devuelve algo mucho más grande que el reflejo (tomó tejido), se conserva la detección original |
| `sam_auto` | **SAM automático:** segmenta todo el frame y se quedan los segmentos pequeños, muy brillantes y casi blancos |

Todo se mide a 288×288 (la resolución de Endo-STTN) contra la **misma referencia física** del Avance 3: el render
`diffuse/` de C3VDv2 no tiene reflejos, así que un píxel mucho más brillante de lo que predice es un reflejo. Se
reportan precisión, sensibilidad, F1 e IoU (estrictos y con tolerancia de 2 px), separados en videos limpios (v1, v2)
y con residuos (v3), además del tiempo por frame.

## Cómo correrlo

En Google Colab (Colab Pro, GPU A100 o L4):
[`Equipo19_SAM_prueba_colab.ipynb`](Equipo19_SAM_prueba_colab.ipynb). Es autocontenido: no depende del resto del
repositorio. Lee los ZIPs de `MyDrive/Equipo19/raw/` (solo los que traen `diffuse/`), descarga los pesos de SAM la
primera vez y guarda todo en `MyDrive/Equipo19/SAM_prueba/`:

| Archivo | Contenido |
|---|---|
| `comparacion_resumen.md` | Tablas por grupo de videos, costo por frame y la decisión |
| `comparacion_resumen.csv`, `comparacion_por_frame.csv` | Métricas resumidas y por frame |
| `comparacion_metricas.png` | F1 por grupo y precisión contra sensibilidad |
| `comparacion_visual.png` | Aciertos, falsos positivos y falsos negativos de cada método en un frame por video |

## Criterio de decisión

Se compara el F1 con tolerancia en los videos **limpios** del mejor método con SAM contra el del mejor método actual.
Solo una ventaja **mayor a 0.02** justifica el siguiente paso (reentrenar Endo-STTN con las máscaras de SAM). También
cuenta el costo: SAM tarda mucho más por frame, y el reentrenamiento usa miles de frames.

## Limitaciones

- **La referencia no es perfecta.** En el Avance 3 marcó cerca del 0.6 % de los píxeles y todos los métodos tuvieron F1
  bajo (0.15–0.20 con tolerancia). Lo que importa es la comparación entre métodos con la misma referencia.
- **Los reflejos son difíciles para SAM:** muchos, pequeños y con bordes difusos. SAM rinde mejor con objetos medianos
  y bien delimitados.
- **SAM automático** depende de una rejilla de puntos: los reflejos más pequeños pueden quedar entre los puntos.

---

*Esta prueba se elaboró con apoyo de Claude (Anthropic) para código y documentación.
Todos los integrantes revisaron y validaron el contenido, y el equipo es responsable de él.*
