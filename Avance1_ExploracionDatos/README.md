# Avance 1 · Exploración de datos (C3VDv2)

**Equipo 19** · Proyecto Integrador · 4 de octubre de 2026

Análisis exploratorio de los videos de colonoscopía de **C3VDv2** que después usa el
[Avance 2](../Avance2_DeepEstimator_EndoSTTN/README.md). El documento completo, con el código, las figuras y las
salidas ya ejecutadas en Google Colab, es el notebook:

**[`Exploracion_de_Datos_Equipo19.ipynb`](Exploracion_de_Datos_Equipo19.ipynb)**

## Contenido del notebook

| Sección | Tema |
|---|---|
| 1 | Descripción de C3VDv2 (192 videos, 169 371 frames) y estrategia de muestreo por video |
| 2 | Obtención de los datos: descarga manual (el repositorio de JHU rechaza descargas automáticas) y acceso desde Colab vía Google Drive |
| 3 | Plan de exploración y por qué la división entrenamiento / validación / prueba se hace por segmento, no por frame |
| 4 | Análisis: inventario e integridad, galería, formación de la imagen, color, brillo y calidad, frecuencias, PCA, efecto de los residuos (v2 vs v3), mapas de referencia y redundancia temporal |
| 5 | Referencias |

## Muestra analizada

Cinco videos registrados del fantoma c2 (860 frames, 8.2 GB), todos con profundidad real por frame:

| Video | Versión | Frames |
|---|---|---:|
| `c2_descending_t4_v1` | v1 | 252 |
| `c2_descending_t4_v2` | v2 (limpio) | 128 |
| `c2_descending_t4_v3` | v3 (con residuos, misma trayectoria que v2) | 128 |
| `c2_descending_t3_v1` | v1 | 198 |
| `c2_rectum_t1_v1` | v1 | 154 |

## Resultados principales

- **Integridad:** 0 archivos dañados. Cada video tiene un mapa de flujo óptico menos que frames (el primero no tiene
  anterior) y no se encontraron mapas de oclusión con el nombre documentado.
- **Formato:** frames RGB de 1350×1080; profundidad en TIFF de 16 bits, 0–100 mm. La profundidad media es de unos
  28–30 mm y alrededor del 3–6 % de los píxeles queda fuera de rango (0 o 100 mm).
- **Reflejos especulares:** la mediana es 1.2 % del campo de visión en los videos v1 y 0.3–0.5 % en v2/v3, con picos
  en vistas de frente a la pared. Son pocos píxeles, pero concentrados y saturados, lo que motiva el Avance 2.
- **Residuos (v2 vs v3):** las poses de v2 y v3 coinciden exactamente; los residuos cubren en promedio 2.2 % del campo
  de visión (hasta 10.7 %) y bajan la nitidez y la saturación.
- **Redundancia temporal:** el SSIM entre frames consecutivos es 0.79–0.89; deja de ser mayor a 0.8 a partir de un
  desfase de 4 frames en v1 y 8 en v2. Por eso el Avance 2 toma **1 de cada 5 frames**.

La sección 4.15 del notebook (Hallazgos) contiene las preguntas que guían la interpretación de cada análisis.

## Cómo correrlo

Abre el notebook en Google Colab. La celda 2.2 monta Google Drive y busca los `.zip` de C3VDv2 en `Equipo19/raw`
(la misma carpeta que usa el Avance 2). Fuera de Colab, pon las rutas de tus videos en `RUTAS_DATOS` (sección 4.2).
