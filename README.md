# Eliminación de artefactos en imágenes de endoscopía · Equipo 19

Eliminación de reflejos especulares en video endoscópico con **Endo-STTN**, guiando el
entrenamiento con un **estimador de profundidad** para que el relleno respete la geometría 3D
del tejido. Datos: **C3VDv2** (colonoscopía 3D).

## Empieza aquí

1. Lee [`Avance2_DeepEstimator_EndoSTTN/README.md`](Avance2_DeepEstimator_EndoSTTN/README.md):
   explica el método y los pasos en orden.
2. Instala el código en tu clon de Endo-STTN con `Avance2_DeepEstimator_EndoSTTN/instalar.ps1`.
3. Sigue el notebook `Avance2_DeepEstimator_EndoSTTN/notebooks/Equipo19_EndoSTTN_Profundidad_local.ipynb` (o la versión `_colab`).

## Estado

| | Tarea |
|---|---|
| ✅ | Pérdida de profundidad integrada en Endo-STTN (AF-SfMLearner congelado) |
| ✅ | C3VDv2 convertido al formato de Endo-STTN (19 videos) |
| ✅ | Pesos de AF-SfMLearner cargan en la GPU de la laptop |
| ✅ | Prueba de arranque: 5 iteraciones, ≈2.1 s/iteración |
| ⬜ | Entrenamiento corto **con** profundidad (2000 iteraciones, 1 video) |
| ⬜ | Entrenamiento corto **sin** profundidad (comparación) |
| ⬜ | Evaluación en un video no visto y resultados en `resultados/` |
| ⬜ | Correr la versión de Colab |

## Estructura

```
Avance2_DeepEstimator_EndoSTTN/
├── README.md        pasos detallados
├── instalar.ps1     copia el código a D:\Git\Endo-STTN
├── endo_sttn/       lo que va encima de Endo-STTN (core/, configs/)
├── scripts/         preparación de C3VD y pruebas
├── notebooks/       explicación y evaluación (local y Colab)
└── resultados/      métricas y figuras
```

## Repositorios externos

- [Endo-STTN](https://github.com/endomapper/Endo-STTN): modelo de inpainting.
- [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner): estimador de profundidad (pesos entrenados en SCARED).
- [proyecto_integrador_52](https://github.com/jmtoral/proyecto_integrador_52): evaluación de profundidad del Equipo 52 (MonoIIT).
