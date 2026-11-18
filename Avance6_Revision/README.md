# Avance 6 · Revisión: pruebas separadas del pipeline end-to-end

**Equipo 19** · Atiende las notas de la revisión con el profesor sobre el código final del
[Avance 5](../Avance5_EndtoEnd/README.md).

> **Estado:** ejecutado en Colab con los datos y modelos reales. El notebook con sus salidas está en
> [`notebooks/Equipo19_Avance6_Revision_colab_ejecutado.ipynb`](notebooks/Equipo19_Avance6_Revision_colab_ejecutado.ipynb).
> Antes se probó de punta a punta con datos sintéticos, incluido un caso con fuga de datos a propósito para comprobar
> que la auditoría la detecta.

## Notas de la revisión y cómo se atienden

| Nota | Qué pidió el profesor | Prueba en el notebook |
|---|---|---|
| 1 | Medir la profundidad contra la referencia como una prueba separada | **Sección 6.** Cada modelo (A, C, D, E y controles) se reconstruye a partir de sus pesos guardados y se evalúa en los videos de prueba, sin reutilizar resultados del entrenamiento |
| 2 | Entrenar solo Endo-STTN con ambas máscaras, sin profundidad, y ver qué tal rellena | **Sección 5.** Mismo entrenamiento que E (clips, lote, épocas, tasa, semillas y pérdida de hueco) pero sin la pérdida de profundidad. Inferencia en prueba y PSNR, SSIM y MAE en el pseudo-hueco, contra el Endo-STTN original, el del Avance 4 y el de E |
| 3 | Ver cómo se comporta el estimador solo, sin nada más | **Sección 4.** AF-SfMLearner con los pesos SCARED sobre los frames originales (modelo A): mapas de profundidad, referencia y error |
| 4 | Comprobar que la división de Endo-STTN y la del estimador no comparten frames | **Sección 3.** Auditoría en tres niveles: video, segmento y contenido del frame (MD5 exacto y huella perceptual). Si hay un cruce, el notebook se detiene |
| 5 | Obtener AbsRel, Sq Rel, RMSE, RMSE log y δ con intervalos de confianza del 95 % para el modelo E | **Sección 6.** *Bootstrap* por bloques de frames consecutivos, promediando las semillas, más intervalos de las diferencias pareadas (E − D, E − C, E Meslouhi − E Arnold, controles). Exporta la tabla en LaTeX para el artículo |

## Decisiones

- **Notebook aparte.** No modifica el notebook del Avance 5 ni sus resultados: lee sus datos y modelos de Drive
  (`end_to_end_mascaras/`) y guarda todo lo nuevo en `MyDrive/Equipo19/avance6_revision/`.
- **El mismo código.** Las celdas de herramientas (lectura de C3VDv2, máscaras, AF-SfMLearner, Endo-STTN) y de
  preparación de datos son copias exactas de las del Avance 5.
- **Endo-STTN solo, comparable con E.** Se entrena igual que E y solo se quita la pérdida de profundidad. Así se separa
  qué aporta seguir entrenando el relleno y qué aporta la señal de profundidad. Responde también el control
  $\lambda_h = 0$ que quedó como trabajo futuro.
- **Intervalos por bloques.** Los frames vecinos están correlacionados; remuestrear frames sueltos daría intervalos
  demasiado estrechos. Se remuestrean bloques de 5 frames consecutivos dentro de cada video (10 000 repeticiones).
- **Auditoría por contenido.** Los ZIPs pueden tener sufijos distintos (`-007`), así que además de los nombres se
  compara una huella MD5 de cada frame preparado a 320×256: el mismo frame da la misma huella aunque el archivo se
  llame distinto.

## Cómo correrlo

En Colab Pro (A100 o L4, RAM alta):
[`notebooks/Equipo19_Avance6_Revision_colab.ipynb`](notebooks/Equipo19_Avance6_Revision_colab.ipynb). Antes de
*Ejecutar todas*, revisa en la primera celda:

- `CARPETAS_PRINCIPALES`: si las semillas 1 y 2 se guardaron en otra carpeta que `end_to_end_mascaras/modelos`,
  agrégala a la lista.
- `CONTROLES`: las carpetas de los controles con ancla y con Endo-STTN congelado.
- `PESOS_ENDO`: los Endo-STTN del Avance 4.

La auditoría lee la división de cada Endo-STTN del `config.json` guardado junto a sus pesos. Si no existe, usa la
división documentada del Avance 4 y lo indica en la salida.

## Salidas (`MyDrive/Equipo19/avance6_revision/resultados/`)

| Archivo | Contenido |
|---|---|
| `auditoria_fuga.md`, `fuga_frames_a_revisar.csv`, `fuga_casi_identicos.png` | Nota 4 |
| `estimador_solo.png` | Nota 3 |
| `relleno_por_frame.csv`, `relleno_resumen.csv`, `relleno_<máscara>.png` | Nota 2 |
| `profundidad_por_frame.csv`, `profundidad_ic95.csv`, `diferencias_ic95.csv`, `profundidad_ic95.png` | Notas 1 y 5 |
| `tabla_profundidad_ic95.tex` | Tabla con IC 95 % lista para el artículo |

## Prueba adicional: punto de partida del estimador

Todos nuestros modelos parten de los pesos oficiales de **AF-SfMLearner entrenado en SCARED**. Para saber si ese punto
de partida influye, se revisaron los pesos de un trabajo anterior: eran métodos publicados entrenados en SCARED. Solo
**Monodepth2** es compatible con nuestro código (mismo codificador ResNet-18 y decodificador); sus pesos de
AF-SfMLearner eran idénticos a los nuestros (MD5), y MonoViT, EndoDAC y Endo-SfMLearner usan otras arquitecturas.

El notebook [`notebooks/Equipo19_PesosEquipoAnterior_colab.ipynb`](notebooks/Equipo19_PesosEquipoAnterior_colab.ipynb)
entrena C, D y E exactamente igual que en el Avance 5, pero partiendo de Monodepth2, y los evalúa en la misma prueba.

AbsRel en prueba (semilla 0; menor es mejor). Cada celda: imagen completa / reflejos.

| Modelo | AF-SfMLearner (original) | Monodepth2 (trabajo anterior) | Mejor en imagen completa | Mejor en reflejos |
|---|---|---|---|---|
| A · sin ajustar | 0.396 / 0.814 | 0.460 / 0.986 | Original | Original |
| C · sin limpiar | 0.123 / 0.074 | 0.116 / 0.144 | Monodepth2 | Original |
| D · por etapas | 0.122 / 0.081 | **0.102** / 0.107 | Monodepth2 | Original |
| E · conjunto | **0.112** / 0.078 | 0.119 / 0.134 | Empate | Original |

Con los pesos originales, E mejora a D (−0.010, IC 95 % [−0.011, −0.009]). Con Monodepth2 se invierte: D mejora a E
(+0.017, IC 95 % [0.012, 0.019]). Dentro de los reflejos, los pesos originales son mejores en todos los modelos. Es
decir, el punto de partida del estimador influye tanto como el método, y la ventaja del entrenamiento conjunto
depende de partir de AF-SfMLearner. Son resultados preliminares con una sola semilla; para confirmarlos hay que
correr `SEMILLAS = [0, 1, 2]`.

Las salidas se guardan en `MyDrive/Equipo19/pesos_equipo_anterior_pruebas/` y no modifican nada del Avance 5.

## Limitación

Los intervalos describen la incertidumbre **dentro del segmento de prueba** (`c2_descending_t4`, dos videos). El
*bootstrap* corrige la correlación entre frames vecinos, pero no la falta de segmentos distintos: no dice cómo
generaliza el modelo a otras partes del colon.

---

*Este avance se elaboró con apoyo de Claude (Anthropic) para código y documentación.
Todos los integrantes revisaron y validaron el contenido, y el equipo es responsable de él.*
