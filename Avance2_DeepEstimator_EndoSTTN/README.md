# Avance 2 · DeepEstimator + Endo-STTN

Las instrucciones para instalar y correr están en el [README principal](../README.md).
Aquí se explica el método y dónde quedan los resultados.

## Método

Durante el entrenamiento de Endo-STTN, el loader tapa una zona **sana** del frame (la máscara de
reflejos desplazada 50 px, sin los reflejos reales), así que se conoce la respuesta correcta.
El frame reconstruido (`comp_img`) y el frame limpio pasan por AF-SfMLearner:

| Paso | Qué pasa | Código |
|---|---|---|
| 1 | Endo-STTN rellena la zona tapada → `comp_img` | `Endo-STTN/core/trainer.py` |
| 2 | AF-SfMLearner estima la disparidad de `comp_img` (con gradiente) y del frame limpio (sin gradiente) | `Endo-STTN/core/depth_loss.py` |
| 3 | `L_depth`: L1 invariante a escala entre ambas, con 5× peso dentro de la máscara | `scale_invariant_l1` |
| 4 | `gen_loss += depth_weight · L_depth` (tras `warmup_iters`) | `trainer.py` |
| 5 | `backward()`: el gradiente atraviesa AF-SfMLearner (pesos congelados) y llega a Endo-STTN | — |
| 6 | `optimG.step()` actualiza solo a Endo-STTN | — |

**Congelado no es lo mismo que cortado.** `requires_grad_(False)` fija los pesos del estimador, pero el
gradiente sigue pasando por sus activaciones. Fuera de la máscara `comp_img` es el frame original,
así que ahí el gradiente es cero: la pérdida solo corrige lo que Endo-STTN inventó.

## Parámetros (`depth` y `losses.depth_weight` en los configs)

| Parámetro | Valor | Efecto |
|---|---|---|
| `depth_weight` | 0.1 (0 en el config sin profundidad) | Peso de `L_depth` |
| `model` | `afsfmlearner` | También acepta `monodepth2` y `monoiit` |
| `warmup_iters` | 200 | Iteraciones iniciales sin `L_depth` |
| `mask_weight` | 4.0 | Peso extra dentro de la máscara |
| `max_frames` | 2 | Frames por paso que pasan por el estimador (memoria) |

## Notebooks

| Notebook | Uso |
|---|---|
| `notebooks/Equipo19_EndoSTTN_Profundidad_local.ipynb` | En cualquier computadora con el repo clonado |
| `notebooks/Equipo19_EndoSTTN_Profundidad_colab.ipynb` | En Google Colab (clona el repo y prepara todo) |

Secciones: datos, estimador de profundidad, visualización del gradiente, prueba de arranque,
entrenamiento, curvas de pérdida y evaluación (preentrenado vs. con vs. sin profundidad).

## Resultados

Ver [`resultados/`](resultados/README.md). Prueba de arranque en la laptop del equipo: 5 iteraciones,
≈2.1 s/iteración, `L_depth` = 0 durante el warm-up (esperado).

## Limitaciones

Un solo video de entrenamiento y pocas iteraciones; el estimador está entrenado en SCARED (no en C3VD).
Siguientes pasos: más videos de C3VD, usar la profundidad real de C3VD como objetivo y probar MonoIIT.
