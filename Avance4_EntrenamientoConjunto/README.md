# Avance 4 · Endo-STTN y el estimador de profundidad entrenados juntos (modelo E)

**Equipo 19** · Exploración a partir del [Avance 2](../Avance2_DeepEstimator_EndoSTTN/README.md)

> **Estado:** implementado y probado de punta a punta con datos sintéticos (`python codigo/prueba_rapida.py`).
> Falta correrlo con C3VDv2 en GPU.

## 1. Idea

En el Avance 2, Endo-STTN está **congelado**: limpia los frames una sola vez y AF-SfMLearner se ajusta con esos
frames limpios (modelo **D**). Endo-STTN se entrenó para que el relleno *se vea* como tejido, no para que le sirva al
estimador de profundidad, y además se entrenó con otro dominio (HyperKvasir, no modelos de silicona).

En este avance las dos redes forman **un solo modelo** y se entrenan **juntas**: la pérdida de profundidad se propaga
hacia atrás también por Endo-STTN, así que el relleno de los reflejos se adapta a lo que necesita el estimador.

```
clip de T frames ──► máscara de reflejos ──► Endo-STTN (entrenable) ──► relleno solo dentro de la máscara
                                                                               │
                                     profundidad real ◄── SI-log ◄── AF-SfMLearner (entrenable)
```

**Hipótesis.** E tiene menor error de profundidad que D, sobre todo dentro de los reflejos, sin que el relleno deje
de parecerse a tejido.

## 2. El riesgo y cómo se controla

Si solo se usa la pérdida de profundidad, Endo-STTN puede aprender a "pintar" cualquier patrón que le facilite el
trabajo al estimador, aunque ya no parezca tejido. Para evitarlo, la pérdida total incluye términos de imagen:

```math
\mathcal{L} = \mathcal{L}_{SI} + \lambda_{h}\,\mathcal{L}_{hueco} + \lambda_{a}\,\mathcal{L}_{ancla}
```

| Término | Qué hace | Por defecto |
|---|---|---|
| $`\mathcal{L}_{SI}`$ | Scale-invariant log de profundidad, igual que en C y D | 1 |
| $`\mathcal{L}_{hueco}`$ | Como en el entrenamiento de Endo-STTN: la máscara de reflejos se desplaza 56 px sobre tejido sano (*pseudo ground truth*) y Endo-STTN debe reconstruir esos píxeles, que sí se conocen. L1 dentro de ese hueco | $`\lambda_h = 1`$ |
| $`\mathcal{L}_{ancla}`$ | L1 entre el relleno y el del Endo-STTN original (`rgb_limpio.npy`), dentro de los reflejos. Limita cuánto se aleja del modelo preentrenado | $`\lambda_a = 0`$ (opcional) |

No se usa la pérdida adversarial de Endo-STTN porque el preentrenado publicado solo trae el generador
(`gen_00009.pth`), no el discriminador.

Durante la validación se registra el **cambio del relleno**: la diferencia media (0–255) dentro de los reflejos entre
el relleno de E y el del Endo-STTN original. Si crece mucho mientras el AbsRel baja, es señal de que Endo-STTN está
"haciendo trampa" y conviene subir $`\lambda_h`$ o activar $`\lambda_a`$.

## 3. Configuración

Para que la comparación **D vs E** aísle el efecto de entrenar Endo-STTN, todo lo demás es igual que en C y D:

| Parámetro | Valor |
|---|---|
| Punto de partida | Endo-STTN `gen_00009.pth` + AF-SfMLearner con pesos SCARED |
| Datos y división | Los mismos del Avance 2 (`datos/c3vd/`, `division.json`) |
| Entrada de Endo-STTN | Clips de 5 frames (`sample_length` de Endo-STTN): la mitad consecutivos y la otra mitad al azar, ordenados |
| Lote / épocas | 2 clips (10 frames) / 15; se guarda la época con menor AbsRel de validación |
| Optimizador | Adam. Estimador: lr 1e-4 (decoder) y 1e-5 (encoder). Endo-STTN: lr 1e-5 con betas (0, 0.99) como en su entrenamiento original |
| Capas de Endo-STTN | `--sttn-capas todo` (por defecto) o `transformer` (encoder y decoder de Endo-STTN congelados) |
| Aumentos | Volteo horizontal del clip completo; brillo ±10 % y desplazamiento ±0.05 después de la limpieza (igual que D) |

La validación y la evaluación usan la inferencia oficial de Endo-STTN (vecinos + referencias), igual que en el
Avance 2, pero con los pesos ajustados.

## 4. Cómo correrlo

**En Google Colab (recomendado):** [`notebooks/Equipo19_Avance4_colab.ipynb`](notebooks/Equipo19_Avance4_colab.ipynb).
Usa los mismos videos del Avance 2 desde `MyDrive/Equipo19/` (`c3vd_preparado.zip` o `raw/`) y reutiliza los
modelos C y D que guardó el notebook del Avance 2 en `MyDrive/Equipo19/modelos/`. Si faltan, los prepara o entrena
ahí mismo. Después entrena E, evalúa A–E y compara visualmente el relleno de D contra el de E. Guarda los frames
limpios en `c3vd_limpio.zip`, los modelos en `modelos/` y los resultados en `resultados_avance4/` (en Drive).

**En una computadora:** requiere haber corrido los pasos 1 y 2 del Avance 2 (`preparar_datos.py` y `limpiar_endosttn.py`), porque se usan
`rgb_limpio.npy` (ancla y diagnóstico) y la misma división de videos.

```bash
python Avance4_EntrenamientoConjunto/ajustar_conjunto.py                    # modelo E
python codigo/evaluar_profundidad.py                                        # A, B, C, D y E
```

Salida en `modelos/conjunto_s<semilla>/`: `encoder.pth`, `depth.pth`, `sttn.pth` (mismo formato que
`gen_00009.pth`) e `historial.json`. Además escribe `rgb_conjunto_s<semilla>.npy` en los videos de validación y
prueba; `evaluar_profundidad.py` lo detecta y añade el modelo **E** y las pruebas de Wilcoxon **D vs E**.

Variantes útiles:

```bash
# solo el transformer de Endo-STTN (menos memoria, cambia menos el relleno)
python Avance4_EntrenamientoConjunto/ajustar_conjunto.py --sttn-capas transformer
# más regularización si el relleno se aleja demasiado del original
python Avance4_EntrenamientoConjunto/ajustar_conjunto.py --peso-hueco 2 --peso-ancla 0.5
# si no alcanza la memoria de la GPU
python Avance4_EntrenamientoConjunto/ajustar_conjunto.py --lote 1
```

## 5. Experimentos

| | Experimento | Resultado |
|---|---|---|
| ✅ | Prueba de punta a punta con datos sintéticos (`codigo/prueba_rapida.py`) | Corren C, D y E, y la evaluación incluye E |
| ✅ | El gradiente de la profundidad llega a Endo-STTN (encoder, transformer y decoder) | Verificado |
| ✅ | Notebook de Colab: celdas probadas con datos sintéticos (sin la de montar Drive) | Corren y generan las figuras |
| ⬜ | E con C3VDv2, semilla 0, comparado contra D de la Prueba 1 | *pendiente* |
| ⬜ | Variantes: `transformer` y $`\lambda_a > 0`$; 3 semillas | *pendiente* |
| ⬜ | Inspección visual del relleno de E contra el de D | *pendiente* |

## 6. Limitaciones

- **Memoria.** Endo-STTN guarda las activaciones de todo el clip a 288×288; requiere bastante más GPU que D.
- **El relleno deja de ser "neutral".** Si E gana, parte de la mejora puede venir de que el relleno codifica pistas de
  profundidad y no de que sea un mejor inpainting. Por eso se reporta el cambio del relleno y se debe inspeccionar.
- **Diferencias de preprocesamiento.** En el entrenamiento, el redimensionado se hace en torch (`area`/`bilinear`) y
  en la inferencia con OpenCV. Las diferencias son pequeñas, pero existen.
- **Sin discriminador.** Sin la pérdida adversarial, el relleno puede volverse más borroso que el del Endo-STTN original.

## Archivos

| Archivo | Contenido |
|---|---|
| [`ajustar_conjunto.py`](ajustar_conjunto.py) | Entrenamiento conjunto Endo-STTN + AF-SfMLearner (modelo E) |
| [`notebooks/Equipo19_Avance4_colab.ipynb`](notebooks/Equipo19_Avance4_colab.ipynb) | Flujo completo en Colab con figuras (curvas, relleno D vs E, profundidad A–E) |
| [`../codigo/evaluar_profundidad.py`](../codigo/evaluar_profundidad.py) | Evaluación de A–E (incluye D vs E) |
| [`../codigo/limpiar_endosttn.py`](../codigo/limpiar_endosttn.py) | Inferencia de Endo-STTN (`limpiar_video`, `cargar`), usada también por E |

---

*Este avance se elaboró con apoyo de Claude (Anthropic) para código y documentación.
Todos los integrantes revisaron y validaron el contenido, y el equipo es responsable de él.*
