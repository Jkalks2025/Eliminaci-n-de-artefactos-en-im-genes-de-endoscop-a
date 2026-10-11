# Eliminación de artefactos en imágenes de endoscopía · Equipo 19

Proyecto para mejorar la estimación de profundidad en video endoscópico eliminando los reflejos especulares.
**Endo-STTN** (inpainting temporal) limpia los frames y con ellos se ajusta un **estimador de profundidad
monocular** (AF-SfMLearner), usando la profundidad real de **C3VDv2** (colonoscopía 3D).

**Integrantes (Equipo 19):**
- Javier Emmanuel García Escobedo
- Mario Raúl García Ramírez
- Juan Miguel Molina Reyes

## Avances

| Avance | Tema | Estado | Documento |
|---|---|---|---|
| 1 | *(por completar)* | En preparación | [`Avance1/`](Avance1/) |
| 2 | Endo-STTN limpia los frames → ajuste fino del estimador de profundidad | Implementado y probado con datos sintéticos; resultados con C3VDv2 pendientes | [`Avance2_DeepEstimator_EndoSTTN/README.md`](Avance2_DeepEstimator_EndoSTTN/README.md) |

Cada avance tiene su propio documento con objetivo, marco teórico, metodología, resultados y referencias.

## Cómo correrlo

Requiere Python 3.8+. Para limpiar y entrenar conviene una GPU NVIDIA con CUDA; también se puede usar Google Colab:
[`Equipo19_Avance2_colab.ipynb`](Avance2_DeepEstimator_EndoSTTN/notebooks/Equipo19_Avance2_colab.ipynb).

**1. Clonar e instalar**
```bash
git clone https://github.com/Jkalks2025/Eliminaci-n-de-artefactos-en-im-genes-de-endoscop-a.git
cd Eliminaci-n-de-artefactos-en-im-genes-de-endoscop-a
python -m venv venv            # activar: venv\Scripts\activate (Windows) o source venv/bin/activate
pip install torch torchvision  # o el comando de pytorch.org para tu versión de CUDA
pip install -r requirements.txt
```

**2. Descargar lo que no va en git** (AF-SfMLearner, sus pesos y el preentrenado de Endo-STTN)
```bash
python preparar_entorno.py
```
Al final indica si falta algo y cómo conseguirlo.

**3. Prueba rápida** (sin datos ni pesos: comprueba que los 4 pasos funcionan, con datos sintéticos)
```bash
python codigo/prueba_rapida.py
```

**4. Datos.** Copia los ZIPs de C3VDv2 (cada uno con `rgb/` y `depth/`) en `datos/raw/` y revisa el formato:
```bash
python codigo/probar_formato.py
```
Debe mostrar la profundidad en milímetros razonables (unos 5–100 mm). Si no, ajusta `PROF_MAX_MM` en `codigo/c3vd.py`.

**5. Correr los cuatro pasos**
```bash
python codigo/preparar_datos.py                          # frames + profundidad real + reflejos, división por video
python codigo/limpiar_endosttn.py                        # Endo-STTN congelado borra los reflejos
python codigo/ajustar_profundidad.py --entrada rgb         # modelo C (frames originales)
python codigo/ajustar_profundidad.py --entrada rgb_limpio  # modelo D (frames limpios)
python codigo/evaluar_profundidad.py                     # A, B, C, D contra la profundidad real
```
Los resultados quedan en `Avance2_DeepEstimator_EndoSTTN/resultados/`. El notebook
[`Equipo19_Avance2_local.ipynb`](Avance2_DeepEstimator_EndoSTTN/notebooks/Equipo19_Avance2_local.ipynb) hace lo mismo con figuras.

## Qué hay en el repositorio

```
README.md                         este archivo (portada)
requirements.txt                  dependencias de Python
preparar_entorno.py               descarga AF-SfMLearner, pesos y preentrenado de Endo-STTN
codigo/                           código del Equipo 19
├── c3vd.py                       lectura de C3VDv2 (rgb + profundidad real), máscaras y división por video
├── preparar_datos.py             paso 1
├── limpiar_endosttn.py           paso 2: Endo-STTN congelado
├── modelo_profundidad.py         AF-SfMLearner, pérdida scale-invariant y métricas
├── ajustar_profundidad.py        paso 3: ajuste fino (modelos C y D)
├── evaluar_profundidad.py        paso 4: evaluación A-D y Wilcoxon
├── probar_formato.py             revisa el formato de un ZIP de C3VDv2
└── prueba_rapida.py              prueba de punta a punta con datos sintéticos
Avance1/                          (por completar)
Avance2_DeepEstimator_EndoSTTN/   documento del Avance 2, figuras, notebooks y resultados
Endo-STTN/                        Endo-STTN original, sin modificaciones (se usa congelado)
datos/raw/                        aquí van los ZIPs de C3VDv2 (no se suben a git)
externos/, modelos/, datos/c3vd/  (no están en git) AF-SfMLearner, pesos, estimadores ajustados y datos preparados
```

## Créditos y licencias

- **Endo-STTN** — R. Daher, F. Vasconcelos, D. Stoyanov, *A Temporal Learning Approach to Inpainting
  Endoscopic Specularities and Its Effect on Image Correspondence*, Medical Image Analysis, 2023.
  [github.com/endomapper/Endo-STTN](https://github.com/endomapper/Endo-STTN) (commit `9419a6c`), licencia MIT
  (`Endo-STTN/LICENCE`), incluido sin modificaciones. Basado en STTN (Zeng et al., ECCV 2020).
- **AF-SfMLearner** — S. Shao et al., Medical Image Analysis, 2022.
  [github.com/ShuweiShao/AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner) (commit `53d01a4`), licencia MIT;
  partes de su red de profundidad provienen de Monodepth2 (Niantic, licencia de uso no comercial).
  No se incluye en el repositorio: lo descarga `preparar_entorno.py`.
- **C3VDv2** — Colonoscopy 3D Video Dataset (Johns Hopkins University). No se incluye en el repositorio.
- Evaluación previa de estimadores de profundidad: [proyecto_integrador_52](https://github.com/jmtoral/proyecto_integrador_52) (Equipo 52).

## Uso de inteligencia artificial

Este proyecto se desarrolló con apoyo de **Claude** (Anthropic), un asistente de inteligencia
artificial, en las siguientes tareas:
- escritura y depuración de código (preparación de C3VDv2, limpieza con Endo-STTN, ajuste fino y
  evaluación del estimador de profundidad, notebooks);
- redacción y organización de la documentación, incluido el marco teórico;
- elaboración de diagramas.

Todos los integrantes del Equipo 19 revisaron el contenido generado: verificamos que el código
funciona en nuestro equipo, contrastamos las explicaciones y referencias con las fuentes originales y
ajustamos lo necesario. Las decisiones de diseño, los experimentos y las conclusiones son del equipo,
que asume la responsabilidad total del contenido de este repositorio.
