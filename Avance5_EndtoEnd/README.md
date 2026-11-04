# Avance 5 – End to End

Carpeta para el Avance 5 del proyecto (pipeline end-to-end).

## Notebook

[`notebooks/Equipo19_EndToEnd_Mascaras_colab.ipynb`](notebooks/Equipo19_EndToEnd_Mascaras_colab.ipynb): entrenamiento
end-to-end (Endo-STTN + AF-SfMLearner) con los dos Endo-STTN reentrenados del Avance 4, donde se identificaron 
El Meslouhi (corrida 4) y Arnold (A1) como los best performers.
Se planea ejecutar el colab 2 veces, uno para cada mascara
- Por el momento solo se probo con la mascara Arnold, esta pendiente correrse con Meslouhi.
- Se esta revisando como se puede implementar SAM.

## Modelos que se comparan

| Modelo | Inpainting (Endo-STTN) | Estimador de profundidad | ¿End-to-end? |
|---|---|---|---|
| **A** | ninguno | AF-SfMLearner original (SCARED), sin ajustar | No |
| **C** | ninguno | ajustado con frames originales | No |
| **D** | reentrenado, **congelado**: limpia una vez y no aprende más | ajustado con esos frames limpios | No: las dos redes se entrenan por separado |
| **E** | reentrenado, **sigue aprendiendo** | ajustado **al mismo tiempo** | **Sí** |

D y E existen para cada máscara (El Meslouhi y Arnold). La comparación principal es **D contra E**: en D el error de
profundidad solo corrige al estimador; en E también regresa hasta Endo-STTN y le enseña a rellenar de la forma que más
ayuda a estimar la profundidad. 
