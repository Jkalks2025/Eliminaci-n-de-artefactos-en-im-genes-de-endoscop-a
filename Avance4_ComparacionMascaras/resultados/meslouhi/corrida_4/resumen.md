# Avance 3 · Endo-STTN original vs reentrenado con C3VDv2

Videos de validación: c2_sigmoid_t3_v2  
Videos de prueba: c2_descending_t4_v2, c2_descending_t4_v3  
Reentrenado: `gen_00007.pth` (iteración 1750)  
Máscaras: meslouhi · frames: 1 de cada 1 · lote 4 · lr 1e-05 · GPU: NVIDIA L4 · entrenamiento iniciado: 2026-10-25T21:28

## Prueba: error del relleno en el hueco desplazado (pseudo ground truth)

PSNR en dB y SSIM (0-1): mayor es mejor. MAE en valores de color 0-255: menor es mejor.

| modelo      |    psnr |   ssim |    mae |
|:------------|--------:|-------:|-------:|
| original    | 29.5098 | 0.8259 | 6.6584 |
| reentrenado | 32.5234 | 0.8786 | 4.2971 |

Wilcoxon pareado por frame (H1: el reentrenado es mejor):

| metrica   |   frames |   mediana_dif (reentrenado - original) |   p_valor |
|:----------|---------:|---------------------------------------:|----------:|
| psnr      |      250 |                                3.06    | 9.165e-42 |
| ssim      |      250 |                                0.05286 | 4.627e-43 |
| mae       |      250 |                               -2.458   | 4.683e-43 |

## Validación por checkpoint

| modelo    |   iteracion |    mae |    psnr |   ssim |   frames |
|:----------|------------:|-------:|--------:|-------:|---------:|
| original  |           0 | 3.0193 | 36.5197 | 0.9373 |       53 |
| gen_00001 |         250 | 1.9665 | 43.1832 | 0.9748 |       53 |
| gen_00002 |         500 | 1.9144 | 43.3864 | 0.976  |       53 |
| gen_00003 |         750 | 1.968  | 43.1945 | 0.9755 |       53 |
| gen_00004 |        1000 | 1.9331 | 43.3377 | 0.9759 |       53 |
| gen_00005 |        1250 | 1.9168 | 43.39   | 0.9764 |       53 |
| gen_00006 |        1500 | 1.9473 | 43.3342 | 0.9762 |       53 |
| gen_00007 |        1750 | 1.9016 | 43.5005 | 0.9768 |       53 |
| gen_00008 |        2000 | 1.9589 | 43.31   | 0.9764 |       53 |
