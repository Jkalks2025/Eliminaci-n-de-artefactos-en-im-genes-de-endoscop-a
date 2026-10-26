# Avance 3 · Endo-STTN original vs reentrenado con C3VDv2

Videos de validación: c2_sigmoid_t3_v2  
Videos de prueba: c2_descending_t4_v2, c2_descending_t4_v3  
Reentrenado: `gen_00008.pth` (iteración 2000)  
Máscaras: meslouhi · frames: 1 de cada 1 · lote 4 · lr 1e-05 · GPU: NVIDIA L4 · entrenamiento iniciado: 2026-10-25T23:25

## Prueba: error del relleno en el hueco desplazado (pseudo ground truth)

PSNR en dB y SSIM (0-1): mayor es mejor. MAE en valores de color 0-255: menor es mejor.

| modelo      |    psnr |   ssim |    mae |
|:------------|--------:|-------:|-------:|
| original    | 29.5098 | 0.8259 | 6.6584 |
| reentrenado | 32.4253 | 0.8763 | 4.3735 |

Wilcoxon pareado por frame (H1: el reentrenado es mejor):

| metrica   |   frames |   mediana_dif (reentrenado - original) |   p_valor |
|:----------|---------:|---------------------------------------:|----------:|
| psnr      |      250 |                                2.582   | 2.186e-40 |
| ssim      |      250 |                                0.05135 | 4.855e-43 |
| mae       |      250 |                               -2.344   | 5.476e-43 |

## Validación por checkpoint

| modelo    |   iteracion |    mae |    psnr |   ssim |   frames |
|:----------|------------:|-------:|--------:|-------:|---------:|
| original  |           0 | 3.0193 | 36.5197 | 0.9373 |       53 |
| gen_00001 |         250 | 2.0209 | 42.986  | 0.9744 |       53 |
| gen_00002 |         500 | 1.9757 | 43.1341 | 0.9752 |       53 |
| gen_00003 |         750 | 1.9474 | 43.2988 | 0.976  |       53 |
| gen_00004 |        1000 | 1.9187 | 43.3587 | 0.9762 |       53 |
| gen_00005 |        1250 | 1.9588 | 43.2473 | 0.9759 |       53 |
| gen_00006 |        1500 | 1.974  | 43.1971 | 0.9755 |       53 |
| gen_00007 |        1750 | 1.9308 | 43.3614 | 0.9759 |       53 |
| gen_00008 |        2000 | 1.8889 | 43.5462 | 0.977  |       53 |
