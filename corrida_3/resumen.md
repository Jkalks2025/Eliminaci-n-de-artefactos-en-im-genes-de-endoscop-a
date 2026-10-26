# Avance 3 · Endo-STTN original vs reentrenado con C3VDv2

Videos de validación: c1_ascending_t3_v1  
Videos de prueba: c2_descending_t4_v2, c2_descending_t4_v3  
Reentrenado: `gen_00005.pth` (iteración 1250)  
Máscaras: meslouhi · frames: 1 de cada 1 · lote 4 · lr 0.0001 · GPU: NVIDIA L4 · entrenamiento iniciado: 2026-10-25T14:18

## Prueba: error del relleno en el hueco desplazado (pseudo ground truth)

PSNR en dB y SSIM (0-1): mayor es mejor. MAE en valores de color 0-255: menor es mejor.

| modelo      |    psnr |   ssim |    mae |
|:------------|--------:|-------:|-------:|
| original    | 29.5098 | 0.8259 | 6.6584 |
| reentrenado | 26.6682 | 0.8017 | 8.7091 |

Wilcoxon pareado por frame (H1: el reentrenado es mejor):

| metrica   |   frames |   mediana_dif (reentrenado - original) |   p_valor |
|:----------|---------:|---------------------------------------:|----------:|
| psnr      |      250 |                               -2.402   |         1 |
| ssim      |      250 |                               -0.02635 |         1 |
| mae       |      250 |                                1.481   |         1 |

## Validación por checkpoint

| modelo    |   iteracion |     mae |    psnr |   ssim |   frames |
|:----------|------------:|--------:|--------:|-------:|---------:|
| original  |           0 |  5.7166 | 30.9369 | 0.8559 |      242 |
| gen_00001 |         250 |  9.5467 | 26.2223 | 0.7997 |      242 |
| gen_00002 |         500 |  8.6972 | 27.2923 | 0.8218 |      242 |
| gen_00003 |         750 |  9.5427 | 26.5006 | 0.8056 |      242 |
| gen_00004 |        1000 |  9.1761 | 26.8605 | 0.8189 |      242 |
| gen_00005 |        1250 |  8.4631 | 27.588  | 0.8294 |      242 |
| gen_00006 |        1500 | 10.0298 | 25.9673 | 0.7929 |      242 |
| gen_00007 |        1750 |  9.296  | 27.1023 | 0.8212 |      242 |
| gen_00008 |        2000 |  9.0038 | 27.3773 | 0.827  |      242 |
