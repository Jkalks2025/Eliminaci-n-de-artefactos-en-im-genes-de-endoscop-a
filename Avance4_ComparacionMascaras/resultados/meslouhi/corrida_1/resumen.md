# Avance 3 · Endo-STTN original vs reentrenado con C3VDv2

Videos de validación: c1_ascending_t3_v1  
Videos de prueba: c2_descending_t4_v2, c2_descending_t4_v3  
Reentrenado: `gen_00001.pth` (iteración 1000)  
Máscaras: meslouhi · frames: 1 de cada 1 · lote 2 · lr 0.0001 · GPU: Tesla T4 · entrenamiento iniciado: 2026-10-21T03:56

## Prueba: error del relleno en el hueco desplazado (pseudo ground truth)

PSNR en dB y SSIM (0-1): mayor es mejor. MAE en valores de color 0-255: menor es mejor.

| modelo      |    psnr |   ssim |    mae |
|:------------|--------:|-------:|-------:|
| original    | 29.5098 | 0.8259 | 6.6584 |
| reentrenado | 27.1846 | 0.8083 | 8.6762 |

Wilcoxon pareado por frame (H1: el reentrenado es mejor):

| metrica   |   frames |   mediana_dif (reentrenado - original) |   p_valor |
|:----------|---------:|---------------------------------------:|----------:|
| psnr      |      250 |                               -1.168   |         1 |
| ssim      |      250 |                               -0.01376 |         1 |
| mae       |      250 |                                1.188   |         1 |

## Validación por checkpoint

| modelo    |   iteracion |     mae |    psnr |   ssim |   frames |
|:----------|------------:|--------:|--------:|-------:|---------:|
| original  |           0 |  5.7166 | 30.9369 | 0.8559 |      242 |
| gen_00001 |        1000 |  7.6581 | 28.527  | 0.8477 |      242 |
| gen_00002 |        2000 |  9.1509 | 26.8462 | 0.8182 |      242 |
| gen_00003 |        3000 |  9.4399 | 26.8355 | 0.8118 |      242 |
| gen_00004 |        4000 | 10.5084 | 26.1195 | 0.7953 |      242 |
| gen_00005 |        5000 | 10.8033 | 25.9462 | 0.7924 |      242 |
| gen_00006 |        6000 | 10.7188 | 26.0306 | 0.7914 |      242 |
| gen_00007 |        7000 | 11.5229 | 25.2356 | 0.7654 |      242 |
| gen_00008 |        8000 | 11.6047 | 25.2799 | 0.758  |      242 |
| gen_00009 |        9000 | 12.0367 | 25.1938 | 0.7514 |      242 |
