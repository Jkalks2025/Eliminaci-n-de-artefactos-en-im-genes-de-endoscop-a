# Avance 3 · Endo-STTN original vs reentrenado con C3VDv2

Videos de validación: c2_sigmoid_t3_v2  
Videos de prueba: c2_descending_t4_v2, c2_descending_t4_v3  
Reentrenado: `gen_00003.pth` (iteración 750)  
Máscaras: meslouhi · frames: 1 de cada 1 · lote 4 · lr 0.0001 · GPU: NVIDIA L4 · entrenamiento iniciado: 2026-10-25T06:48

## Prueba: error del relleno en el hueco desplazado (pseudo ground truth)

PSNR en dB y SSIM (0-1): mayor es mejor. MAE en valores de color 0-255: menor es mejor.

| modelo      |    psnr |   ssim |    mae |
|:------------|--------:|-------:|-------:|
| original    | 29.5098 | 0.8259 | 6.6584 |
| reentrenado | 31.6667 | 0.8719 | 4.7474 |

Wilcoxon pareado por frame (H1: el reentrenado es mejor):

| metrica   |   frames |   mediana_dif (reentrenado - original) |   p_valor |
|:----------|---------:|---------------------------------------:|----------:|
| psnr      |      250 |                                2.495   | 3.351e-27 |
| ssim      |      250 |                                0.04704 | 2.371e-40 |
| mae       |      250 |                               -1.973   | 2.337e-35 |

## Validación por checkpoint

| modelo    |   iteracion |    mae |    psnr |   ssim |   frames |
|:----------|------------:|-------:|--------:|-------:|---------:|
| original  |           0 | 3.0193 | 36.5197 | 0.9373 |       53 |
| gen_00001 |         250 | 2.8294 | 40.8397 | 0.9644 |       53 |
| gen_00002 |         500 | 2.3869 | 41.441  | 0.9717 |       53 |
| gen_00003 |         750 | 2.0947 | 42.8286 | 0.9746 |       53 |
| gen_00004 |        1000 | 2.6909 | 41.1513 | 0.9663 |       53 |
| gen_00005 |        1250 | 2.1001 | 42.9417 | 0.9742 |       53 |
| gen_00006 |        1500 | 2.4488 | 41.88   | 0.9707 |       53 |
| gen_00007 |        1750 | 2.1474 | 42.5877 | 0.975  |       53 |
| gen_00008 |        2000 | 2.0973 | 42.988  | 0.9747 |       53 |
| gen_00009 |        2250 | 2.5963 | 41.3507 | 0.966  |       53 |
| gen_00010 |        2500 | 2.1977 | 42.5359 | 0.9728 |       53 |
| gen_00011 |        2750 | 2.3593 | 42.1369 | 0.9695 |       53 |
| gen_00012 |        3000 | 2.842  | 40.671  | 0.9636 |       53 |
| gen_00013 |        3250 | 2.2245 | 42.5616 | 0.9727 |       53 |
| gen_00014 |        3500 | 2.3275 | 42.2085 | 0.9713 |       53 |
| gen_00015 |        3750 | 2.537  | 41.558  | 0.9667 |       53 |
| gen_00016 |        4000 | 2.2356 | 42.4973 | 0.9726 |       53 |
