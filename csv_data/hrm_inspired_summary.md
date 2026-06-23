# CTM Paper Results Summary

- Total experiments scanned: 81
- OK: 81
- Error: 0


## Stage hr00

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| cifar10 | baseline | 1 | 60.28 | 47.44 | 81.86 | 200000 |
| parity | baseline | 1 | 67.95 | 67.95 | 88.20 | 190000 |
| sort | baseline | 1 | 91.42 | 83.92 | - | 100000 |

## Stage hr01

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| cifar10 | bp1 | 1 | 73.79 | 72.88 | 80.12 | 200000 |
| cifar10 | bp10 | 1 | 78.98 | 78.25 | 82.78 | 200000 |
| cifar10 | bp2 | 1 | 77.57 | 77.05 | 81.86 | 200000 |
| cifar10 | bp5 | 1 | 81.59 | 81.53 | 83.14 | 200000 |
| parity | bp1 | 1 | 61.06 | 60.34 | 63.69 | 200000 |
| parity | bp10 | 1 | 52.84 | 52.53 | 53.73 | 200000 |
| parity | bp2 | 1 | 54.89 | 53.03 | 55.33 | 200000 |
| parity | bp5 | 1 | 53.25 | 53.21 | 54.01 | 200000 |
| sort | bp1 | 1 | 0.31 | 0.00 | - | 100000 |
| sort | bp10 | 1 | 0.31 | 0.00 | - | 100000 |
| sort | bp2 | 1 | 0.31 | 0.00 | - | 100000 |
| sort | bp5 | 1 | 0.62 | 0.00 | - | 100000 |

## Stage hr02

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| cifar10 | warmup0p1 | 1 | 80.42 | 80.25 | 83.17 | 200000 |
| cifar10 | warmup0p2 | 1 | 80.47 | 80.18 | 83.71 | 200000 |
| cifar10 | warmup0p5 | 1 | 79.56 | 79.07 | 82.57 | 200000 |
| parity | warmup0p1 | 1 | 53.99 | 53.52 | 54.66 | 200000 |
| parity | warmup0p2 | 1 | 55.18 | 51.81 | 55.87 | 200000 |
| parity | warmup0p5 | 1 | 54.89 | 52.67 | 55.33 | 200000 |
| sort | warmup0p1 | 1 | 0.31 | 0.00 | - | 100000 |
| sort | warmup0p2 | 1 | 0.31 | 0.07 | - | 100000 |
| sort | warmup0p5 | 1 | 0.31 | 0.00 | - | 100000 |

## Stage hr03

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| cifar10 | detach1 | 1 | 69.34 | 62.72 | 82.16 | 200000 |
| cifar10 | detach10 | 1 | 70.80 | 66.11 | 84.19 | 200000 |
| cifar10 | detach2 | 1 | 69.57 | 61.80 | 83.47 | 200000 |
| cifar10 | detach5 | 1 | 71.21 | 60.53 | 83.98 | 200000 |
| parity | detach1 | 1 | 66.21 | 65.84 | 77.32 | 180000 |
| parity | detach10 | 1 | 71.20 | 68.28 | 99.97 | 200000 |
| parity | detach2 | 1 | 62.74 | 62.74 | 76.57 | 130000 |
| parity | detach5 | 1 | 61.52 | 61.30 | 73.58 | 60000 |
| sort | detach1 | 1 | 20.42 | 17.40 | - | 100000 |
| sort | detach10 | 1 | 71.22 | 59.86 | - | 100000 |
| sort | detach2 | 1 | 28.96 | 27.05 | - | 100000 |
| sort | detach5 | 1 | 57.19 | 53.99 | - | 100000 |

## Stage hr04

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| cifar10 | atan2 | 1 | 68.27 | 63.50 | 84.32 | 200000 |
| cifar10 | atan2_b95 | 1 | 63.52 | 62.96 | 84.59 | 200000 |
| parity | atan2 | 1 | 63.25 | 63.23 | 79.62 | 50000 |
| parity | atan2_b95 | 1 | 76.28 | 76.20 | 99.97 | 200000 |
| sort | atan2 | 1 | 94.97 | 87.78 | - | 100000 |
| sort | atan2_b95 | 1 | 91.35 | 85.42 | - | 100000 |

## Stage hr05

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| cifar10 | stablemax | 1 | 69.25 | 59.71 | 84.18 | 200000 |
| parity | stablemax | 1 | 74.21 | 74.17 | 79.02 | 160000 |
| sort | stablemax | 1 | 0.35 | 0.07 | - | 100000 |

## Stage hr06

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| cifar10 | gated_attn | 1 | 74.05 | 70.35 | 84.87 | 200000 |
| parity | gated_attn | 1 | 77.12 | 76.93 | 99.96 | 120000 |

## Stage hr07

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| parity | inj_gru_gate | 1 | 71.63 | 71.28 | 86.91 | 190000 |

## Stage hr08

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| parity | h2_l2_iter4 | 1 | 61.42 | 61.34 | 72.39 | 200000 |
| parity | h2_l5_iter10 | 1 | 58.57 | 58.38 | 72.08 | 200000 |
| parity | h3_l3_iter9 | 1 | 60.48 | 60.45 | 72.24 | 200000 |
| sort | h1_l50 | 1 | 87.53 | 81.25 | - | 100000 |
| sort | h2_l25 | 1 | 94.55 | 89.31 | - | 100000 |
| sort | h2_l2_iter4 | 1 | 0.24 | 0.03 | - | 100000 |
| sort | h2_l5_iter10 | 1 | 0.31 | 0.03 | - | 100000 |
| sort | h3_l3_iter9 | 1 | 0.28 | 0.14 | - | 100000 |
| sort | h5_l10 | 1 | 92.57 | 89.06 | - | 100000 |

## Stage hr09

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| cifar10 | ema0p999 | 1 | 66.82 | 65.13 | 84.22 | 200000 |
| cifar10 | ema0p9999 | 1 | 69.31 | 66.72 | 83.82 | 200000 |
| parity | ema0p999 | 1 | 67.75 | 67.69 | 87.07 | 160000 |
| parity | ema0p9999 | 1 | 67.67 | 67.63 | 86.10 | 150000 |
| sort | ema0p999 | 1 | 87.53 | 81.25 | - | 100000 |
| sort | ema0p9999 | 1 | 87.53 | 81.25 | - | 100000 |

## Stage hr10

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| parity | act_h16_e0p05 | 1 | 56.54 | 54.41 | 55.69 | 200000 |
| parity | act_h16_e0p1 | 1 | 59.30 | 59.17 | 66.74 | 200000 |
| parity | act_h50_e0p05 | 1 | 69.32 | 69.32 | 79.65 | 150000 |
| parity | act_h50_e0p1 | 1 | 85.78 | 79.97 | 100.00 | 80000 |
| sort | act_h16_e0p05 | 1 | 0.10 | 0.00 | - | 100000 |
| sort | act_h16_e0p1 | 1 | 0.10 | 0.00 | - | 100000 |
| sort | act_h50_e0p05 | 1 | 83.61 | 78.61 | - | 100000 |
| sort | act_h50_e0p1 | 1 | 84.76 | 73.82 | - | 100000 |

## Stage hr11

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| parity | bp5_atan2 | 1 | 53.97 | 53.38 | 54.37 | 200000 |
| sort | bp5_atan2 | 1 | 0.31 | 0.10 | - | 100000 |

## Stage hr12

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| parity | h2_l5_bp3 | 1 | 55.94 | 55.81 | 62.94 | 200000 |
| sort | h2_l5_bp3 | 1 | 0.31 | 0.03 | - | 100000 |

## Stage hr13

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| parity | h2_l5_gated | 1 | 65.64 | 65.43 | 81.31 | 200000 |
| sort | h2_l5_gated | 1 | 0.31 | 0.03 | - | 100000 |

## Stage hr14

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| parity | bp5_atan2_smax | 1 | 52.36 | 51.84 | 52.58 | 200000 |
| sort | bp5_atan2_smax | 1 | 0.76 | 0.00 | - | 100000 |

## Stage hr15

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| parity | full_hrm | 1 | 54.58 | 53.95 | 56.46 | 200000 |
| sort | full_hrm | 1 | 0.31 | 0.03 | - | 100000 |
