# CTM Paper Results Summary

- Total experiments scanned: 118
- OK: 118
- Error: 0


## Stage baseline

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| cifar10 | baseline | 5 | 69.04 +/- 4.38 | 66.12 +/- 6.84 | 84.23 +/- 0.68 | 200000 |
| mazes | baseline | 5 | 89.05 +/- 0.99 | 88.34 +/- 2.11 | 90.05 +/- 0.95 | 100000 |
| parity | baseline | 5 | 69.66 +/- 2.40 | 68.01 +/- 2.11 | 97.02 +/- 5.11 | 200000 |
| qamnist | baseline | 5 | 38.83 +/- 10.92 | 21.26 +/- 2.99 | 99.57 +/- 0.09 | 200000 |
| sort | baseline | 5 | 90.51 +/- 4.36 | 85.16 +/- 5.05 | - | 100000 |

## Stage jepa

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| cifar10 | jepa_w0p1 | 5 | 73.55 +/- 4.10 | 71.52 +/- 4.52 | 84.44 +/- 0.14 | 200000 |
| mazes | jepa_w0p1 | 5 | 89.09 +/- 1.30 | 89.08 +/- 1.29 | 89.99 +/- 1.29 | 100000 |
| qamnist | jepa_w0p5 | 5 | 47.86 +/- 8.45 | 28.26 +/- 9.82 | 99.55 +/- 0.15 | 200000 |

## Stage revise

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| cifar10 | revise_w0.2_cp0.3 | 5 | 70.83 +/- 3.92 | 67.69 +/- 4.23 | 84.06 +/- 0.90 | 200000 |
| mazes | revise_w0.1_cp0.15 | 5 | 89.40 +/- 0.54 | 89.18 +/- 0.55 | 90.29 +/- 0.48 | 100000 |
| parity | revise_w0.1_cp0.15 | 5 | 72.11 +/- 3.86 | 71.78 +/- 3.70 | 97.93 +/- 3.02 | 200000 |

## Stage sparsity

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| cifar10 | sparsity0p1 | 5 | 70.14 +/- 3.98 | 66.48 +/- 4.36 | 83.80 +/- 0.62 | 200000 |
| cifar10 | sparsity0p25 | 5 | 68.26 +/- 4.61 | 64.80 +/- 2.93 | 84.36 +/- 0.41 | 200000 |
| cifar10 | sparsity0p5 | 5 | 70.20 +/- 3.31 | 66.22 +/- 3.49 | 84.31 +/- 0.64 | 174400 |
| cifar10 | sparsity0p75 | 5 | 66.92 +/- 3.49 | 62.65 +/- 4.64 | 82.60 +/- 1.04 | 71200 |
| mazes | sparsity0p1 | 5 | 89.03 +/- 0.72 | 88.87 +/- 0.80 | 89.57 +/- 0.74 | 100000 |
| mazes | sparsity0p25 | 5 | 89.45 +/- 0.86 | 89.07 +/- 0.92 | 90.22 +/- 0.81 | 100000 |
| mazes | sparsity0p5 | 5 | 89.93 +/- 0.47 | 89.85 +/- 0.53 | 90.63 +/- 0.39 | 100000 |
| mazes | sparsity0p75 | 5 | 89.55 +/- 1.01 | 89.51 +/- 1.00 | 90.26 +/- 1.01 | 100000 |
| parity | sparsity0p1 | 5 | 66.92 +/- 3.62 | 65.84 +/- 5.25 | 89.98 +/- 7.22 | 200000 |
| parity | sparsity0p25 | 4 | 71.95 +/- 3.45 | 71.72 +/- 3.37 | 99.97 +/- 0.06 | 200000 |
| parity | sparsity0p5 | 5 | 72.56 +/- 4.31 | 70.58 +/- 5.59 | 95.97 +/- 9.01 | 200000 |
| parity | sparsity0p75 | 4 | 72.78 +/- 3.37 | 72.07 +/- 3.00 | 97.75 +/- 4.50 | 200000 |
| sort | sparsity0p5 | 5 | 90.22 +/- 2.82 | 87.08 +/- 2.69 | - | 100000 |
