# CTM Paper Results Summary

- Total experiments scanned: 109
- OK: 109
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
| cifar10 | revise_w0.2_cp0.3 | 5 | 70.88 +/- 2.66 | 66.30 +/- 3.64 | 84.68 +/- 0.22 | 200000 |
| mazes | revise_w0.1_cp0.15 | 5 | 89.06 +/- 1.23 | 89.02 +/- 1.23 | 90.04 +/- 1.29 | 100000 |

## Stage sparsity

| task | sweep | seeds | best_test_acc | final_test_acc | best_test_acc_mc | final_iter |
|---|---|---|---|---|---|---|
| cifar10 | sparsity0p1 | 5 | 72.16 +/- 3.54 | 68.46 +/- 2.97 | 84.28 +/- 0.30 | 200000 |
| cifar10 | sparsity0p25 | 4 | 74.16 +/- 2.64 | 72.54 +/- 3.43 | 84.76 +/- 0.55 | 200000 |
| cifar10 | sparsity0p5 | 5 | 70.14 +/- 1.21 | 67.29 +/- 2.93 | 84.49 +/- 0.36 | 200000 |
| cifar10 | sparsity0p75 | 5 | 70.48 +/- 1.79 | 68.27 +/- 2.61 | 84.29 +/- 0.58 | 200000 |
| mazes | sparsity0p1 | 5 | 89.49 +/- 1.02 | 89.45 +/- 1.01 | 90.00 +/- 1.04 | 100000 |
| mazes | sparsity0p25 | 5 | 88.51 +/- 1.68 | 87.98 +/- 2.67 | 89.15 +/- 1.59 | 100000 |
| mazes | sparsity0p5 | 5 | 89.68 +/- 0.86 | 89.60 +/- 0.94 | 90.35 +/- 0.84 | 100000 |
| mazes | sparsity0p75 | 5 | 89.42 +/- 0.97 | 88.71 +/- 1.73 | 90.18 +/- 0.91 | 100000 |
| parity | sparsity0p1 | 5 | 71.03 +/- 6.51 | 70.38 +/- 7.08 | 80.65 +/- 12.88 | 200000 |
| parity | sparsity0p25 | 5 | 75.75 +/- 5.13 | 74.83 +/- 6.49 | 95.47 +/- 6.69 | 200000 |
| parity | sparsity0p5 | 5 | 74.76 +/- 3.23 | 74.61 +/- 3.08 | 92.95 +/- 6.74 | 200000 |
| parity | sparsity0p75 | 5 | 74.77 +/- 5.09 | 73.45 +/- 5.71 | 95.20 +/- 7.44 | 200000 |
