# Offline GCBC comparison (merged from multi-GPU shards)

Runs: 240 | envs: 9 | backbones: ['cem-wm', 'ctm', 'flash', 'flash-deep', 'flash-shallow', 'gru', 'lstm', 'mlp', 'transformer']


## Expert data quality

| env | expert_succ |
|---|---|
| cartpole | 50.0% |
| cartpole-partial | 50.0% |
| mountaincar | 100.0% |
| mountaincar-partial | 100.0% |
| pendulum | 75.0% |
| pendulum-partial | 75.0% |
| point-state | 100.0% |
| reacher-partial | 100.0% |
| tworoom-state | 100.0% |

## success_rate mean+-std

| env | cem-wm | ctm | flash | flash-deep | flash-shallow | gru | lstm | mlp | transformer |
|---|---|---|---|---|---|---|---|---|---|
| cartpole | 69.4+-3.9 | 37.5+-0.0 | 37.5+-0.0 | 37.5+-0.0 | 37.5+-0.0 | 37.5+-0.0 | 37.5+-0.0 | 37.5+-0.0 | 37.5+-0.0 |
| cartpole-partial | 38.9+-5.2 | 36.1+-2.0 | 40.3+-3.9 | 38.9+-2.0 | 37.5+-3.4 | 37.5+-0.0 | 37.5+-0.0 | 40.3+-2.0 | 43.1+-3.9 |
| mountaincar | 0.0+-0.0 | 66.7+-47.1 | 100.0+-0.0 | 33.3+-47.1 | 98.6+-2.0 | 100.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 |
| mountaincar-partial | 0.0+-0.0 | 61.1+-43.3 | 100.0+-0.0 | 66.7+-47.1 | 94.4+-5.2 | 100.0+-0.0 | 100.0+-0.0 | 0.0+-0.0 | 97.2+-2.0 |
| pendulum | 55.6+-3.9 | 54.2+-3.4 | 54.2+-6.8 | 51.4+-2.0 | 43.1+-2.0 | 50.0+-3.4 | 51.4+-7.1 | 47.2+-10.9 | 43.1+-12.9 |
| pendulum-partial | 5.6+-2.0 | 48.6+-9.8 | 48.6+-2.0 | 50.0+-3.4 | 48.6+-2.0 | 44.4+-5.2 | 48.6+-3.9 | 12.5+-3.4 | 43.1+-3.9 |
| point-state | - | 100.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 |
| reacher-partial | 77.8+-2.0 | 66.7+-0.0 | 66.7+-6.8 | 76.4+-7.9 | 72.2+-5.2 | 66.7+-3.4 | 62.5+-3.4 | 54.2+-0.0 | 59.7+-2.0 |
| tworoom-state | 55.6+-2.0 | 72.2+-9.8 | 66.7+-5.9 | 72.2+-7.1 | 56.9+-15.7 | 76.4+-5.2 | 73.6+-8.6 | 37.5+-9.0 | 63.9+-8.6 |

## Flash 混合 vs 单路径

| env | flash | shallow | deep | mix-shallow | mix-deep |
|---|---|---|---|---|---|
| cartpole | 37.5 | 37.5 | 37.5 | +0.0 | +0.0 |
| cartpole-partial | 40.3 | 37.5 | 38.9 | +2.8 | +1.4 |
| mountaincar | 100.0 | 98.6 | 33.3 | +1.4 | +66.7 |
| mountaincar-partial | 100.0 | 94.4 | 66.7 | +5.6 | +33.3 |
| pendulum | 54.2 | 43.1 | 51.4 | +11.1 | +2.8 |
| pendulum-partial | 48.6 | 48.6 | 50.0 | +0.0 | -1.4 |
| point-state | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 |
| reacher-partial | 66.7 | 72.2 | 76.4 | -5.6 | -9.7 |
| tworoom-state | 66.7 | 56.9 | 72.2 | +9.7 | -5.6 |

## CTM vs RNN 系记忆策略

| env | CTM | RNN 均值 | CTM-RNN | 判定 |
|---|---|---|---|---|
| cartpole | 37.5 | 37.5 | +0.0 | 持平 |
| cartpole-partial | 36.1 | 39.4 | -3.2 | CTM 输 |
| mountaincar | 66.7 | 100.0 | -33.3 | CTM 输 |
| mountaincar-partial | 61.1 | 99.1 | -38.0 | CTM 输 |
| pendulum | 54.2 | 48.1 | +6.0 | CTM 赢 |
| pendulum-partial | 48.6 | 45.4 | +3.2 | CTM 赢 |
| point-state | 100.0 | 100.0 | +0.0 | 持平 |
| reacher-partial | 66.7 | 63.0 | +3.7 | CTM 赢 |
| tworoom-state | 72.2 | 71.3 | +0.9 | 持平 |

## Partial vs Full obs (记忆 backbone 的 POMDP 优势)

| env | mlp_full | mlp_partial | Δmlp | ctm_full | ctm_partial | Δctm |
|---|---|---|---|---|---|---|
| cartpole | 37.5 | 40.3 | +2.8 | 37.5 | 36.1 | -1.4 |
| mountaincar | 100.0 | 0.0 | -100.0 | 66.7 | 61.1 | -5.6 |
| pendulum | 47.2 | 12.5 | -34.7 | 54.2 | 48.6 | -5.6 |