# Offline GCBC comparison (merged from multi-GPU shards)

Runs: 480 | envs: 12 | backbones: ['ctm', 'flash', 'flash-deep', 'flash-shallow', 'gru', 'lstm', 'mlp', 'transformer']


## Expert data quality

| env | expert_succ |
|---|---|
| acrobot | 100.0% |
| acrobot-partial | 100.0% |
| cartpole | 50.0% |
| cartpole-partial | 50.0% |
| mountaincar | 100.0% |
| mountaincar-partial | 100.0% |
| pendulum | 20.0% |
| pendulum-partial | 20.0% |
| point-state | 100.0% |
| reacher | 100.0% |
| reacher-partial | 100.0% |
| tworoom-state | 100.0% |

## success_rate mean+-std

| env | ctm | flash | flash-deep | flash-shallow | gru | lstm | mlp | transformer |
|---|---|---|---|---|---|---|---|---|
| acrobot | 93.3+-4.2 | 97.5+-3.3 | 95.0+-4.9 | 96.7+-3.1 | 95.8+-2.6 | 97.5+-5.0 | 95.0+-3.1 | 94.2+-2.0 |
| acrobot-partial | 85.0+-10.1 | 85.0+-10.7 | 83.3+-9.5 | 0.0+-0.0 | 69.2+-15.5 | 59.2+-10.7 | 0.0+-0.0 | 0.8+-1.7 |
| cartpole | 40.0+-2.0 | 39.2+-2.0 | 39.2+-2.0 | 39.2+-2.0 | 39.2+-2.0 | 39.2+-2.0 | 39.2+-2.0 | 39.2+-2.0 |
| cartpole-partial | 39.2+-5.7 | 42.5+-4.1 | 41.7+-4.6 | 41.7+-3.7 | 43.3+-8.2 | 43.3+-7.7 | 41.7+-3.7 | 39.2+-5.0 |
| mountaincar | 0.0+-0.0 | 20.0+-26.7 | 0.0+-0.0 | 0.0+-0.0 | 29.2+-35.9 | 0.0+-0.0 | 0.0+-0.0 | 0.0+-0.0 |
| mountaincar-partial | 0.0+-0.0 | 2.5+-5.0 | 0.0+-0.0 | 0.0+-0.0 | 15.8+-31.7 | 0.0+-0.0 | 0.0+-0.0 | 0.0+-0.0 |
| pendulum | 18.3+-3.3 | 18.3+-4.2 | 19.2+-4.2 | 16.7+-2.6 | 23.3+-5.0 | 17.5+-3.1 | 17.5+-4.1 | 18.3+-2.0 |
| pendulum-partial | 21.7+-4.9 | 23.3+-4.2 | 20.0+-3.1 | 19.2+-3.3 | 20.8+-2.6 | 20.8+-5.3 | 19.2+-3.3 | 18.3+-3.3 |
| point-state | 100.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 | 97.5+-5.0 | 100.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 |
| reacher | 67.5+-8.5 | 62.5+-7.0 | 71.7+-11.3 | 63.3+-10.0 | 68.3+-9.4 | 69.2+-7.7 | 65.0+-4.2 | 68.3+-10.7 |
| reacher-partial | 60.8+-7.3 | 61.7+-6.7 | 61.7+-9.6 | 52.5+-8.6 | 64.2+-7.7 | 60.0+-7.7 | 53.3+-1.7 | 56.7+-4.2 |
| tworoom-state | 70.0+-8.5 | 75.0+-8.7 | 66.7+-11.8 | 32.5+-5.5 | 75.8+-4.1 | 71.7+-7.2 | 40.8+-8.1 | 65.0+-7.3 |

## Flash 混合 vs 单路径

| env | flash | shallow | deep | mix-shallow | mix-deep |
|---|---|---|---|---|---|
| acrobot | 97.5 | 96.7 | 95.0 | +0.8 | +2.5 |
| acrobot-partial | 85.0 | 0.0 | 83.3 | +85.0 | +1.7 |
| cartpole | 39.2 | 39.2 | 39.2 | +0.0 | +0.0 |
| cartpole-partial | 42.5 | 41.7 | 41.7 | +0.8 | +0.8 |
| mountaincar | 20.0 | 0.0 | 0.0 | +20.0 | +20.0 |
| mountaincar-partial | 2.5 | 0.0 | 0.0 | +2.5 | +2.5 |
| pendulum | 18.3 | 16.7 | 19.2 | +1.7 | -0.8 |
| pendulum-partial | 23.3 | 19.2 | 20.0 | +4.2 | +3.3 |
| point-state | 100.0 | 97.5 | 100.0 | +2.5 | +0.0 |
| reacher | 62.5 | 63.3 | 71.7 | -0.8 | -9.2 |
| reacher-partial | 61.7 | 52.5 | 61.7 | +9.2 | +0.0 |
| tworoom-state | 75.0 | 32.5 | 66.7 | +42.5 | +8.3 |

## CTM vs RNN 系记忆策略

| env | CTM | RNN 均值 | CTM-RNN | 判定 |
|---|---|---|---|---|
| acrobot | 93.3 | 95.8 | -2.5 | 持平 |
| acrobot-partial | 85.0 | 43.1 | +41.9 | CTM 赢 |
| cartpole | 40.0 | 39.2 | +0.8 | 持平 |
| cartpole-partial | 39.2 | 41.9 | -2.8 | 持平 |
| mountaincar | 0.0 | 9.7 | -9.7 | CTM 输 |
| mountaincar-partial | 0.0 | 5.3 | -5.3 | CTM 输 |
| pendulum | 18.3 | 19.7 | -1.4 | 持平 |
| pendulum-partial | 21.7 | 20.0 | +1.7 | 持平 |
| point-state | 100.0 | 100.0 | +0.0 | 持平 |
| reacher | 67.5 | 68.6 | -1.1 | 持平 |
| reacher-partial | 60.8 | 60.3 | +0.6 | 持平 |
| tworoom-state | 70.0 | 70.8 | -0.8 | 持平 |

## Partial vs Full obs (记忆 backbone 的 POMDP 优势)

| env | mlp_full | mlp_partial | Δmlp | ctm_full | ctm_partial | Δctm |
|---|---|---|---|---|---|---|
| acrobot | 95.0 | 0.0 | -95.0 | 93.3 | 85.0 | -8.3 |
| cartpole | 39.2 | 41.7 | +2.5 | 40.0 | 39.2 | -0.8 |
| mountaincar | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |
| pendulum | 17.5 | 19.2 | +1.7 | 18.3 | 21.7 | +3.3 |
| reacher | 65.0 | 53.3 | -11.7 | 67.5 | 60.8 | -6.7 |