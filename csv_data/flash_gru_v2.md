# Offline comparison: GCBC on expert data vs world-model+CEM

生成: 2026-07-05 18:54:45 | runs: 56 | backbones: ['mlp', 'ctm', 'lstm', 'flash', 'flash-shallow', 'flash-deep']

范式: ExpertPolicy 收集 → GCBC 训练 (对齐 stable-wm)

配置: collect=80eps, gcbc=500steps, noise=0.1, seeds=[0, 1]


## Expert data quality (goal-reaching success rate)

| env | expert_succ | n_episodes | n_steps |
|---|---|---|---|
| acrobot-partial | 100.0% | 80 | 8267 |
| acrobot | 100.0% | 80 | 8267 |
| reacher-partial | 100.0% | 80 | 899 |
| pendulum-partial | 75.0% | 80 | 3781 |

## success_rate mean+-std (GCBC on expert data)

| env | mlp | ctm | lstm | flash | flash-shallow | flash-deep |
|---|---|---|---|---|---|---|
| acrobot | 100.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 | -1.0+-0.0 | -1.0+-0.0 | -1.0+-0.0 |
| acrobot-partial | 0.0+-0.0 | 100.0+-0.0 | 66.7+-16.7 | -1.0+-0.0 | -1.0+-0.0 | -1.0+-0.0 |
| pendulum-partial | 20.8+-4.2 | 41.7+-8.3 | 50.0+-0.0 | -1.0+-0.0 | -1.0+-0.0 | -1.0+-0.0 |
| reacher-partial | 66.7+-0.0 | 58.3+-8.3 | 62.5+-4.2 | -1.0+-0.0 | -1.0+-0.0 | -1.0+-0.0 |

## Flash 混合 vs 单路径

| env | flash(混合) | flash-shallow | flash-deep | 混合-shallow | 混合-deep |
|---|---|---|---|---|---|
| acrobot | -1.0 | -1.0 | -1.0 | +0.0 | +0.0 |
| acrobot-partial | -1.0 | -1.0 | -1.0 | +0.0 | +0.0 |
| pendulum-partial | -1.0 | -1.0 | -1.0 | +0.0 | +0.0 |
| reacher-partial | -1.0 | -1.0 | -1.0 | +0.0 | +0.0 |

## CTM vs RNN 系记忆策略

| env | CTM | RNN 均值 | CTM-RNN | 判定 |
|---|---|---|---|---|
| acrobot | 100.0 | 100.0 | +0.0 | 持平 |
| acrobot-partial | 100.0 | 66.7 | +33.3 | CTM 赢 |
| pendulum-partial | 41.7 | 50.0 | -8.3 | CTM 输 |
| reacher-partial | 58.3 | 62.5 | -4.2 | CTM 输 |
