# Offline comparison: GCBC on expert data vs world-model+CEM

生成: 2026-07-05 10:37:29 | runs: 36 | backbones: ['mlp', 'ctm', 'lstm', 'flash', 'flash-shallow', 'flash-deep']

范式: ExpertPolicy 收集 → GCBC 训练 (对齐 stable-wm)

配置: collect=100eps, gcbc=1000steps, noise=0.1, seeds=[0, 1, 2]


## Expert data quality (goal-reaching success rate)

| env | expert_succ | n_episodes | n_steps |
|---|---|---|---|
| reacher | 100.0% | 100 | 1083 |
| reacher-partial | 100.0% | 100 | 1083 |

## success_rate mean+-std (GCBC on expert data)

| env | mlp | ctm | lstm | flash | flash-shallow | flash-deep |
|---|---|---|---|---|---|---|
| reacher | 70.8+-5.9 | 72.9+-2.9 | 70.8+-2.9 | 70.8+-5.9 | 62.5+-5.1 | 70.8+-2.9 |
| reacher-partial | 60.4+-5.9 | 72.9+-5.9 | 66.7+-5.9 | 62.5+-8.8 | 56.2+-0.0 | 64.6+-5.9 |

## Flash 混合 vs 单路径

| env | flash(混合) | flash-shallow | flash-deep | 混合-shallow | 混合-deep |
|---|---|---|---|---|---|
| reacher | 70.8 | 62.5 | 70.8 | +8.3 | +0.0 |
| reacher-partial | 62.5 | 56.2 | 64.6 | +6.2 | -2.1 |

## CTM vs RNN 系记忆策略

| env | CTM | RNN 均值 | CTM-RNN | 判定 |
|---|---|---|---|---|
| reacher | 72.9 | 70.8 | +2.1 | 持平 |
| reacher-partial | 72.9 | 66.7 | +6.2 | CTM 赢 |
