# Offline comparison: GCBC on expert data vs world-model+CEM

生成: 2026-07-05 10:27:53 | runs: 48 | backbones: ['flash', 'mlp', 'ctm', 'lstm', 'gru', 'transformer', 'flash-shallow', 'flash-deep']

范式: ExpertPolicy 收集 → GCBC 训练 (对齐 stable-wm)

配置: collect=100eps, gcbc=1000steps, noise=0.1, seeds=[0, 1, 2]


## Expert data quality (goal-reaching success rate)

| env | expert_succ | n_episodes | n_steps |
|---|---|---|---|
| reacher-partial | 100.0% | 100 | 1083 |
| reacher | 100.0% | 100 | 1083 |

## success_rate mean+-std (GCBC on expert data)

| env | flash | mlp | ctm | lstm | gru | transformer | flash-shallow | flash-deep |
|---|---|---|---|---|---|---|---|---|
| reacher | 62.5+-8.8 | 60.4+-5.9 | 72.9+-5.9 | 66.7+-5.9 | 68.8+-8.8 | 66.7+-7.8 | 56.2+-0.0 | 64.6+-5.9 |
| reacher-partial | 62.5+-8.8 | 60.4+-5.9 | 72.9+-5.9 | 66.7+-5.9 | 68.8+-8.8 | 66.7+-7.8 | 56.2+-0.0 | 64.6+-5.9 |

## Flash 混合 vs 单路径

| env | flash(混合) | flash-shallow | flash-deep | 混合-shallow | 混合-deep |
|---|---|---|---|---|---|
| reacher | 62.5 | 56.2 | 64.6 | +6.2 | -2.1 |
| reacher-partial | 62.5 | 56.2 | 64.6 | +6.2 | -2.1 |

## CTM vs RNN 系记忆策略

| env | CTM | RNN 均值 | CTM-RNN | 判定 |
|---|---|---|---|---|
| reacher | 72.9 | 67.4 | +5.6 | CTM 赢 |
| reacher-partial | 72.9 | 67.4 | +5.6 | CTM 赢 |
