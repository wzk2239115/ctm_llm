# Offline comparison: GCBC on expert data vs world-model+CEM

生成: 2026-07-05 19:11:59 | runs: 30 | backbones: ['mlp', 'ctm', 'lstm', 'flash']

范式: ExpertPolicy 收集 → GCBC 训练 (对齐 stable-wm)

配置: collect=100eps, gcbc=1000steps, noise=0.1, seeds=[0, 1]


## Expert data quality (goal-reaching success rate)

| env | expert_succ | n_episodes | n_steps |
|---|---|---|---|
| acrobot-partial | 100.0% | 100 | 10243 |
| mountaincar-partial | 100.0% | 100 | 12421 |
| reacher-partial | 100.0% | 100 | 1083 |

## success_rate mean+-std (GCBC on expert data)

| env | mlp | ctm | lstm | flash |
|---|---|---|---|---|
| acrobot-partial | 0.0+-0.0 | 87.5+-12.5 | 83.3+-0.0 | 95.8+-4.2 |
| mountaincar-partial | 0.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 |
| reacher-partial | 66.7+-0.0 | 79.2+-4.2 | 79.2+-12.5 | 79.2+-4.2 |

## CTM vs RNN 系记忆策略

| env | CTM | RNN 均值 | CTM-RNN | 判定 |
|---|---|---|---|---|
| acrobot-partial | 87.5 | 83.3 | +4.2 | CTM 赢 |
| mountaincar-partial | 100.0 | 100.0 | +0.0 | 持平 |
| reacher-partial | 79.2 | 79.2 | -0.0 | 持平 |
