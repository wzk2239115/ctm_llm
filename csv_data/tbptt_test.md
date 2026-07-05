# Offline comparison: GCBC on expert data vs world-model+CEM

生成: 2026-07-05 17:48:03 | runs: 6 | backbones: ['flash', 'mlp', 'ctm']

范式: ExpertPolicy 收集 → GCBC 训练 (对齐 stable-wm)

配置: collect=80eps, gcbc=500steps, noise=0.1, seeds=[0]


## Expert data quality (goal-reaching success rate)

| env | expert_succ | n_episodes | n_steps |
|---|---|---|---|
| mountaincar | 100.0% | 80 | 9931 |
| acrobot-partial | 100.0% | 80 | 8267 |

## success_rate mean+-std (GCBC on expert data)

| env | flash | mlp | ctm |
|---|---|---|---|
| acrobot-partial | 80.0+-0.0 | 0.0+-0.0 | 100.0+-0.0 |
| mountaincar | 100.0+-0.0 | 100.0+-0.0 | 100.0+-0.0 |
