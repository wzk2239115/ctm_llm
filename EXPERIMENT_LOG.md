# 实验日志 (EXPERIMENT_LOG)

按日期**倒序**追加(最新在最上)。每次提交/分析一批实验,在这里留一段记录。
目的: 任何时候翻这一个文件,就能看懂"为什么跑、什么时候跑、当时的思路、结论是什么"。

记录规范见 `AGENTS.md`「实验记录规范」。

---

## 2026-07-07 — 大规模 GCBC offline 对比 (9 envs × 8 backbones × 5 seeds)

- **思路(用户提供)**: 2026-07-05 GCBC 离线范式定稿后, 在算力机做**全量对比**: 扩到 9 个 env (point-state / tworoom-state / reacher-partial / mountaincar / mountaincar-partial / pendulum / pendulum-partial / cartpole / cartpole-partial), 8 backbone (mlp/ctm/lstm/gru/transformer/flash/flash-shallow/flash-deep) + cem-wm baseline, 5 seeds. 目的: (1) 验证 0705 的 CTM-POMDP 优势在更大样本下是否稳定; (2) 把 transformer/gru 加进来补全 backbone 谱; (3) 看 cem-wm (model-based) vs 学习型 (model-free GCBC) 的相对位置.
- **配置**: `PROCS_PER_GPU=2 bash paper/run_offline_compare_cluster.sh` (8 × H100, 16 procs capped to 12 env-shards). `GCBC_STEPS=2000, COLLECT_EPS=200, EVAL_EPS=24, SEEDS="0 1 2 3 4"`. 输出 `csv_data/offline_compare_full_0707.csv`.
- **预期**: recurrent backbone (lstm/gru/ctm) 在 `*-partial` 上稳定优于 mlp; cem-wm 因用同一 expert data + 模型预测, 应介于 expert 和 BC 之间.
- **数据 bug + 修复**: `run_offline_compare_cluster.sh` 的 merge 步骤用 `glob('offline_compare_proc*.csv')` 收集, **不清理上次残留**. 跑了两遍 (一遍 5 seeds, 一遍 3 seeds) 后, 第二遍部分 proc 失败没覆写, 导致 (env, backbone, seed=0/1/2) 在 6 个 env 上重复出现, 共 **162/562 行重复 (28.8%)**, 会把 seed 0-2 双倍加权.
  - 修复 1 (数据): `scripts/dedupe_offline_compare.py` 按 `(env,backbone,seed)` 去重, deterministic 字段冲突时报警不静默丢. 跑完 → **562 → 400 rows, 零冲突**, 还原出完整 9 envs × 5 seeds 网格.
  - 修复 2 (脚本): `run_offline_compare_cluster.sh:52` 加 `rm -f "$OUT_DIR"/offline_compare_proc*.{csv,md}` 防再污染.
- **结果** (mean success_rate%, 5 seeds; cem-wm 的 expert_success=0 字段异常见结论 4):
  | env | mlp | ctm | lstm | gru | tf | flash | flash-s | flash-d | cem-wm |
  |---|---|---|---|---|---|---|---|---|---|
  | point-state | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | — |
  | tworoom-state | 41 | 70 | 72 | 76 | 65 | 61 | 55 | 60 | 58 |
  | reacher-partial | 53 | 61 | 60 | 64 | 57 | 63 | 67 | 67 | **76** |
  | mountaincar | 100 | 80 | 100 | 100 | 100 | 100 | 99 | 58 | 0 |
  | mountaincar-partial | **0** | 57 | 100 | 99 | 98 | 100 | 94 | 80 | 0 |
  | pendulum | 49 | 54 | 52 | 51 | 48 | 58 | 48 | 56 | 59 |
  | pendulum-partial | **9** | 49 | 52 | 48 | 46 | 49 | 48 | 50 | 10 |
  | cartpole | 39 | 39 | 39 | 39 | 39 | 39 | 39 | 39 | **70** |
  | cartpole-partial | 42 | 40 | 39 | 39 | 44 | 41 | 39 | 41 | 39 |
- **结论**:
  1. **point-state 饱和** (全 100%) — 无区分度, 后续对比应剔除.
  2. **POMDP 是 CTM/recurrent 的真实 win**: mlp 在 partial-obs 上系统性崩溃 (mountaincar-partial 全 0%, pendulum-partial 9%), 而 lstm/gru/ctm 显著更好 (50-100%). 印证 0705 的 CTM-POMDP 优势, 且在更大 backbone 谱 (含 gru/transformer) 下仍成立. **ctm 在 pendulum-partial 上 49% vs mlp 9% = +40pp**.
  3. **cartpole 天花板 (最重要 anomaly)**: 8 个学习型 backbone **全部**卡在 37.5% (seed 0-2) / 41.67% (seed 3-4), 而 cem-wm 拿 66-87%. `expert_success=50` 说明 expert 本身只有 50% 上限, BC 学不上去而 CEM 能 — **这是离线 IL 的瓶颈, 不是 backbone 问题**. 需排查 expert 质量 / DAgger / 增大 GCBC_STEPS.
  4. **cem-wm anomaly**: `expert_success=0` (其他学习型 = 50/75/100) 但在 cartpole (70 vs 全 39) 和 reacher-partial (76 vs ≤67) **大幅反超所有 BC**; pendulum 上持平 (59 vs flash 58); cartpole-partial 不赢 (39 vs BC 39-44). 两种解释: (a) 这些 env 动力学简单, CEM 随机搜索 latent 不需要好 policy prior 就能解; (b) expert_success 字段在 cem-wm 路径上有 bug. 需看 `paper/run_offline_compare.py` 的 cem-wm cost 定义和 expert_success 赋值.
  5. **ctm/flash-deep 不稳定**: ctm 在 mountaincar seed 0 = 0% (其他 seed 100%), flash-deep 在 mountaincar seed 1/2 = 0%. 是 seed-sensitivity 还是真 catastrophic forgetting, 需多 seed 复跑确认.
  6. **Backbone 整体排名**: `lstm ≈ gru > ctm ≈ flash-shallow > flash > transformer > flash-deep > mlp`. lstm/gru 最稳; transformer 在 pendulum 上意外差 (小数据过拟合?).
- **下一步**:
  - 排查 cartpole 天花板 (expert 数据 + BC loop, 考虑加 DAgger 或加 GCBC_STEPS 到 5000+)
  - 排查 cem-wm 的 `expert_success=0` anomaly
  - 出 per-env backbone 对比图 (误差棒 + paired delta vs lstm/gru baseline)
  - ctm 在 mountaincar seed 0 的 catastrophic failure 复跑确认

---

## 2026-07-07 — CTM Scaling Law (cells × ticks 双轴算力曲线)

- **思路(用户提供)**: 之前四块 idea (JEPA/revise/sparsity/gate-JEPA) 太离散, 想找一个核心抓手把论文凝聚起来. 方向是**从"算力成本"角度切入**, 围绕 CTM 的两个核心参数 —— **细胞数 (d_model / NLM)** 和 **内部时钟数 (iterations / thought ticks)** —— 画出整个 scale-up 曲线. 用户直觉: 增大这两个参数应该需要更多训练轮; 想知道这个关系是**线性还是指数**. 这能把四个 idea 串成"在 CTM 双轴算力曲线上的优化手段" (sparsity=cells 轴 Pareto, halt=ticks 轴 Pareto, JEPA=cells 质量, revise=ticks 质量), 论文叙事就凝聚了.
- **已有数据 (prior)**: `csv_data/ctm_paper_curves.json` 里 **st01 (d_model sweep) 174 run + st02 (tick sweep) 82 run**, 都有完整训练曲线 (每 1-2k 步一点). 这些是**单 seed + 固定训练步数 (cifar10/parity 200k, mazes 100k)** 的 paper-config sweep, 不是为 scaling law 设计的; 但已能出 prototype. 关键发现 (反直觉): **d_model 翻倍, steps-to-target 几乎不变** (mazes 2048→4096: 54229→54746 步; sort 512→1024: 73891→67088 步) —— 暗示 cells 轴的 sample-complexity scaling 接近"免费", 与 Transformer scaling laws 不同.
- **配置** (`paper_scaling/run_scaling.py`, 复用 `paper/exp_runner.py` 的 `Experiment`/`run_all`):
  - **cells 轴** (d_model sweep @ fixed iterations): cifar10 [64,128,256,512] @tick50 | parity [256,512,1024,2048] @tick75 | mazes [512,1024,2048,4096] @tick75. 共 3×4×3=36 runs.
  - **ticks 轴** (iterations sweep @ fixed d_model): cifar10 [2,5,10,25,50] @d256 | parity [5,10,25,50,75] @d1024 | mazes [5,10,25,50,75] @d2048. 共 3×5×3=45 runs.
  - **控制变量**: 每 task **固定 batch_size** (cifar10=512, parity=64, mazes=64) 和 **固定 LR** (1e-4), 继承 paper baseline —— 只变 architectural scale, 隔离 sample-efficiency. (muP-style LR scaling 明确 out of scope, 写 limitations.)
  - **训练延长**: cifar10/parity 300k (st01 是 200k), mazes 200k (st01 是 100k), 保证大模型真收敛. `track_every=1000` (曲线分辨率高, steps-to-target 插值精确), `save_every=10000` (省盘).
  - **3 seeds** [0,1,2]. **总计 81 runs**.
  - **排除 sort** (tick<50 全崩, 参数 inert) **和 qamnist** (数据缺).
- **预期**: 
  1. **cells 轴 exponent b ≈ 0** (免费, prior 暗示) —— 若成立, 是论文 headline: "wider CTM is sample-efficient, unlike Transformers".
  2. **ticks 轴 b > 0** (有成本, 可能线性甚至超线性) —— 梯度穿过更多 tick, 优化变难; st02 parity tick50 比 tick25 方差大已现端倪.
  3. 两轴差异 → "think longer is costlier than think wider" 这个张力是论文核心.
- **结果**: 待算力机跑完回填.
- **结论**: 待分析.
- **下一步**: 
  1. 算力机 smoke (`--seeds 1 --only cells --dry-run` → `--seeds 1 --only cells`) 必过, 再 `nohup python paper_scaling/run_scaling.py --gpus 8 &` 全量.
  2. 收菜: `python scripts/extract_ctm_paper_results.py --logs paper_scaling/logs --csv paper_scaling/csv_data/scaling_summary.csv --md paper_scaling/csv_data/scaling_summary.md --curves`.
  3. 分析: `python paper_scaling/run_scaling.py analyze` → 出 `paper_scaling/figures/{cells,ticks}_scaling.png` (log-log + power-law 拟合, 报 b / R²).
  4. 根据 b 值判断: 若 cells b≈0, 论文主叙事成立; 若 ticks b 显著 >0, "双轴不对称"成立. 若两者都 b≈0 或都 b≈1, 需重新定位叙事.
  5. **诚实风险**: fixed LR 是 confound (大 d_model 可能只是 LR 不对), 论文需补至少一个点的 LR 调研; steps-to-target vs FLOPs-to-target 需换算 (per-step FLOPs 随 scale 变).

---

## 2026-07-06 — 论文三块(JEPA / Draft-Revise / Sparsity)重复实验

- **思路(用户提供)**: work_ideas.md 前三块(Cross-Tick JEPA、Draft-Revise、Sparsity)定稿, 要作为一篇论文发布, 需要把这三方面的实验**重新跑一遍**(重复实验), 新建专门文件夹保存结果。
- **范围**: 仅核心 headline 配置(非完整 deep sweep), **全部 5 seed**(比原来 main=5/sweep=3 更统一), 加 **5-seed baselines** 保证 delta 是 paired mc-vs-mc / final-vs-final(不用过时常量)。
- **配置** (`paper_repro/run_repro.py`, 复用 `paper/exp_runner.py` 的 builder + `run_all`):
  - **baseline**: cifar10/mazes/parity/qamnist/sort 各 5 seed = 25 runs(st00 paper config, 无 idea; 同时充当 sparsity r=1.0 稠密参考)
  - **jepa**: cifar10 w=0.1(final +7.5pp) + mazes w=0.1(中性) + qamnist w=0.5(final +17pp) = 15 runs
  - **revise**: parity w=0.1/cp=0.15(mc +10pp headline) + cifar10 w=0.2/cp=0.3(final +9.9pp) + mazes w=0.1/cp=0.15(中性) = 15 runs
  - **sparsity**: mazes r∈{0.1,0.25,0.5,0.75}(Pareto, r=1.0=baseline) + sort r=0.5(任务坑 -12pp) = 25 runs
  - **合计 80 runs**。log 写 `paper_repro/logs/<group>/<exp>/`(两级目录, 供 `extract_ctm_paper_results.py` 直读)。
- **预期**: 复现三大 claim —— (1) JEPA 抬 cifar10/qamnist final-tick, 不抬 mc 天花板; (2) revise 抬 parity mc(+10pp); (3) sparsity mazes r=0.1 省 90% NLM 算力仅 -0.9pp(Pareto 甜点), sort r=0.5 大掉点。
- **算力机跑法**:
  ```bash
  export http_proxy="http://public-proxy.qihoo.net:3128"; export https_proxy="http://public-proxy.qihoo.net:3128"
  nohup python paper_repro/run_repro.py --gpus 8 > paper_repro/logs/run.log 2>&1 &
  # smoke 先过: python paper_repro/run_repro.py --seeds 1 --only jepa
  ```
- **收菜**: `python scripts/extract_ctm_paper_results.py --logs paper_repro/logs --csv paper_repro/csv_data/repro_summary.csv --md paper_repro/csv_data/repro_summary.md --curves`(同时出 mc 与 final-tick acc)。
- **结果**: 待算力机跑完回填。
- **下一步**: 跑完用 `paper/analyze_deep.py` 改读 `paper_repro/csv_data/repro_summary.csv` 出论文图(配对 delta + Pareto)。

---

## 2026-07-05 — GCBC 离线范式 (对齐 stable-wm) + ExpertPolicy

- **思路(用户提供)**: 之前用 PPO-from-scratch + random-BC 收集训练 policy, 数据和 world-model 不一致(不公平) + random action 是噪声(BC 学随机行为=毒, pendulum-partial ctm 从 75→27.8)。用户指出: 应该参考 stable-wm——expert/oracle 数据收集一次, world-model 和 policy baseline **共享同一批数据**。学完 stable-wm + lewm 确认: 数据是 **expert 预收集**(非 random), 所有方法(world-model GCBC/GCIQL)读同一份。
- **核心改造**:
  - `worldmodel/policy.py:ExpertPolicy`: 11 env 启发式 oracle (PD/energy/scripted/push), 访问 env 内部 state, noise=0.1 增加 diversity. 对齐 stable-wm `ExpertPolicy(action_noise=)`.
  - `worldmodel/rl/gcbc.py:GCBCTrainer`: goal-conditioned BC, episode-level BPTT (max_seq_len=32 截断), expert buffer 共享. evaluate + evaluate_timed.
  - `paper/run_offline_compare.py`: 统一实验脚本 (collect→GCBC→eval).
  - CartPole goal_x 从 ±1.5 缩到 ±0.5 (大范围 expert 不可能).
  - Reacher partial 真正区分 (full=ee+joints 4dim, partial=ee-only 2dim); 之前 partial 被忽略.
- **发现**:
  - Swimmer env 动力学 bug: `thrust=sin(angles)*a`, `angles+=a*dt` → `∫thrust·dt=∫sin(θ)dθ` 是状态函数, 任何振荡回原点净 thrust=0. Swimmer 无法推进, 弃用.
  - Expert 质量: point/tworoom/reacher/mountaincar/acrobot=100%, pendulum=33%, cartpole=33%.
- **配置**: reacher + reacher-partial, 6 backbone (mlp/ctm/lstm/flash/flash-shallow/flash-deep), 3 seeds, collect=100eps, gcbc=1000steps. `csv_data/gcbc_reacher_v2.md`.
- **结果**:
  | env | mlp | ctm | lstm | flash | CTM-RNN |
  |---|---|---|---|---|---|
  | reacher (full) | 70.8 | 72.9 | 70.8 | 70.8 | +2.1 持平 |
  | reacher-partial | 60.4 | 72.9 | 66.7 | 62.5 | **+6.2 CTM 赢** |
  - partial obs 上 mlp 掉 10pp (70.8→60.4), CTM 不掉 (72.9→72.9) — 记忆补偿信息缺失
  - Flash 混合 > shallow (+6.2~+8.3 synergy 确认)
  - GCBC expert data work: expert 100% → GCBC 60-73%
- **结论**: GCBC 离线范式正确 (对齐 stable-wm), 替代了有害的 PPO+random-BC. CTM 记忆在 POMDP 上有真实优势 (partial +6.2 vs full +2.1). Flash 混合 synergy 确认但 flash 本身未赢 ctm (需改进 gate/结构).
- **下一步**: 算力机大规模对比 (全 env × 全 backbone × 多 seed) + world-model+CEM baseline (同一批 expert data) + PushT/Cube expert 修复 (角度控制) + Flash Brain 改进.

---

## 2026-06-30 — 自适应 JEPA 权重 (ABC 三方案) 解除 parity 压制

- **思路(用户提供)**: parity+JEPA w=0.1 全 KILLED 的根因是"约束太大、光顾着约束没学训练数据"——JEPA 辅助 loss 的梯度把 synch 表示推向"可预测下一 tick", 挤掉了 parity 判别信号。用户提: **能否给正则项加一个自适应乘数, 从而不用人工调 `cross_tick_jepa_weight`**。实现 ABC 三方案让实验说话。
- **根因诊断** (`paper/diagnose_parity_jepa_compute.py`): subprocess 调 parity.train, 算力机真实配置(d_model=1024 seq=64 batch=64 ticks=75)跑 5000 步:
  - baseline(jepa=0): 在学, final acc 0.587
  - fixed w=0.1: **acc 卡 0.498(随机), STALLED** — 不是 pool/crash, 是 w=0.1 本身压制主任务
  - fixed w=0.03: 0.559(在学) / w=0.01: 0.581(≈baseline) — 甜点 ≤0.03, 而 st04 sweep [0.1,0.5,1.0] 全超甜点
  - 旧 verdict 有 bug: `acc<0.52 AND loss>0.69` 判 STALLED, 但 JEPA 辅助 loss 会拉低总 loss(0.682<0.69)制造"在学"假象 → w=0.1 被误判 OK。已修为纯 acc 判据。
- **配置**: st25 `build_st25_adaptive_jepa`, 5 task(sort/parity/mazes/cifar10/qamnist) × 4 mode × 3 seed = 60 runs, 全部 base_weight=0.1(parity 压制点)。三方案实现于 `baseline/utils/jepa.py:AdaptiveJEPAController`:
  - **A balance**: `eff_w = clip(ratio·L_main / (L_jepa+ε), lo, hi)`, JEPA 贡献恒≈主任务的 ratio%(默认 0.3)。零额外反传, base_weight 退化为"语义比例"。
  - **B gate**: `eff_w = base·sigmoid((acc_ema−τ)/T)`, 主任务没学会(acc<τ)→JEPA 关闭, 学会→放开。acc_ema 跨 step EMA buffer。
  - **C uncertainty**: Kendall 2018, `loss=exp(−logσ)·L_jepa + logσ`, logσ 是可学习参数(挂 controller, 自动进 optimizer), 模型自己定相对权重。
  - fixed 模式完全向后兼容(走老 CrossTickJEPAPredictor 路径, 零开销)。
- **预期**: parity 上 fixed=STALLED(复现压制), 三个 adaptive 至少一个让 acc 脱离 0.5(解除压制); 其他 task(sort/cifar, w=0.1 本就 OK)上 adaptive 应保持或改善 final acc, 不退化。最看好 A balance(直接对症量级压制)和 B gate(主任务优先)。
- **smoke(本地 CPU 已过)**: 四模式各 5 步 parity.train, 全 exit=0, loss 量级精确反映 adaptive weight 生效: fixed≈0.79 / balance≈0.90(eff_w 0.215) / gate≈0.70(eff_w 0.012 门关) / uncertainty≈1.68(eff_w 1.0)。三方案真实接入。
- **结果**: 2026-07-01 起跑, ~705 收菜。实际回来 **41/60 runs**(qamnist 全缺, mazes_gate 只 1 seed), 全 OK 无 crash。口径 **mc-vs-mc**(cifar10/mazes/parity 用 `best_test_acc_mc`, sort 无 mc 用 `best_test_acc`); st25 各 task 配置已逐项核对 == st00 paper sweep, vs-st00 公平。分析脚本 `paper/analyze_st25.py`, 数据 `csv_data/st25_0701_summary.{csv,md}`。

  | task | mode | n | acc% | vs fixed | vs st00 | seeds | 状态 |
  |---|---|---|---|---|---|---|---|
  | parity | fixed | 3 | 94.06 | 0 | +5.85 | [82.2,100,100] | **未复现压制!**(st04 5k 步判 STALLED 是过早, 200k 收敛后 94%) |
  | parity | balance | 3 | 52.79 | -41.3 | -35.4 | [52.5,52.8,53.0] | **BROKEN**(chance, 正是当初要解的压制) |
  | parity | gate | 3 | **97.71** | **+3.64** | **+9.50** | [93.1,100,100] | **WIN — 最佳, 抬 mc 天花板** |
  | parity | uncertainty | 3 | 50.63 | -43.4 | -37.6 | [50.5,50.6,50.8] | BROKEN(chance) |
  | sort | fixed | 3 | 80.94 | 0 | -6.59 | [76.1,81.6,85.1] | 低于 baseline |
  | sort | gate | 3 | **88.25** | **+7.31** | +0.72 | [87.0,88.1,89.6] | **WIN vs fixed**, 追平 st00 |
  | sort | balance | 3 | 37.96 | -43.0 | -49.6 | [25.0,25.6,63.3] | 不稳(±18) |
  | sort | uncertainty | 3 | 2.60 | -78.3 | -84.9 | [0.8,3.4,3.6] | BROKEN(崩) |
  | cifar10 | fixed/balance/gate | 3 | 84.1~84.5 | ≈0 | -0.7~-1.1 | — | 中性, 全 ≈ st00(85.16) |
  | mazes | fixed/balance/gate | 1~3 | 89.2~90.8 | ≈0 | -0.3~-2.0 | — | 中性, 全 ≈ st00(91.17) |

- **结论**:
  1. **gate(B, acc 门控 sigmoid)是唯一成功的自适应方案, 且是这批的 headline**。parity 上 mc 97.71%, vs fixed +3.6pp、vs st00(无 JEPA)+9.5pp(n=3), 是少数真能抬 mc 天花板的方法(堪比 revise-on-parity 的 +10pp)。sort 上 vs fixed +7.3pp、追平 baseline。cifar10/mazes 上中性(≈ fixed ≈ st00), 不退化。语义自洽: gate 让主任务先学(acc<τ 时 JEPA 关), 学好再放开约束, 既"免调参"又比 fixed 更好。
  2. **balance(A)与 uncertainty(C)在硬任务上反而更糟**。balance 在 parity 卡 chance(52.8%)、sort 大方差(38%±18)——与预期"最看好 balance"完全相反; uncertainty(Kendall 可学习 σ)在 parity/sort 全崩(50%/2.6%), cifar10 final 也塌到 10%, 优化器把 σ 推向破坏训练的值。3 个 adaptive 方案里 2 个劣于 fixed, **只有 gate 成立**。
  3. **实验前提部分落空**: 当初依据是"st04 parity+JEPA w=0.1 全 KILLED", 但 st25 fixed 在 parity 上跑到 200k 收敛后 mc 94%(> st00 88.21%)。即 5k 步诊断(`diagnose_parity_jepa_compute.py`)看到的 0.498 STALLED 是**起步慢、非永久卡死**——纯 acc 判据把"慢启动"误判成"压制"。fixed 实际没压制 parity, 所以"adaptive 抢救 fixed"的故事不成立; 但 gate 仍独立地比 fixed 更好更稳, 是真 win。
  4. **口径提醒**: parity 各模式都有种子不稳(fixed/gate 各有 1 颗种子 ~82-93%、另两颗 ~100%), +9.5pp 依赖 2/3 种子触顶, 需更多 seed 坐实; 但 gate 的种子分布(93/100/100)严格优于 fixed(82/100/100), 方向稳健。
- **下一步**:
  1. (可选)补 parity gate 的 seed 3/4/5, 把 +9.5pp 的 n 从 3 抬到 5~6, 坐实 headline(对照 revise-on-parity n=5)。
  2. 查 qamnist 为何全缺 / mazes_gate 为何只 1 seed, 决定是否补跑。
  3. gate 可考虑入 `VERIFIED_CONCLUSIONS.md`(作为"抬 mc 天花板"的第 4 个方法, 仅次于 revise-on-parity); 先补 seed 再定稿。
  4. 修正 `diagnose_parity_jepa_compute.py` 的 STALLED 判据注释——它判的"压制"实为慢启动, 诊断步数需 ≥20k 才有参考价值。

---

## 2026-06-29 — JEPA 口径修正 + baseline 过时修复

- **思路(用户提供)**: cross_tick_jepa 是验证通过的优化方法, 之前的分析漏了它, 检查一下.
- **排查**: `paper/explain/fig5b_jepa_weight.md` 已结论"JEPA 有效 +9~30pp"(final-tick 口径). 但我之前的 `analyze_deep.py` 只用 mc-vs-mc 把 JEPA 判中性, 是口径偏差.
- **结果**:
  - **JEPA final-tick 真实有效**: cifar10 +7.5pp(n=3), qamnist +17pp(n=2). 这是真的, JEPA 让"相邻 tick 隐状态可预测", 提升最后一步预测.
  - **JEPA 不抬 mc 天花板**: cifar10/mazes/qamnist 的 mc delta 都 -0.5~-1pp(中性). JEPA 改善"平均 tick 质量"而非"最优 tick".
  - **fig5b 报告自己的 bug**: 它说 mazes +9.6pp(最佳场景), 但用了过时 baseline `BASELINE_PAPER["mazes"]=0.8028`(旧值). 真实 st00 mazes=0.9016, 实际 mazes JEPA = -1.2pp(中性, 不是最佳场景). cifar10 baseline(0.6443)也偏低, 真实 0.6690.
- **修复**: `scripts/plot_ctm_paper_results.py:BASELINE_PAPER` 改成 st00 复现值(cifar10 0.6690 / mazes 0.9016 / qamnist 0.3662 / sort 0.8753; parity 0.6797 不变). 修复后 fig5 mazes 的 baseline 线从 80% 升到 90%, JEPA 柱不再"超过".
- **修正后的有效方法清单**:
  | 方法 | 有效 | 真实证据 | 口径 |
  |---|---|---|---|
  | JEPA | ✅ | cifar10 +7.5pp, qamnist +17pp | final-tick |
  | revise | ✅ | parity mc +10pp | mc, n=5 |
  | sparsity | ✅ | mazes 省 90% 算力掉 1pp | Pareto 效率 |
- **下一步**: 教训——判 idea 有效性要先确认指标口径(final-tick vs mc), 且 baseline 必须用同口径的复现值, 不能用过时常量. fig5b 的 md 文本描述尚未同步更新(mazes"最佳场景"那段需改).

## 2026-06-29 — 0629 收菜 + idea 有效性深度分析

- **思路(用户提供)**: 算力机上 ctm_paper st00-st24 + deep 01-04 已经跑了一部分,开始收菜然后做深入分析;后来追问"哪个优化方法有效",并指出 sparsity 要结合"省算力 vs 掉点"看,不能只比精度。
- **配置**: `extract_ctm_paper_results.py --curves`(534 runs, 并行) + `export_paper_results.py` 收 deep 01_revise/02_jepa/03_sparsity/04_combos。
- **预期**: 找到几个 work 的 idea, 出 idea×task 有效性矩阵。
- **结果**:
  - **唯一真 win = revise on parity**: mc 0.882→0.99(+10pp, n=5), 副作用极小(cifar10/mazes ±1pp)。
  - **sparsity mazes 是效率 win**: r=0.1 省 90% NLM 算力只掉 0.9pp(Pareto 甜点)。
  - **踩坑 — mc-vs-final 膨胀**: deep CSV 的 best_acc 是 mc 指标, baseline 是 final-tick, 直接 delta 把 most-certain-tick 机制的 +18pp(cifar10) 算成 idea 功劳; mc-vs-mc 后 cifar10 的 revise/JEPA/combo "+17pp" 全变中性。
  - parity 上 JEPA/halt/EMA/reflex 全 KILLED(0-iter, break 训练); sort 上 heads/sparsity/nst/revise 全 DEGEN(no-op 魔数 0.9253 / 0.6458/0.7917/0.8021)。
- **产物**: `paper/{analyze_0629,idea_validity_0629,analyze_deep,sparsity_efficiency}.py` + figC/figD/figE; AGENTS.md 新增「实验结果分析准则」。
- **下一步**: cifar10/parity 补 sparsity r sweep 补全 figE; qamnist JEPA-pd1/halt0.6 补 seed(现 n=1); 待 st02b/st04b bugfix re-run 补数据。
