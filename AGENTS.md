# CTM-LLM

## Setup
- **开发机** (spark-4562): `/home/wzk/projects/ctm_llm`
- **算力机** (nb-wangzekai-ctm-0): `/home/jovyan/h800fast/wangzekai/ctm_llm`, 8 × H100 (80GB)
- 仓库用作开发机与算力机之间的代码同步桥梁: 开发机修改 → `git push` → 算力机 `git pull`

## Key Context
- 模型: CTM-LLM (Continuous Thought Machines 架构移植为因果语言模型)
- Tokenizer: minimind-o 同源 (vocab_size=6400)
- 参考项目: 开发机 `/home/wzk/projects/minimind-o`, `/home/wzk/projects/continuous-thought-machines`
- 基础设施: SwanLab 日志, 单卡训练、8×H100 分布式均可

## worldmodel/ — 世界模型对比实验框架 (参考 stable-worldmodel 自研重写)
- **定位**: 参考 `galilai-group/stable-worldmodel` (MIT) 的设计**自己重写**的轻量世界模型研究框架, 用于把 CTM 作为 world-model baseline, 用 MPC solver (CEM) 评测, **对标 DINO-WM / LeWM (JEPA 系)**. **零新增依赖** (仅 torch + numpy, 不引 gymnasium/lancedb/ogbench), 算力机 `git pull` 即可跑, 无需装包/设代理.
- **目录**: `worldmodel/` (envs/data/solver/wm/policy/world/train) + `scripts/smoke_worldmodel.py` + `paper/run_worldmodel.py`
- **核心抽象**: world model 实现 `Costable.get_cost(info_dict, action_candidates)→(n_envs,n_samples)` 即可被 solver 规划; `WorldModel(encoder, predictor)` 是 **encoder-agnostic** 的 —— CTM-WM 与 JEPA-WM 只差 encoder (CTM 的 internal-tick synchronisation 表示 vs 小 CNN), 其余 (predictor/CEM/数据) 完全相同, 保证对比公平.
- **环境**: `PointStateReach` (向量, smoke 用) / `PointImageReach` (纯 torch 渲染的 2D 导航图像, 真实对比用). 都是 goal-conditioned, 自定义 Env/Space 协议 (非 gymnasium).
- **关键陷阱 — JEPA encoder 崩塌**: 从零训 JEPA latent 预测时, encoder 易崩塌为常数 (latent_var→0, cost 无梯度, CEM 退化为随机). `WorldModel` 内置 VICReg 方差正则 (`var_weight>0`) + 可选 EMA target encoder (`ema_decay>0`) 防崩塌. **CTM encoder 因深递归梯度流弱, 比 CNN 更易崩塌**, 是真实研究点. 诊断看 `latent_var` 列: ~0 = 崩塌.
- **Smoke**: `python scripts/smoke_worldmodel.py --env point-state --local` (state, 最快) / `--env point-image --var_weight 1.0` (image+CNN) / `--model ctm --env point-image --var_weight 1.0` (image+CTM).
- **对比实验**: `python paper/run_worldmodel.py --env point-image --episodes 60 --epochs 50 --var_weight 1.0` → 扫 CTM `iterations` × JEPA `latent_dim`, 写 `csv_data/worldmodel_results.csv` (success_rate/loss/latent_var). sweep 故意变 encoder 关键超参, 起两步走「功能验证」作用.
- **算力机后台**: `nohup python paper/run_worldmodel.py --env point-image --episodes 80 --epochs 80 --var_weight 1.0 > logs/worldmodel.log 2>&1 &` (无需代理, 无外部下载)

## 路径差异
- 开发机: `dataset_data` → symlink → `/home/wzk/projects/minimind-o/dataset/`
- 算力机: `dataset_data` → symlink → `/home/jovyan/h800fast/wangzekai/minimind-o/dataset/`
- `model_tokenizer` 同理, 指向各自路径下的 `minimind-o/model/`

## 注意事项
- **实验运行/分析一律用 `.py`/`.sh` 脚本, 默认不再用 notebook (`.ipynb`)**: notebook 难以在算力机后台 `nohup` 跑、cell 状态隐式难复现、两步走验证不便、且 `.ipynb` 的 JSON diff 噪音大不利 git 同步. 范例见 `paper/run_02_jepa.py`、`paper/run_04_combos.py` (各自取代同名 deep notebook); 分析也写成独立脚本, 用 `exp_runner.collect_csv` 读 `csv_data/*.csv` 出图, 不依赖 cell 执行顺序. 新需求优先新增 `run_*.py`/`export_*.py`, 不要再新建 notebook.
- **算力机启动 server/worker 前必须设代理** (否则 huggingface/datasets 下载会卡):
  ```bash
  export http_proxy="http://public-proxy.qihoo.net:3128"
  export https_proxy="http://public-proxy.qihoo.net:3128"
  ```
- `dataset_data` 和 `model_tokenizer` 是符号链接, 已在 `.gitignore` 中排除, 每台机器需手动创建
- 训练数据: `sft_t2a_mini.parquet` (515k 条英文对话), 按 `DATA_DOWNLOAD.md` 下载
- 检查点: `out/ctm_llm_{hidden_size}.pth` (half) + `_resume.pth` (含 optimizer state)

## Pool 并发工程准则
- **日志路径必须 per-experiment 隔离**, 禁止多任务共享同一日志文件 (会被并发覆盖). 用 `CTM_LOG_DIR` + `{exp_name}.log` 模式.
- **失败诊断要有 fallback 链**: `.fail.json` → per-experiment `.log` → `pool_last_run.log`, `.fail.json` 中存 `log_path` 便于定位.
- **`cluster_pool.py` 改动必须重启 server**, worker 通常 auto-pull 无需手动干预.
- **数据路径不要硬编码** (如 MNIST 的 `"data/"`), 应通过参数/环境变量传入, 避免换环境踩坑.
- **GPU slot 分配用 `node:gpu` 格式** (如 `ip:0`), bare IP 会导致 `gpu_sets_overlap` 阻塞并行.
- **task ID 用微秒+单调序号**, 避免快速批量 submit 时碰撞.
- **torchrun entry point 必须是 Python 文件**, 不能是 shell 脚本.
- **pool submit payload 的 `env` 字段会透传给 worker 子进程**, 用于传 `CTM_EXPERIMENT_NAME` 等上下文.
- **批量 submit 必须加 `--no-wait`**: `experiment_plan*.py submit` 默认 `--wait` 会在每个任务完成后才提交下一个, 1057 个任务要等几天. 加 `--no-wait` 一口气全部入队, workers 自动并行消费. 同理, `cluster_pool.py submit` 的 `--wait` 默认 0 (不等待).
- **GPU slots 自动计算**: `--gpu-slots 0` (默认) 会根据显存和 d_model 自动算并发数. 80GB H100 + d_model=512 ≈ 16 slots/卡, d_model=1024 ≈ 8 slots/卡. 也可手动指定 `--gpu-slots N` 覆盖.
- **启动 pool**: server 只需 `--port`, worker 只需 `--master-addr` (不再需要 `--config` 集群文件). Worker 自动注册到 server 并被发现.

### Pool 启动命令
```bash
# 算力机必须先设代理, 否则 huggingface/datasets 下载会卡住:
export http_proxy="http://public-proxy.qihoo.net:3128"
export https_proxy="http://public-proxy.qihoo.net:3128"

# 任选一台起 server
python scripts/cluster_pool.py server --port 8765

# 所有节点起 worker (自动连 server)
python scripts/cluster_pool.py worker --master-addr 11.131.210.78 --port 8765

# 提交任务
python scripts/experiment_plan_ctm_paper.py submit --stage all --no-wait
```

### Smoke 测试流程
1. **本地先跑** (快速看错误): `python scripts/smoke_baseline.py --iterations 10 --local`
2. **再跑 pool** (测集群基建): `python scripts/smoke_baseline.py --iterations 10`
3. 两轮都过了再正式 submit 实验计划

## 实验验证两步走: smoke + 功能验证 (强制)
任何 idea/特性的实验, **两步都过才算正规实验**, 缺一步会导致"跑了等于没跑"——整批结果无效却看不出来.
- **Step 1 — Smoke 验证 (能跑通)**: 少量 iter, 不崩、shape/loss 正常 (见上).
- **Step 2 — 功能验证 (有效果)**: 证明该特性**确实改变了模型行为**, 而不是参数被静默丢弃. 固定 seed, 跑两组「关键超参取不同值」(如 draft-revise 的 `corrupt_prob=0.05 vs 0.3`, sparsity 的 `topk=0.25 vs 0.75`), 对比训练曲线/best_acc:
  - **应不同的两组若逐点相同 → idea 没接入, 立即排查, 绝不批量跑.**
  - 代码侧根因排查: CTM forward 用 `getattr(self, '<idea>_mode', 默认值)` 读特性开关, 各 task `train.py` 的 "Set idea attributes on model" 块**必须把 `args.*` 拷到 model**; 漏赋值 → forward 恒取默认值 → 特性 inert.
  - **判据**: 同 seed、不同关键超参 → 结果必须不同. 不要拿"vs 旧 `BASELINE_ACC` 常量的 delta"当效果证据 (常量会过时, 非确定性噪声会造假 delta).
- **历史教训**: `01_revise` 119 个 run 全是 baseline——4 个 task `train.py` 漏了 `model.draft_mode` 等 4 行赋值, draft-revise 从未生效. 只因 sort 训练可复现 (同 seed 逐位相同) 才暴露, cifar10/mazes/parity 靠非确定性噪声伪装出假 delta, 浪费大量卡时.

### 算力机调试注意
- **`.fail.json` 和 per-experiment `.log` 文件在算力机上生成**, 开发机无法直接访问. 需要用户从算力机手动复制到开发机 (或粘贴内容) 才能诊断失败原因.
- 调试失败任务时, 请用户提供: `runs/metrics/{exp_name}.fail.json` 内容 + 对应的 `logs/{exp_name}.log` 末尾 traceback.

## 实验记录规范 (强制)
每批实验必须有据可查, 方便日后回溯"为什么跑、当时怎么想的、结论是什么". 全部记录在 **`EXPERIMENT_LOG.md`**(根目录, 倒序追加, 最新在最上).
- **何时记**: 提交实验前先写一段(至少思路+配置+预期), 收菜/分析后回填结果+结论. 不要等忘光了再补.
- **必填字段**(缺一不可):
  1. **日期** `YYYY-MM-DD`
  2. **思路(用户提供)**: 一句话写清楚这次实验的**动机/来源**, 尤其是用户口头给的 idea 或方向. 这是日后看懂"为什么有这批实验"的关键, 必须记原话意图, 不要只记技术细节.
  3. **配置**: stage 名 / 关键超参 / 跑在哪(`logs/...` 路径).
  4. **预期**: 期望看到什么现象.
  5. **结果**: best_acc / delta / 口径(mc-vs-mc 还是 final) / 样本数.
  6. **结论**: idea 是否有效, 口径是否需修正(如 sparsity 看 Pareto 而非单精度).
  7. **下一步**: 是否有 follow-up / 待补 seed / 待修 wiring.
- **和已有文件分工**: `EXPERIMENT_PLAN.md` = 将来要跑的计划(matrix size/命令); `EXPERIMENT_LOG.md` = 跑过的日志(日期+思路+结论); `VERIFIED_CONCLUSIONS.md` = 已验证有效、可写论文的结论(JEPA/revise/sparsity 三块, 含口径+边界). 两者互补, 不要混.
- 范例见 `EXPERIMENT_LOG.md` 首条(0629 收菜分析).

## 实验结果分析准则 (强制)
判 idea 是否有效, 单看"精度 delta vs baseline"经常**误导**, 必须按 idea 性质选对口径. 已踩的坑:

### 坑 1 — mc-vs-final 指标膨胀 (ctm_paper deep 三件套)
- `export_paper_results.py` 导出的 deep CSV: `best_acc` 取的是 **`test_accuracies_most_certain`** (per-sample 最优 tick), 但 `baseline` 列是 **final-tick** paper baseline. 直接算 delta 会把 most-certain-tick 机制本身的 +18pp(cifar10)/+20pp(parity) 算成 idea 的功劳 → 假 win.
- **正确口径**: mc-vs-mc, 用 st00 的 `best_test_acc_mc` 当 baseline (cifar10=0.8516 / parity=0.8821 / mazes=0.9117; sort 无 mc, 用 element-level full_list 0.8753).
- 实测: 用 mc-vs-mc 后, deep 里 cifar10 的 revise/JEPA/combo "+17pp win" 全变中性 (-0.9pp)——那只是把 final-tick 往 mc 天花板推, 没抬动天花板本身. **唯一真 win 是 revise on parity (mc 0.882→0.99, +10pp, n=5)**.
- 脚本: `paper/analyze_deep.py` 内置 `MC_BASELINE` / `FINAL_BASELINE` 两套, 同时报 `delta_mc`(公平) 和 `delta_inflated`(膨胀假象, 灰幽灵柱) 防再踩.

### 坑 2 — 效率型 idea 必须看 Pareto, 不能只看精度
- **sparsity (top-k 稀疏激活) 的目的不是涨点, 是省算力换可接受掉点**. 只看精度 delta 会判成"NEUTRAL/NEGATIVE"而错过真正的 win.
- **正确口径**: 精度 vs 有效算力(r = 激活神经元比例)的 Pareto 前沿. 算力模型: NLM 思考回路做 ~r 的功, 省 (1-r); backbone(resnet) 不稀疏化, 端到端加速 < (1-r), 需 sparse kernel 才变现.
- 实测 (mazes, 唯一有 r∈{0.1..0.9} 完整 sweep 的任务): **r=0.1 省 90% NLM 算力只掉 0.9pp**——教科书级效率 trade-off. r=0.25 省 75% 掉 0.5pp. 前沿点 {0.1, 0.25}. 而 sort 上稀疏是大坑 (r=0.5 掉 12pp, sort 需全量神经元).
- 脚本: `paper/sparsity_efficiency.py` 出 `figE_sparsity_pareto.png` (x=算力比例 r, y=精度, 左下=又便宜又准=好).
- **通用规则**: 任何"以省资源为目标"的 idea (稀疏/量化/蒸馏/早停 halt) 都必须报 Pareto (质量 vs 资源), 不能只报单一精度. halt 的"省 tick 数"也是同类——应看 精度 vs 平均思考步数 的前沿, 而不是只比 final acc.

### 正常数据筛选口径 (写死在分析脚本里)
判 idea 有效性前, 先按口径过滤 run, 不能拿坏数据出结论:
- **KILLED** (`final_iter==0`): idea 让训练没起步 (parity 上 JEPA/halt/EMA/reflex 全这样), 不是"效果差"是"break 训练", 排除并单独标.
- **PARTIAL** (`final_iter < 0.5*planned`): 负 delta 不可信, 排除 (parity revise 130-160k/200k 算 NORMAL, mc 已近顶可留).
- **BUG**: 已知 plan bug 污染的 run 按名排除 (st02 sort tick1-25 的 memory_length 耦合; st04 sort 的 weight=1.0 混淆), 见 `scripts/fix_bug_st02_*.py` / `fix_bug_st04_*.py` docstring.
- **DEGEN** (no-op): 同一种子签名跨 ≥2 stage 复现 = task 忽略该 arg (sort 上 heads/sparsity/nst/revise 全退到 0.9253 或 0.6458/0.7917/0.8021 魔数), 标 inert 排除, 不是真"无效"是"没接上".
- **CRASHED** (`final_iter < 0.1*planned`, 如 04_combos 的 jepa0p1+spar0p5 只跑 2k): 排除.
- 脚本: `paper/idea_validity_0629.py` (ctm_paper st00-st24) + `paper/analyze_deep.py` (deep 01-04) 各自实现这套口径.

### 收菜/分析流水线
1. **算力机收菜**: `python scripts/extract_ctm_paper_results.py --curves` (扫 `logs/ctm_paper/st*`, 并行 `--workers 0`=min(cpu,16)) → `runs/metrics/ctm_paper_{summary.csv,summary.md,curves.json}`; deep 用 `python scripts/export_paper_results.py --log-root logs/deep/01_revise --output csv_data/01_revise_results.csv` 逐个收.
2. **dev 机出图**: `python scripts/plot_ctm_paper_results.py` (fig1-20) + `paper/analyze_0629.py` (figA/figB 深度) + `paper/idea_validity_0629.py` (figC 有效性矩阵) + `paper/analyze_deep.py` (figD deep mc-vs-mc) + `paper/sparsity_efficiency.py` (figE Pareto).
3. **三件套就位**: `csv_data/ctm_paper_{summary.csv,summary.md,curves.json}` 必须放规范名 (无日期后缀), plot 脚本默认读它; 带日期的 `_0629` 等是备份.
