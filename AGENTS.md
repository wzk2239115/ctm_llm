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

## 路径差异
- 开发机: `dataset_data` → symlink → `/home/wzk/projects/minimind-o/dataset/`
- 算力机: `dataset_data` → symlink → `/home/jovyan/h800fast/wangzekai/minimind-o/dataset/`
- `model_tokenizer` 同理, 指向各自路径下的 `minimind-o/model/`

## 注意事项
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
