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
