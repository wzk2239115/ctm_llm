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
