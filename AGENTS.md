# CTM-LLM

## Setup
- 开发机: 当前机器 (spark-4562)
- 算力机: 8 × H100 (80GB), `git pull` 同步代码
- 仓库用作开发机与算力机之间的代码同步桥梁

## Key Context
- 模型: CTM-LLM (Continuous Thought Machines 架构移植为因果语言模型)
- Tokenizer: minimind-o 同源 (vocab_size=6400)
- 参考项目: `/home/wzk/projects/minimind-o` (LLM 框架) + `/home/wzk/projects/continuous-thought-machines` (CTM 架构参考)
- 基础设施: SwanLab 日志, 单卡训练、8×H100 分布式均可

## 注意事项
- `dataset_data` 和 `model_tokenizer` 是指向 minimind-o 的符号链接, 算力机需创建同路径链接
- 训练数据: `sft_t2a_mini.parquet` (515k 条英文对话)
- 检查点: `out/ctm_llm_{hidden_size}.pth` (half) + `_resume.pth` (含 optimizer state)
