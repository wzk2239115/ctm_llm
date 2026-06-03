# 数据下载

CTM-LLM 目前只使用文本对话数据，来自 minimind-o 项目。

## 下载

```bash
export http_proxy="http://public-proxy.qihoo.net:3128"
export https_proxy="http://public-proxy.qihoo.net:3128"

# ----- 训练数据 -----
mkdir -p /home/jovyan/h800fast/wangzekai/minimind-o/dataset
HF_ENDPOINT=https://hf-mirror.com hf download jingyaogong/minimind-o_dataset \
  --local-dir /home/jovyan/h800fast/wangzekai/minimind-o/dataset \
  --repo-type dataset \
  --include "sft_t2a_mini.parquet"

ln -sf /home/jovyan/h800fast/wangzekai/minimind-o/dataset /home/jovyan/h800fast/wangzekai/ctm_llm/dataset_data

# ----- Tokenizer (来自 minimind-3o 模型) -----
mkdir -p /home/jovyan/h800fast/wangzekai/minimind-o/model
HF_ENDPOINT=https://hf-mirror.com hf download jingyaogong/minimind-3o \
  --local-dir /home/jovyan/h800fast/wangzekai/minimind-o/model \
  --include "tokenizer*"

ln -sf /home/jovyan/h800fast/wangzekai/minimind-o/model /home/jovyan/h800fast/wangzekai/ctm_llm/model_tokenizer
```
