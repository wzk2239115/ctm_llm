# 数据下载

CTM-LLM 目前只使用文本对话数据，来自 minimind-o 项目。

## 下载

```bash
export http_proxy="http://public-proxy.qihoo.net:3128"
export https_proxy="http://public-proxy.qihoo.net:3128"

# 创建数据目录并下载
mkdir -p /home/jovyan/h800fast/wangzekai/minimind-o/dataset
HF_ENDPOINT=https://hf-mirror.com hf download jingyaogong/minimind-o_dataset \
  --local-dir /home/jovyan/h800fast/wangzekai/minimind-o/dataset \
  --repo-type dataset \
  --include "sft_t2a_mini.parquet"

# 创建符号链接 (仓库 .gitignore 已排除 dataset_data)
cd /home/jovyan/h800fast/wangzekai/ctm_llm
ln -sf /home/jovyan/h800fast/wangzekai/minimind-o/dataset dataset_data
```
