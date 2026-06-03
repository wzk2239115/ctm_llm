# 数据下载

CTM-LLM 目前只使用文本对话数据（语音/图像列忽略），数据来自 minimind-o 项目。

## 下载源

- **HuggingFace Mirror**: [jingyaogong/minimind-o_dataset](https://hf-mirror.com/datasets/jingyaogong/minimind-o_dataset)

## 所需文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `sft_t2a_mini.parquet` | ~1.5GB | 515k 条英文对话（快速迭代用） |

放入 `dataset/`（开发机上 `dataset_data/` 符号链接指向的目录）。

## 算力机操作

开发机 `/home/wzk/projects/minimind-o/dataset/` 下已存在，算力机需创建同路径目录并下载，或从开发机拷贝。

```bash
export http_proxy="http://public-proxy.qihoo.net:3128"
export https_proxy="http://public-proxy.qihoo.net:3128"

mkdir -p /home/wzk/projects/minimind-o/dataset

# 下载（仅 mini 文件即可）
HF_ENDPOINT=https://hf-mirror.com hf download jingyaogong/minimind-o_dataset \
  --local-dir /home/wzk/projects/minimind-o/dataset \
  --repo-type dataset \
  --include "sft_t2a_mini.parquet"
```
