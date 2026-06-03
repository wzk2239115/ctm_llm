# 数据下载

CTM-LLM 目前只使用文本对话数据（语音/图像列忽略），数据来自 minimind-o 项目。

## 下载源

- **HuggingFace**: [jingyaogong/minimind-o_dataset](https://huggingface.co/datasets/jingyaogong/minimind-o_dataset)
- **ModelScope**: [gongjy/minimind-o_dataset](https://www.modelscope.cn/datasets/gongjy/minimind-o_dataset)

## 所需文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `sft_t2a_mini.parquet` | ~1.5GB | 515k 条英文对话（快速迭代用） |

放入 `dataset/`（即 `dataset_data/` 符号链接指向的目录）。

## 算力机操作

```bash
# 开发机上已下载，算力机需手动下载或从开发机拷贝
# 方式一: HuggingFace
wget https://huggingface.co/datasets/jingyaogong/minimind-o_dataset/resolve/main/sft_t2a_mini.parquet

# 方式二: 直接拷贝
rsync -avP 开发机:~/projects/minimind-o/dataset/sft_t2a_mini.parquet /path/to/minimind-o/dataset/
```
