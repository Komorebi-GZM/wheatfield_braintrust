# OCR 子模块 · 手写课程笔记结构化（Xiaomi-OCR-0 QLoRA）

麦田智囊（wheatfield_braintrust）的 **OCR-only 模块**：把手写课程笔记照片转写为
**保留结构与公式**的结构化 Markdown（标题层级 / 分点列表 / 表格 / LaTeX 公式）。

> 设计依据见仓库根目录 `OCR模型设计说明书.md`。本仓库只做 OCR：推理、微调、评测、数据管线。

## 当前状态（2026-11-15）

- 数据：23 张真实手写页，已逐页对照原图复核（v2 标注），固定划分 **train 13 / val 5 / test 5**。
- 训练：4-bit QLoRA（r=8、α=32、仅语言模型 372 模块）已完成，最佳 checkpoint = `checkpoint-26`（验证集 eval_loss 最低）。
- 评测：基座 vs 微调、同一参考集/同一尺度/同一提示词/同一评估器，5 页全跑。
- 结论：**混合变化**。正文宏 CER 1.4872 → 0.7725，但把公式体计入后 1.1217 → 1.7952；可评分页只有 3 页，失败率双方同为 66.7%。**不能宣称整体 OCR 能力已改善**。
- 完整细节：`HANDOFF.md`（复现说明）、`final_comparison_report_v2.md`（验收报告）、`CHECKPOINT_MANIFEST.md`（权重交付清单）。

## 目录

```
ocr/                     # 推理与后处理（对外主入口 ocr.engine.recognize_page）
scripts/                 # 数据准备、训练、批量推理、评测、对比、诊断
configs/                 # 训练 / 推理配置模板（实际参数以 run 的 args.json 为准）
tests/                   # 后处理、消息格式、评测排除规则单测
datasets/handwriting_notes/  # 23 张原图 + 标注 + 固定划分 + 版本哈希（入库）
data/                    # 早期临时数据与远程标注对照（Git 忽略）
outputs/                 # 训练产物、预测、指标；仅小型文字产物入库，权重与图片忽略
```

## 环境

- 训练：**Windows 11 + RTX 4060 Laptop 8GB**（已实测跑通，52 步约 4 分钟）；Linux/WSL2 亦可
- 推理/评测：Windows / macOS 均可，需要能加载 4-bit 的 GPU
- Python 3.12、torch 2.6.0+cu124、transformers 5.14.1、ms-swift 4.6.0、bitsandbytes 0.50.2、peft 0.20.0

```bash
pip install -r requirements.txt
```

> 原生 Windows 需要与 CUDA 匹配的 `bitsandbytes>=0.50.2` wheel（本项目即在原生 Windows 上训练成功）。
> `ms-swift` 需 `>=4.6` 才认识 `qwen3_5` 的 model_type/template。

## 数据

```bash
# 校验版本与划分（13/5/5、互斥、覆盖 23 页）
python scripts/data_version.py
```

- 标注：`datasets/handwriting_notes/annotation_template.jsonl`（含 `□` 表示不可辨）
- 固定划分：`datasets/handwriting_notes/splits/`（`train/val/test.jsonl` + Swift 格式）
- 版本与哈希：`datasets/handwriting_notes/data_version_v2.json`
- **不要重新划分**：改动划分会让所有历史指标失效；合成数据只允许进 train。

## 训练

```bash
powershell -File scripts/train_qlora_v2.ps1 -OutputRoot outputs/qlora-v2
```

参数与选择依据见 `HANDOFF.md` 第 4 节。Windows 上不要依赖 `last-checkpoint` 符号链接，用真实 `checkpoint-N` 目录。

## 推理与评测

```bash
# 批量预测（基座）
python scripts/infer_test_baseline.py \
  --manifest datasets/handwriting_notes/splits/test.jsonl \
  --output outputs/eval_v3/baseline/test_pred_v3.jsonl \
  --max-pixels 1024000 --max-new-tokens 1024

# 批量预测（微调）
python scripts/infer_test_baseline.py \
  --manifest datasets/handwriting_notes/splits/test.jsonl \
  --output outputs/eval_v3/finetuned/test_pred_v3.jsonl \
  --adapter outputs/qlora-v2/v2data-20261115-134816/checkpoint-26 \
  --max-pixels 1024000 --max-new-tokens 1024

# 指标（含 □ 的参考页整页排除，双方同规则）
python scripts/evaluate.py --pred <pred.jsonl> --ref datasets/handwriting_notes/splits/test.jsonl --out <dir>
```

单图推理：

```python
from ocr import recognize_page
r = recognize_page("datasets/handwriting_notes/images/IMG_20261105_000037.jpg",
                   adapter_dir="outputs/qlora-v2/v2data-20261115-134816/checkpoint-26")
print(r["markdown"])
```

指标：正文 CER（公式文本不计）、含公式体 CER、公式 EM/CER、结构（标题序列/列表/表格）、
触顶与重复检测。**报告 CER 时必须说明是否包含公式体，以及可评分页数。**

## 单测

```bash
pytest tests -q -p no:cacheprovider --basetemp=.pytest_tmp
```

## 分支与权重

- 分支：`feat/ocr-xiaomi-qlora`
- 入库范围：代码 + 配置 + 脚本 + 测试 + 标注/划分 + 协议 + 报告 + 小型预测与指标文件
- **LoRA adapter 不入普通 Git**（21.7 MB 权重 + 校验值见 `CHECKPOINT_MANIFEST.md`），需按该文件的方案另行托管
- 原图为个人课程笔记，仓库为公开仓库，使用前请确认隐私边界（见 `HANDOFF.md` 第 10 节）
