# OCR 子模块 · 手写课程笔记结构化（Xiaomi-OCR-0 QLoRA）

麦田智囊（wheatfield_braintrust）的 **OCR-only 模块**：把手写课程笔记照片转写为
**保留结构与公式**的结构化 Markdown（标题层级 / 分点列表 / 表格 / LaTeX 公式）。

> 设计依据见仓库根目录 `OCR模型设计说明书.md`。本仓库只做 OCR：推理、微调、评测、数据管线。

## 目录

```
ocr/             # 推理与后处理（对外主入口 ocr.engine.recognize_page）
scripts/         # 合成数据、数据集准备、训练、评测、单图推理
configs/         # QLoRA 训练 / 推理配置
tests/           # 指标与后处理单测
data/            # raw(真实页) / annotations / synth / splits   (Git 忽略)
outputs/         # 训练产物、评测结果                          (Git 忽略)
```

## 环境

- 训练：**Windows RTX 4060 8GB**（推荐 WSL2 + CUDA；约 15h+），或云端 GPU
- 推理/评测：Mac M2 16GB 可跑（不训练）
- Python 3.10+

```bash
pip install -r requirements.txt
```

> 说明：原生 Windows 上 `bitsandbytes` 不稳定，建议 WSL2 或使用预编译 wheel；
> `ms-swift` 若暂不支持 Qwen3.5 多模态架构，先用 `Qwen/Qwen2.5-VL-3B-Instruct` 做对照基线。

## 快速开始

1. **准备真实数据**（20–30 页手写笔记）
   - 图片放 `data/raw/`
   - 标注 `data/annotations/real.jsonl`，每行 `{"image":"xxx.jpg","markdown":"..."}`（整页级，见设计说明书 §3.2）

2. **生成合成数据**（弱监督，只进训练）
   ```bash
   python scripts/build_synth.py --count 800 --out data/synth --font /path/to/CJK-handwriting.ttf
   ```

3. **划分 + 导出训练格式**
   ```bash
   python scripts/prepare_dataset.py \
     --real-dir data/raw --real-ann data/annotations/real.jsonl \
     --synth-manifest data/synth/manifest.jsonl --out-dir data/splits \
     --val-pages 5 --test-pages 5
   ```
   - 产出 `train_swift.jsonl`（ms-swift）、`train_sharegpt.json`（LLaMA-Factory）、`val.jsonl`、`test.jsonl`
   - **合成页只进 train；val/test 只用真实页**

4. **QLoRA 微调（4060 / WSL2）**
   ```bash
   bash scripts/train_qlora.sh          # 参数见 configs/train_qlora.yaml
   ```

5. **评测**
   ```bash
   # 先用引擎生成预测（示例）
   python scripts/infer_one.py data/raw/page1.jpg --adapter outputs/qlora-v1 --show-markdown
   # 批量预测后：
   python scripts/evaluate.py --pred outputs/pred/test_pred.jsonl \
     --ref data/splits/test.jsonl --out outputs/eval
   ```
   指标：文本 CER、公式归一化 exact-match / CER、结构（标题序列/列表/表格）匹配。

6. **推理**
   ```python
   from ocr import recognize_page
   r = recognize_page("data/raw/page1.jpg", adapter_dir="outputs/qlora-v1")
   print(r["markdown"], r["structure_ok"], r["latex_issues"])
   ```

## 单测

```bash
pytest -q          # 或  python tests/test_metrics.py
```

## 远程 branch 试验

- 分支：`feat/ocr-xiaomi-qlora`
- 入库：代码 + 配置 + 评测报告；**LoRA adapter 用 git-lfs 或 HF 托管，不入普通 git**
