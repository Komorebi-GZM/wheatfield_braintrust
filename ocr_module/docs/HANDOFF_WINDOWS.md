# Windows 端交接 Prompt · Xiaomi-OCR-0 手写笔记 OCR 微调

> 用法：把下方 **「=== 交接 Prompt 开始 ===」到「=== 结束 ===」之间整段** 复制给 Windows 上的编码 Agent（或按步骤人工执行）。

---

=== 交接 Prompt 开始 ===

你在一台 **Windows 机器（RTX 4060 8GB）** 上工作。任务：把仓库 `wheatfield_braintrust` 的 OCR 子模块跑通，并完成 **Xiaomi-OCR-0（0.8B）4-bit QLoRA 微调**，把手写课程笔记照片转成**保留结构与公式**的结构化 Markdown（标题层级 / 分点列表 / 表格 / LaTeX 公式）。

**先读完 `ocr_module/docs/OCR模型设计说明书.md` 再动手。** 严格按阶段执行，**第 1–3 步完成后先回报环境情况，再继续训练**（训练 15h+，确认无误再启动）。

## 0. 基本信息
- 仓库：`https://github.com/Komorebi-GZM/wheatfield_braintrust.git`
- 分支：`feat/ocr-xiaomi-qlora`
- 模块目录：`ocr_module/`（推理/数据/训练/评测）；集成改动在 `maitian_agent/`（`tools/ocr.py` 新增 `ocr_type="xiaomi"`）
- 基座：`SeerRay-Lab/Xiaomi-OCR-0`（0.8B）。**若 ms-swift 不支持 Qwen3.5 多模态架构，回退 `Qwen/Qwen2.5-VL-3B-Instruct` 并说明差异**
- 训练预算：15h+ 属正常（可后台/分批）

## 1. 环境与硬件核查（先做，务必回报）
```bat
nvidia-smi
python --version
```
- 推荐 **WSL2 + Ubuntu 22.04 + CUDA 12.1**；原生 Windows 需预编译 `bitsandbytes` wheel（否则 QLoRA 4-bit 不可用）。
- 记录：GPU 型号 / 显存 / 驱动 / CUDA / 可用内存 / 磁盘。

## 2. 获取代码
```bash
git clone -b feat/ocr-xiaomi-qlora https://github.com/Komorebi-GZM/wheatfield_braintrust.git
cd wheatfield_braintrust
```

## 3. 建环境装依赖
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    WSL2: source .venv/bin/activate
pip install --upgrade pip
# 按 CUDA 版本装 torch（示例 cu121）
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft "ms-swift>=2.5" pillow matplotlib pylatexenc
# WSL2 下 bitsandbytes 直接 pip install bitsandbytes；原生 Windows 用预编译 wheel
cd ocr_module && pip install -r requirements.txt && cd ..
```

## 4. 放入数据（20–30 页手写课程笔记）
- 图片放 `ocr_module/data/raw/`
- 标注 `ocr_module/data/annotations/real.jsonl`，每行：
  `{"image":"page01.jpg","markdown":"# 标题\n- 要点\n$$ \\frac{a}{b} $$\n| a | b |\n|---|---|"}`
  规则：整页级；保留标题/列表/表格/公式；**不可辨认用 `□`，不要臆造**；插图用 `> [图示：…]` 占位。
- 若暂无标注：先只做「合成 + 基座推理验证」，标注可后补（但微调效果依赖真实标注）。

## 5. 合成数据 + 划分
```bash
cd ocr_module
python scripts/build_synth.py --count 800 --out data/synth --font <中文手写字体.ttf>
python scripts/prepare_dataset.py --real-dir data/raw --real-ann data/annotations/real.jsonl \
  --synth-manifest data/synth/manifest.jsonl --out-dir data/splits --val-pages 5 --test-pages 5
```
确认：`val.jsonl`/`test.jsonl` **只含真实页**，`train_swift.jsonl` 含真实+合成。

## 6. 基座冒烟（训练前必做）
```bash
python scripts/infer_one.py data/raw/page01.jpg --show-markdown
```
记录：能否加载、显存占用、单页耗时、输出里标题/列表/表格/公式是否合理。**这一步失败先别训练。**

## 7. QLoRA 训练
```bash
bash scripts/train_qlora.sh        # 参数见 configs/train_qlora.yaml
```
- 监控显存与 loss；**OOM 处理顺序**：`max_pixels` 1024000→589824 → `lora_rank` 8→4 → 保持 `freeze_vit=true`。
- 记录最终 adapter 目录（默认 `outputs/qlora-v1`，仅几十 MB）、总耗时、loss 曲线。

## 8. 评测
批量预测测试集到 `outputs/pred/test_pred.jsonl`（每条 `{"image":..., "markdown":...}`），然后：
```bash
python scripts/evaluate.py --pred outputs/pred/test_pred.jsonl --ref data/splits/test.jsonl --out outputs/eval
```
对照：**基座 zero-shot vs 微调后**；报告 文本 CER / 公式 exact-match+CER / 结构（标题序列·列表·表格）匹配率，并附错误样例（公式、表格、层级）。

## 9. 必须回报的内容
1. 环境：GPU/显存/驱动/CUDA/Python/关键包版本
2. 基座推理：能否加载、显存、单页耗时、输出质量初判
3. 训练：真实/合成数据量、超参、总耗时、loss 曲线、adapter 路径
4. 评测：三组指标对照表 + 每类错误样例
5. 问题与建议（架构不支持 / OOM / 数据不足等）

## 10. 约束
- **不要把权重/adapter/数据提交到普通 git**；adapter 用 git-lfs 或 HF 托管；用户私有笔记**不得上传公网**。
- 失败要如实报告卡在哪一步，不要假装完成；未验证前不要声称"已跑通"。

=== 结束 ===
