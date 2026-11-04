# 手写教案 OCR 模型微调：可行性方案与实验计划

> 目标项目：`Komorebi-GZM/wheatfield_braintrust`（麦田智囊 · 乡村教育 Agent 系统）
> 评估对象：以 Xiaomi-OCR-0 为代表的小模型方案 vs. 专用 OCR 识别模型方案
> 交付物：可行性判断 + 推荐技术路线 + 可执行实验计划（本次不训练）
> 算力约束：本机 Mac M2 16GB；Windows RTX 4060 **8GB**（24G 内存 / 1.5T 盘）

---

## 0. 结论摘要（TL;DR）

1. **本项目的 OCR 落点是「手写教案（中文手写体）→ 文本 → LLM 结构化」，不是通用文档解析。** 当前实现用 `EasyOCR(ch_sim+en, gpu=False)`，而项目原始设计规格（`.trae/documents/麦田智囊-任务框架总览.md`）要求的是 **PaddleOCR**，并给出了 **手写识别准确率 ≥90%（自述已验证 92%）、≤5 秒/张** 的硬指标。这是本次微调的真正目标。
2. **Xiaomi-OCR-0 是「推理/部署」仓库，不含任何训练代码**；它是一个 0.8B 的 OCR 中心 VLM（基座 Qwen3.5-0.8B，权重 `SeerRay-Lab/Xiaomi-OCR-0`）。要用它微调，必须另起训练框架（ms-swift / LLaMA-Factory / PEFT），并自行处理 Qwen3.5 多模态架构兼容性。
3. **可行性判断：可行，但「微调 0.8B VLM」不是首选。** 8GB 显存下 0.8B VLM 只能做 4-bit QLoRA，且全页推理很可能**突破 ≤5 秒/张**；而本项目只需要「文字行识别」，专用轻量识别模型（PaddleOCR PP-OCRv5 rec / SVTR）**更小、更快、更省显存，8GB 4060 跑全参微调都绰绰有余**，与本项目现有「OCR→LLM 结构化」的架构天然契合。
4. **推荐路线（分三步走）**：
   - **第一步（0 成本快赢）**：先把 EasyOCR 换成 **PaddleOCR 预训练模型**，在你的真实手写教案测试集上做 baseline。大概率已能显著超过 EasyOCR，可能直接达标。
   - **第二步（主攻）**：用 **CASIA-HWDB / SCUT-HCCDoc 等公开手写数据 + 你的少量本地标注** 对 **PP-OCRv5 识别模型做微调**（域自适应），目标 CER 下降、行准确率 ≥90%、单张 ≤5s。
   - **第三步（可选增强）**：若第二步仍不达标、或需要「整页→结构化」端到端，再评估 **0.8B VLM（Xiaomi-OCR-0）QLoRA** 作为实验分支，而非默认方案。

---

## 1. 背景与现状

### 1.1 项目中的 OCR 调用链路（已核对源码）

| 位置 | 现状 |
|---|---|
| `maitian_agent/agents/wisdom_transfer.py` | `WisdomTransferAgent`：`Reader(['ch_sim','en'], gpu=False)` → `readtext(detail=1)` 逐行取文本 → 拼成 `ocr_text` → LLM 按固定字段（学科/年级/课题/教学目标/重难点/教学过程/作业/反思）结构化 |
| `maitian_agent/tools/ocr.py` | 已有 `OCRProcessor` 工厂：支持 `easyocr` / `paddleocr` 两种实现（`PaddleOCRProcessor` 已写，默认 `ocr_type="easyocr"`） |
| `maitian_agent/config/settings.py` | `ocr_type="easyocr"`, `ocr_languages=["ch_sim","en"]` |
| `demo/agents/wisdom_transfer.py` | Demo 版同样直接用 EasyOCR |
| `.trae/.../任务框架总览.md` | 设计规格写明 OCR 用 **PaddleOCR**；指标：手写体识别准确率 **≥90%**（自述 92%）、智慧传承 **≤5 秒/张** |
| `docs/麦田智囊-项目完善与开发计划.md` | 有专门 Task：把 `OCRProcessor` 接入 `WisdomTransferAgent`，移除裸 `from easyocr import Reader` |

**结论：** 项目已经在代码层预留了「可切换 OCR 引擎」的抽象（`OCRProcessor`），所以替换/微调一个 OCR 模型是**低侵入**的——这正是微调落地的切入点。

### 1.2 为什么需要微调（痛点）

- **EasyOCR 对中文手写体识别能力弱**：EasyOCR 的中文模型主要面向印刷/场景文字，对连笔、行草、非规整排版的教案手写稿 CER 很高，直接导致下游结构化字段错乱。
- 教案是**手写 + 表格/分点 + 学科术语**混合体，通用模型域不匹配。
- 项目要的是**准确率≥90% + 速度≤5s/张**，二者需要用「小而专」的识别模型来同时满足。

### 1.3 目标与非目标

**目标**
- 提升中文手写教案的**字符级准确率**与下游**结构化字段准确率**；
- 单页/单图识别时延 **≤5 秒**（4060 8GB 上）；
- 交付**可切换、可回退**的 OCR 模型配置（通过现有 `OCRProcessor` 接入）。

**非目标（本次不做）**
- 不做文档版式分析/整页 Markdown 还原（项目现架构是「OCR 取文本 + LLM 结构化」，版式由 LLM 承担，不需要 VLM 的整页能力）；
- 不做多语种/公式识别（教案以中文为主）；
- 本次只出**方案与实验计划**，不训练、不下载权重。

---

## 2. 候选技术路线对比

> 三个路线并非互斥，推荐按 C → A →（可选）B 递进。

### 路线 C：不微调，先换预训练引擎做 Baseline（快赢，约 1 天）

- **做法**：保持 `OCRProcessor` 接口，把默认引擎从 `easyocr` 切到 `paddleocr`（代码已具备 `PaddleOCRProcessor`），在真实手写教案样本上与 EasyOCR 对比。
- **优点**：几乎零成本、风险最低；PaddleOCR 中文识别（PP-OCRv5）本就强于 EasyOCR，很可能一步就明显改善。
- **缺点**：预训练模型仍以印刷/常规手写为主，遇到本领域连笔字仍有差距；不一定能到 ≥90%。
- **结论**：**必做的第一步**，为后续微调提供对照基线。

### 路线 A：专用轻量识别模型微调（**推荐主攻**）

- **模型**：PaddleOCR **PP-OCRv5 识别模型**（`SVTR_LCNet` 骨干），或 TrOCR/SVTR 系。
- **做法**：冻结/微调识别网络，在「CASIA-HWDB 连续手写 + SCUT-HCCDoc 手写文档 + 本地标注教案」上做域自适应训练。
- **优点**：
  - **显存极低**：识别模型仅 ~10MB 量级，8GB 4060 全参微调轻松（<2GB 占用）；
  - **速度极快**：单行识别毫秒级，整页在 ≤5s 内轻松达标；
  - **与项目架构天然契合**：输出就是文本行，直接喂给现有 LLM 结构化链；
  - PaddleOCR 有官方训练脚本与自定义数据格式，工程成熟。
- **缺点**：只负责「文字识别」，不解决版式；需要做数据标注（行级/页级）。
- **结论**：**本项目的性价比最优解**。

### 路线 B：小 VLM 微调（Xiaomi-OCR-0 0.8B / Qwen2.5-VL-3B，实验分支）

- **做法**：用 ms-swift / LLaMA-Factory 对 0.8B VLM 做 **4-bit QLoRA**，数据为「教案图片 → 结构化 Markdown/文本」整页级样本。
- **优点**：端到端「图→结构化文本」，对非规整手写/版式鲁棒性上限更高；Xiaomi-OCR-0 本身就是 OCR 中心 VLM。
- **缺点（8GB 下明显）**：
  - **无训练代码**：Xiaomi-OCR-0 仓库仅推理；且 Qwen3.5 多模态架构的微调工具支持可能滞后，需自行打通；
  - **显存紧**：0.8B 4-bit + LoRA + 视觉激活，batch=1、控分辨率下约 6–8GB，**勉强可行但脆弱**；
  - **速度风险大**：整页 VLM 推理在 4060 上很可能 **>5 秒/张**，与项目硬指标冲突；
  - 整页级标注成本高。
- **结论**：作为**可选增强/技术储备**，不作为默认方案；只有在路线 A 不达标且能接受时延放宽时才启动。

### 对比总表

| 维度 | C 换预训练 | **A 识别模型微调（推荐）** | B VLM QLoRA |
|---|---|---|---|
| 中文手写准确率上限 | 中 | 高 | 高（整页鲁棒） |
| 8GB 4060 显存 | 极低 | **低（<2GB）** | 高（6–8GB，紧） |
| 推理时延（≤5s） | 远低于 5s | **远低于 5s** | ⚠️ 可能 >5s |
| 训练代码成熟度 | 无需训练 | **高（PaddleOCR 官方）** | 低（需自建/兼容 Qwen3.5） |
| 与现有架构契合 | 高 | **高（输出文本行）** | 中（需改链路） |
| 数据成本 | 无 | 中（行/页标注） | 高（整页标注） |
| 落地风险 | 低 | **低-中** | 高 |

---

## 3. 数据方案

### 3.1 数据来源（公开 + 本地）

| 类型 | 数据集 | 规模/说明 | 用途 |
|---|---|---|---|
| 中文手写（单字/连续行） | **CASIA-HWDB 1.x / 2.x**（中科院自动化所） | 离线手写，约百万级单字样本 / 约 5000+ 连续手写页，1000+ 书写人；标准 train/test 划分 | 预训练/域自适应主力 |
| 中文手写文档（拍照） | **SCUT-HCCDoc** | 约 1.2 万张拍照文档图，140 万+ 手写文本实例（行级标注） | 贴近「拍照教案」真实分布 |
| 手写数学表达式（可选） | HME100K / CROHME | 10 万级 | 若教案含公式 |
| 英文手写（可选） | IAM | ~1500 页 | 若含英文 |
| **本项目域内数据** | 你的本地标注教案 | **建议 100–300 页**真实手写教案，留 30–50 页做测试集 | 域自适应 + 评测锚点 |

> 注：CASIA-HWDB 与 SCUT-HCCDoc 均有学术使用许可限制，商用前需确认授权；本项目为演示/研究用途风险较低，但仍建议记录来源与许可。

### 3.2 标注规范（行级为主）

- **格式**：每行一条 `图片相对路径\t转写文本`（PaddleOCR rec 训练格式），或 JSONL `{"image":..., "text":..., "box":[...]}`。
- **粒度**：以「文本行」为标注单元（与现有 `readtext` 逐行取文本一致）。
- **规则**：保留原意，标点统一为中文全角；无法辨认的字用 `□` 占位，不臆造（与 `SKILL.md` 里"不填补不可读内容"的原则一致）。
- **质检**：双人交叉复核 10% 抽样。

### 3.3 域内数据合成（低成本扩充）

- 用公开课本文本 + 项目已有教案内容，渲染到**横线本/方格本/纯白纸**背景上，叠加轻微透视、模糊、光照、纸张纹理，生成「伪手写教案页」；
- 手写字体可用开源中文手写字体（如「站酷快乐体」等）做弱监督，但注意合成分布与真人手写差异，**合成数据只做增强、不替代真实标注**。

### 3.4 数据划分

- 训练 / 验证 / 测试 = 8 : 1 : 1（域内真实数据）；测试集**必须固定**且含不同书写人、不同纸张，防止数据泄漏（同一页裁出的行不能跨集）。

---

## 4. 训练方案（8GB GPU 为主）

### 4.1 路线 A：PP-OCRv5 识别模型微调

| 项 | 设定 |
|---|---|
| 框架 | PaddleOCR（官方训练脚本，`tools/train.py`，rec 配置） |
| 骨干 | SVTR_LCNet，输入 `3×48×320`（可按需加大高度到 64） |
| 初始化 | PP-OCRv5 官方中文识别权重（预训练 → 微调） |
| 优化 | AdamW/余弦退火，lr≈5e-4（微调）/1e-3（从头），warmup |
| batch | 4060 8GB 可 batch 32–64（AMP），显存 <2GB |
| epoch | 50–100，早停看验证 CER |
| 增强 | 随机透视、模糊、亮度、笔画粗细、纸张叠加 |
| 耗时预估 | 单 epoch（万级行）约数分钟；整体数小时可收敛 |
| M2 本地 | 仅做推理/小样本验证；训练放 4060 |

> 训练数据以「中文行图像 + 转写」为主；CASIA 连续页可先做行切分后使用，SCUT-HCCDoc 已有行级标注可直接用。

### 4.2 路线 B：0.8B VLM QLoRA（仅实验分支）

| 项 | 设定 |
|---|---|
| 框架 | ms-swift 或 LLaMA-Factory（QLoRA，4-bit NF4） |
| 模型 | 首选 `SeerRay-Lab/Xiaomi-OCR-0`；兼容性受阻时退用 `Qwen2.5-VL-3B-Instruct` 做对照 |
| 显存 | 4060 8GB：batch=1 + 梯度累积 + 梯度检查点 + 限制 `max_pixels`（≈512–768 边长），约 6–8GB |
| 耗时 | 数百张整页样本，数小时~十几小时；推理时延需实测是否 ≤5s |
| 风险 | Qwen3.5 架构工具链支持、整页标注成本、时延指标 |

### 4.3 框架/环境清单

- **Windows RTX 4060**：CUDA + PaddlePaddle-GPU（先按官方选 CUDA 版本）→ PaddleOCR；QLoRA 分支用 PyTorch + ms-swift/LLaMA-Factory。
- **Mac M2 16GB**：MLX / PaddleOCR-CPU 仅做推理和小样本冒烟测试，不承担大规模训练。

---

## 5. 评测方案

### 5.1 指标

| 层级 | 指标 | 目标 |
|---|---|---|
| 识别层 | **CER 字符错误率**（编辑距离/参考长度）、行完全正确率 | CER 显著低于 EasyOCR baseline；行准确率 ≥90% |
| 业务层 | 结构化字段（学科/年级/课题/教学目标…）抽取正确率、KIE F1 | 相比 baseline 明显提升，尽量 ≥90% |
| 性能 | 单图识别 P50/P95 时延 | **P95 ≤5 秒/张**（4060） |
| 稳定性 | 空白/模糊/倾斜图不崩溃、不臆造 | 通过 |

### 5.2 对照实验矩阵（必做）

1. EasyOCR（现状）—— 基线
2. PaddleOCR-PP-OCRv5 预训练（路线 C）
3. PP-OCRv5 微调（路线 A）
4. （可选）VLM QLoRA（路线 B）

在**同一固定测试集**上跑同一套指标，产出对比表 + 每类错误样例（连笔、数字、学科术语、表格）。

### 5.3 复现与数据可追溯

- 固定随机种子、固定测试集、保存训练配置与 checkpoint；
- 评测脚本输出 `metrics.json` + 错误样例目录，便于回归。

---

## 6. 集成方案（低侵入）

1. 在 `OCRProcessor.create()` 中新增 `ocr_type="paddleocr_ft"` 分支，加载微调后的识别模型（det 用 PP-OCRv5 检测，rec 用微调权重）；
2. `WisdomTransferAgent.__init__` 改为 `self.ocr = OCRProcessor.create(settings.ocr_type)`，`recognize_handwriting()` 调 `self.ocr.recognize(image_path)`（正是 `开发计划` 里预留的 Task）；
3. `settings.py` 增加 `ocr_model_dir`、`ocr_use_gpu` 等配置；默认引擎可一键回退 `easyocr`；
4. 下游 LLM 结构化链**不改**（继续保持「OCR 文本 → LLM 结构化」的解耦）。

---

## 7. 实验计划（分阶段里程碑）

| 阶段 | 内容 | 产出 | 预估 | 依赖 |
|---|---|---|---|---|
| **P0 基线复现** | 部署 PaddleOCR，收集 30–50 页真实教案测试集，跑 EasyOCR vs PaddleOCR 对照 | baseline 报告 + 固定测试集 | 2–3 天 | 你的本地数据 |
| **P1 数据工程** | 收集/清洗 CASIA、SCUT-HCCDoc；标注 100–300 页本地数据；合成增强；划分 | 训练/验证/测试集 + 标注规范 | 1–2 周 | 标注人力 |
| **P2 路线 A 微调** | PP-OCRv5 rec 微调，迭代超参 | 微调 checkpoint + 评测报告 | 3–5 天（含调参） | P1 + 4060 |
| **P3 达标判定** | 对比矩阵；判断是否 ≥90% / ≤5s | 结论 + 是否需 P4 | 1 天 | P2 |
| **P4（可选）路线 B** | 0.8B VLM QLoRA 实验分支 | 对照结果 / 可行性结论 | 1–2 周 | P3 未达标时 |
| **P5 集成与回归** | 接入 `OCRProcessor`/`WisdomTransferAgent`，端到端回归 | 可切换模型配置 + 端到端 demo | 3–5 天 | P2/P4 |

**关键决策点**：P0 结束后若 PaddleOCR 预训练已达 ≥90%，可直接跳到 P5 集成，微调降级为「可选优化」；P3 若路线 A 达标则不做 B。

---

## 8. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 域内真实手写数据不足/标注贵 | 高 | 公开集先训、合成增强、先小样本（100 页）验证收益 |
| 连笔/行草识别仍差 | 高 | 增加真实样本、针对性增强；必要时上 VLM 分支 |
| 8GB 显存不足（VLM 分支） | 中 | 4-bit QLoRA + 控分辨率 + batch1；A 路线不受影响 |
| Qwen3.5 微调工具不支持 | 中 | 退用 Qwen2.5-VL-3B 做对照，或等工具链更新 |
| 评测泄漏/虚高 | 中 | 固定测试集、按书写人/页划分、错误样例复核 |
| 数据集许可 | 低 | 记录来源，商用前确认授权 |

---

## 9. 需要你确认/提供的事项

1. **本地标注数据**：现有多少页？标注格式（行文本 / JSON / 字段）？能否提供 30–50 页做测试集？（这是 P0 的前提）
2. **优先目标**：是「只求文字识别准确率」还是「要端到端结构化字段准确」？（影响是否走 VLM）
3. **时延红线**：≤5s/张 是否硬指标？（若可放宽，VLM 路线才有意义）
4. **部署环境**：最终 OCR 跑在 Windows 4060（GPU）还是 Mac/服务器 CPU？决定训练与推理落点。
5. **数据许可**：CASIA/SCUT-HCCDoc 用于演示是否可接受？

---

## 10. 参考

- Xiaomi-OCR-0 仓库（推理/评测，无训练代码）：<https://github.com/xiaomi-research/Xiaomi-OCR-0> · 权重 `SeerRay-Lab/Xiaomi-OCR-0`
- 本项目 demo 仓库：<https://github.com/Komorebi-GZM/wheatfield_braintrust>
- CASIA 手写数据库：<https://nlpr.ia.ac.cn/databases/handwriting/Offline_database.html>
- SCUT-HCCDoc：<https://www.sciencedirect.com/science/article/abs/pii/S0031320320303629>
- PP-OCRv5 介绍：<http://www.paddleocr.ai/v3.0.3/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html>
- PaddleOCR 垂直场景微调 PP-OCRv4：<https://cloud.baidu.com/article/3360328>
- LLaMA-Factory 微调 Qwen2.5-VL-3B（QLoRA 参考）：<https://blog.csdn.net/watson2017/article/details/150590108>
- EffOCR 论文（EasyOCR/PaddleOCR/TrOCR 同数据微调对比）：<https://aclanthology.org/2024.acl-long.440.pdf>
