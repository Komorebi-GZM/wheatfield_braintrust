# 手写课程笔记 OCR 模型设计说明书

> 归属：麦田智囊（wheatfield_braintrust）**OCR 子模块**（本仓库仅承担 OCR 能力）
> 基座模型：Xiaomi-OCR-0（0.8B，`SeerRay-Lab/Xiaomi-OCR-0`，Qwen3.5-0.8B 基座）
> 训练方式：4-bit QLoRA 微调
> 算力：优先 Windows RTX 4060 8GB（训练可接受 15h+），Mac M2 16GB 兜底（推理/评测）
> 本轮交付：**模型设计（不训练、不下载大权重、不过度克隆）**

---

## 1. 任务定义

### 1.1 问题

输入一张**手写课程笔记**照片，输出**保留结构与公式**的结构化 Markdown。不是"文字行拼接"，而是"整页→结构化学术文本"。

### 1.2 输入 / 输出规格

**输入**：单页手写笔记图片（jpg/png，手机拍摄或扫描）；多页按页逐张处理。

**输出**：单个 Markdown 文档，必须保含以下四类结构（用户已确认）：

| 结构类型 | 输出形式 |
|---|---|
| 标题层级 / 小节 | `# / ## / ###`（按笔记层级推断） |
| 分点 / 列表 / 缩进 | `-` / `1.` / 嵌套缩进 |
| 表格 | Markdown 表格（模型出 OTSL，后处理转 HTML/Markdown 表） |
| 公式（行内 / 行间） | LaTeX：行内 `$...$`，行间 `$$...$$` |

**明确不处理**：插图/图表/示意图的**内容**——仅保留占位（如 `> [图示：xxx]`），不臆造。

### 1.3 非目标

- 不做多语种（以中文为主，允许夹杂少量英文/符号）；
- 不保证插图/几何图形结构还原；
- 不做实时（离线批处理，10–30s/页可接受）。

---

## 2. 模型选型

### 2.1 主选：Xiaomi-OCR-0（0.8B）QLoRA

**为什么是它：**

- 原生能力就对准了「结构 + 公式」：输出**结构化 Markdown**，公式用 **LaTeX**，表格用 **OTSL→HTML**，官方公式基准 UniMER-Test 97.84、表格 PubTabNet 92.27（印刷域）。
- 参数量仅 0.8B：8GB 显存可做 **4-bit QLoRA**；LoRA adapter 仅几十 MB，便于推送到远程 branch。
- 与项目"OCR 结果再交给 LLM 结构化"的解耦架构天然衔接（OCR 阶段直接产出带结构的 Markdown，下游可选择性再整理）。

**风险与对策：**

- 原模型训练数据以**印刷文档**为主，**手写 + 课程笔记风格**存在域差 → 靠 QLoRA 域自适应 + 合成手写数据弥补。
- Qwen3.5 多模态架构的训练工具支持可能滞后 → 训练框架用 **ms-swift**（对 Qwen-VL 系支持较全）；若受阻，退用 **Qwen2.5-VL-3B** 做对照基线（能力相近、工具成熟），但 3B 在 8GB 更紧。

### 2.2 备选

| 方案 | 适用 | 说明 |
|---|---|---|
| Qwen2.5-VL-3B QLoRA | 主选工具链不通时 | 文档/公式能力强，8GB 下需更严格控分辨率 |
| 双阶段流水线（检测+行识别+公式识别） | 只求速度、可接受结构损失时 | 复杂度高，本设计不采用（用户要结构与公式） |

**结论：采用「Xiaomi-OCR-0 + QLoRA」单模型端到端结构化方案。**

---

## 3. 数据设计（本项目最大难点：仅 20–30 页）

### 3.1 真实数据（种子）

- 用户提供 **20–30 页手写课程笔记**。
- 划分建议（页级，严禁同页跨集）：**训练 16–20 页 / 验证 4–5 页 / 测试 4–5 页**；因样本极少，测试集必须固定且首批锁定，不得回流训练。
- 若页数不足 20，改用 **5-fold 交叉验证**，报告均值±方差。

### 3.2 标注规范（整页级）

每页产出一条样本：

```json
{
  "image": "notes/001.jpg",
  "markdown": "# 第三章 电磁感应\n\n## 3.1 法拉第定律\n\n- 感应电动势 ...\n\n$$ \\varepsilon = -\\frac{d\\Phi}{dt} $$\n\n| 量 | 符号 |\n|---|---|\n| 磁通 | $\\Phi$ |"
}
```

规则：
- 只标注**可见内容**；不可辨认处用 `□` 或 `[?]`，**严禁补全/猜测**（与上游 SKILL 原则一致）。
- 标题层级以笔记实际缩进/编号为准；列表保留层级。
- 公式转 LaTeX；表格按单元格还原。
- 双人交叉复核 20% 抽样。

### 3.3 合成数据（关键扩增，弥补 20–30 页的不足）

真实数据太少，**必须用合成数据撑起训练规模**，否则 0.8B 模型会过拟合：

1. **文本来源**：公开教材/讲义文本 + 项目已有教案文本 + 公式语料。
2. **渲染管线**：将「带 Markdown/LaTeX 结构的文本」渲染为**手写风格页面**：
   - 中文手写字体（开源手写字体若干种，模拟不同书写人）；
   - 公式用 LaTeX 渲染后做"手写化"扰动（笔画抖动、倾斜）；
   - 页面合成：横线本/方格本/空白纸背景 + 透视 + 光照 + 阴影 + 噪点 + 折痕；
   - 结构保留：标题字号更大、列表缩进、表格线绘制。
3. **规模**：合成 **500–2000 页**，仅进训练集；真实页永远做验证/测试锚点。
4. **注意**：合成分布 ≠ 真人手写，合成仅作"结构与公式形态"的弱监督；真实 20–30 页决定最终效果上限。

### 3.4 公开数据（能力保持 / 防遗忘，可选小比例混入）

| 数据集 | 用途 | 说明 |
|---|---|---|
| CASIA-HWDB 1.x/2.x | 中文手写字符/行 | 保持手写字符识别能力 |
| SCUT-HCCDoc | 拍照手写文档 | 贴近"拍照笔记"分布 |
| HME100K / CROHME | 手写数学公式 | 强化公式识别 |
| UniMER-Test 合成公式 | 公式 LaTeX 输出 | 保持公式能力 |
| 印刷文档解析集 | 结构与表格 | 防结构能力遗忘 |

混入比例建议：合成页 : 公开增强 : 真实页 ≈ 6 : 3 : 1（可调）。

### 3.5 防过拟合

- LoRA 只训少量参数；配合 dropout、数据增强、早停；
- 真实数据重复采样（oversample）+ 合成多样性；
- 监控验证集 CER 与结构/公式指标，避免只记住 20 页。

---

## 4. 结构与公式的"保证机制"（不只靠模型）

单靠模型输出不可靠，设计三层保障：

1. **输出协议约束**：固定 system prompt，要求"仅输出 Markdown + LaTeX，标题用 #、列表用 -、表格用 Markdown 表、公式用 $/$$"，并给出 few-shot 示例（取真实笔记）。
2. **后处理校验**：
   - **LaTeX 语法校验**：用 `pylatexenc`/`latex2mathml` 解析，非法公式标记待修；
   - **表格转换**：OTSL → HTML/Markdown（复用 Xiaomi-OCR-0 的 `postprocess/` 逻辑）；
   - **结构解析**：解析 Markdown AST，校验标题层级合法、括号/环境配对；
3. **失败回退**：
   - 结构化解析失败 → 回退「纯文本 + 二次 LLM 规范化」；
   - 公式解析失败 → 保留原样并标记 `⚠️公式待校`，不臆造；
   - 可选：用 LLM 只做**格式规范化**（不改内容），需人工抽检防幻觉。

---

## 5. 训练设计（4-bit QLoRA / 8GB / 15h+）

| 项 | 设定 |
|---|---|
| 框架 | **ms-swift**（首选，Qwen-VL 支持好）；备选 LLaMA-Factory |
| 量化 | 4-bit NF4（bnb），双量化 |
| LoRA | `r=8~16`, `alpha=16~32`, `dropout=0.05`，作用于 attention/MLP |
| 精度 | bf16 计算 / fp16 视 CUDA 支持 |
| batch | per-device=1，grad_accum=8~16（等效 batch 8–16） |
| 显存策略 | 梯度检查点开；`max_pixels` 限制长边 ≈ 768–1024（结构/公式需较高分辨率，是显存主要矛盾点） |
| 学习率 | 1e-4 ~ 2e-4，cosine，warmup 3% |
| epoch | 3–8（合成数据量大时更多 step），按验证指标早停 |
| 训练预算 | Windows 4060 8GB：**15h+ 可接受**；预计 1–3k step 视数据量 |
| 产物 | **LoRA adapter**（几十 MB）+ 训练配置 + 日志 |
| Mac M2 16GB | **不训练**；仅做推理冒烟、标注预览、评测（MLX / MPS） |

**显存预算粗估（4060 8GB）**：0.8B 4-bit 权重 ≈0.5–0.6GB；视觉塔激活 + 高分辨率图像是主要开销；梯度检查点 + 限分辨率后约 6–8GB，需实测微调 `max_pixels`。若 OOM：降分辨率 → 降 LoRA r → 减 grad_accum 或冻结视觉塔（优先保语言侧结构输出）。

> 视觉塔（ViT）是否训练：建议**先冻结视觉塔，只训 LLM 投影+LoRA**（省显存、防过拟合），不达标再解冻部分视觉层。

---

## 6. 评测设计

固定测试集（4–5 页真实，首批锁定），四层指标：

| 层级 | 指标 | 说明 |
|---|---|---|
| 文本 | **CER**（去格式后编辑距离/长度） | 基础识别质量 |
| 结构 | 标题层级匹配率、列表项数匹配、**表格结构匹配率**（行列/单元格） | "保结构" |
| 公式 | 公式 LaTeX **归一化后 exact-match**、公式 CER、行内/行间分类正确率 | "保公式" |
| 端到端 | 人工评分（结构完整性/公式正确/可读性，1–5 分）+ 综合保持率 | 最终验收 |

**对照实验**：
1. Xiaomi-OCR-0 **zero-shot**（微调前）；
2. 微调后（本设计）；
3. （参考）EasyOCR（项目现状）仅文本 CER。

输出 `metrics.json` + 每类错误样例（公式、表格、层级）目录，便于回归与迭代。

---

## 7. 推理与集成设计（本仓库 = OCR-only 模块）

### 7.1 本仓库定位

本仓库只承载 OCR：**模型加载 / 推理 / 评测 / 数据管线**，不承载麦田智囊的 Agent/RAG/前端。对外只暴露一个「整页→结构化 Markdown」的接口。

### 7.2 建议目录骨架

```
ocr_module/
├── README.md
├── requirements.txt            # transformers/peft/accelerate/ms-swift(训练)/pylatexenc/Pillow...
├── configs/
│   ├── train_qlora.yaml        # QLoRA 超参
│   └── infer.yaml              # 基座+adapter+分辨率+prompt
├── data/
│   ├── raw/                    # 20-30 页真实笔记（不入库/或 git-lfs）
│   ├── annotations/            # 整页级 image->markdown
│   ├── synth/                  # 合成脚本输出
│   └── splits/                 # train/val/test 清单（页级）
├── scripts/
│   ├── build_synth.py          # 合成手写笔记页
│   ├── prepare_dataset.py      # 转 ms-swift 数据格式
│   ├── train_qlora.sh          # 训练入口（4060）
│   └── evaluate.py             # CER/结构/公式指标
├── ocr/
│   ├── __init__.py
│   ├── engine.py               # 加载基座+adapter，recognize_page()
│   ├── postprocess.py          # OTSL→表、LaTeX 校验、结构校验
│   └── schema.py               # 输出 schema 与校验
└── tests/
    └── test_metrics.py
```

### 7.3 对外接口

```python
# ocr/engine.py
def recognize_page(image_path: str) -> dict:
    return {
        "markdown": "...",        # 结构化 Markdown（含 LaTeX / 表格）
        "structure_ok": True,     # 结构校验结果
        "latex_issues": [],       # 非法公式列表
        "model": "xiaomi-ocr-0-qlora-v1",
    }
```

与麦田智囊对接：将其作为 OCR 工具/微服务，`WisdomTransferAgent` 拿到 `markdown` 后再做字段结构化（保持现有解耦）。

### 7.4 远程 branch 试验流程

- 分支：`feat/ocr-xiaomi-qlora`
- 入库内容：**训练配置、推理封装、评测脚本、数据清单、报告**；
- **LoRA adapter 与权重不入 git**：用 `.gitignore` + **git-lfs** 或托管到 HF/对象存储，branch 中记录版本与下载地址；
- 推送前跑 `evaluate.py` 产出对照报告一并提交，便于远程 review。

---

## 8. 里程碑（设计 → 数据 → 训练 → 评测 → 集成）

| 阶段 | 内容 | 产出 |
|---|---|---|
| M0 设计 | 本文档定稿（已完成） | 模型设计说明书 |
| M1 数据 | 真实 20–30 页标注 + 合成管线 + 划分 | 数据集 + 标注规范 |
| M2 训练 | QLoRA（4060，15h+） | LoRA adapter + 训练日志 |
| M3 评测 | 四层指标 + 对照 | metrics + 错误样例 + 报告 |
| M4 集成 | 封装 `recognize_page` + 远程 branch 推送 | 可试用 OCR 模块 |

---

## 9. 风险与应对

| 风险 | 应对 |
|---|---|
| 20–30 页太少 → 过拟合 | 合成数据扩增 + LoRA + 强正则 + 早停 |
| 手写域差大 → 公式/结构错 | 混入 HME100K/UniMER 合成 + 增强 |
| 8GB 显存 OOM | 冻结视觉塔、降分辨率、减 r、梯度检查点 |
| Qwen3.5 工具链不支持 | 退用 Qwen2.5-VL-3B 基线对照 |
| LaTeX 语法错误 | 后处理校验 + 标记待校，不臆造 |
| 合成-真实分布差 | 真实页只做验证/测试锚点，合成不参与评测 |

---

## 10. 假设与待确认

- 假设笔记为**中文手写**，公式为数学表达式；结构含标题/列表/表格/公式（用户已确认）。
- 待确认：① 20–30 页是否均为手写、含多少表格/公式；② 是否接受合成数据参与训练；③ 最终 OCR 推理跑在 Windows 4060 还是 Mac（影响封装与性能基线）。

---

## 11. 参考

- Xiaomi-OCR-0（推理/评测仓库）：<https://github.com/xiaomi-research/Xiaomi-OCR-0> · 权重 `SeerRay-Lab/Xiaomi-OCR-0`
- 麦田智囊 demo 仓库：<https://github.com/Komorebi-GZM/wheatfield_braintrust>
- PaddleOCR PP-OCRv5：<http://www.paddleocr.ai/v3.0.3/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html>
- CASIA 手写库：<https://nlpr.ia.ac.cn/databases/handwriting/Offline_database.html>
- SCUT-HCCDoc：<https://www.sciencedirect.com/science/article/abs/pii/S0031320320303629>
- LLaMA-Factory 微调 Qwen2.5-VL（QLoRA 参考）：<https://blog.csdn.net/watson2017/article/details/150590108>
