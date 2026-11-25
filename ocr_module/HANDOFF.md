# HANDOFF — 手写课程笔记 OCR（Xiaomi-OCR-0 + QLoRA）交付说明

- 最后更新：2026-11-15
- 项目目录：`ocr_module/`（下称"项目根"）。**本文件所有相对路径都以 `ocr_module/` 为根**，不含硬编码绝对路径。
- 交付分支：`feat/ocr-xiaomi-qlora`（仓库 `Komorebi-GZM/wheatfield_braintrust`）
- 结论一句话：链路、数据、训练都已完成并可复现；**基座 vs 微调是"混合变化"，不能宣称整体 OCR 能力已改善**。
- 详细验收：`final_comparison_report_v2.md`（本文件不重复其全部证据，只给可执行路径）

---

## 0. 结果速览（数值已与产物核对）

| 指标 | 基座 | 微调 checkpoint-26 | 出处 |
|---|---|---|---|
| 有效（可评分）测试页 | **3**（5 页中 2 页排除） | **3** | `outputs/eval_v3/*/metrics.json` |
| 正文宏平均 CER（公式文本不计） | **1.4872** | **0.7725** | 同上 |
| 正文字符加权 CER | 1.4386 | 0.7433 | 同上 |
| 含公式体宏平均 CER | **1.1217** | **1.7952** | `outputs/eval_v3/supplement.json` |
| 含公式体字符加权 CER | 1.1125 | 1.9119 | 同上 |
| 失败率（命中上限或检出重复） | **66.7%** | **66.7%** | `outputs/eval_v3/comparison_v3.json` |
| 公式精确匹配率 / 标题序列匹配率 | 0.0 / 0.0 | 0.0 / 0.0 | `outputs/eval_v3/*/metrics.json` |

**必须一起读的三件事：**

1. 主指标（公式不计）改善，是因为评估器的 `plain_text()` 会把 `$...$` / `$$...$$` **整段删除**；微调恰好把内容裹进公式，等于从分母里消失。把公式体算回来，微调反而更差。
2. 只有 **3 个可评分页**（000020、000028 因参考含 `□` 整页排除，双方同规则）。单页变化即可主导均值，**不支持泛化结论**。
3. 唯一可确认的实质改善在 `000037`（不再编造 `<br>` 宽表、结构与术语更接近原图）。`000044`、`000051` 两臂都失败。

---

## 1. 环境与依赖

| 组件 | 本机实测版本 |
|---|---|
| OS | Windows 11（WDDM 驱动模型） |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB |
| Python | 3.12.13 |
| torch | 2.6.0+cu124 |
| transformers | 5.14.1 |
| peft | 0.20.0 |
| ms-swift | 4.6.0 |
| bitsandbytes | 0.50.2 |
| qwen-vl-utils | 0.0.14 |

安装：

```bash
# 1) 虚拟环境（本机在仓库上一级：E:/ocr/wheatfield_braintrust/.venv，可自行更换）
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash
# .venv\Scripts\activate             # Windows cmd / PowerShell

# 2) 依赖
cd ocr_module
pip install -r requirements.txt
```

- `requirements.txt` 已写明 `transformers>=5.2`、`ms-swift>=4.6`、`bitsandbytes>=0.50.2`、`qwen-vl-utils>=0.0.14`：低于这些版本时 `qwen3_5` 的 model_type/template 不被识别，4-bit 加载方式也不同。
- 原生 Windows 的 bitsandbytes 需与该 CUDA 版本匹配的 wheel；若装不上，走 WSL2（训练脚本也能在 bash 下改写成 `swift sft ...` 直接跑）。
- 模型缓存默认落在 `outputs/modelscope_cache`（训练脚本会设 `MODELSCOPE_CACHE`），该目录被 Git 忽略。

---

## 2. 路径与可移植性

- 所有脚本都**不硬编码绝对路径**：默认值均为相对 `ocr_module/` 的路径，`--manifest/--pred/--output` 等参数可覆盖。
- 但有两类**不可移植的记录**，迁移时须知：
  1. **运行 sidecar**（`*.jsonl.protocol.json`）与逐页预测里的 `image` 字段保存的是原始机器的绝对路径（`E:\ocr\...`）。它们只用于溯源，**不能直接在新机器上重跑**；重跑请用 `--manifest` 指向本机路径。
  2. `checkpoint-26/adapter_config.json` 里的 `base_model_name_or_path` 也是绝对路径。**加载时不会用它**：`ocr/engine.py` 用 `base_model` 参数（模型 id 或本机路径）加载基座，再挂 adapter。若要改这个字段，请同时记录它的新 SHA-256（原值见 `CHECKPOINT_MANIFEST.md`）。
- 迁移步骤：拷贝仓库 → 装依赖 → 按第 3 节的哈希校验数据 → 用第 5 节命令重跑（推荐先 `--dry-run`）。

---

## 3. 数据版本与固定划分

| 项 | 值 |
|---|---|
| 标注文件 | `datasets/handwriting_notes/annotation_template.jsonl` |
| 版本 / 哈希 | v2（2026-11-15）/ `34f262c270b2bf955a9d039b7d80f355c13e20bd87b8e7709c97201d19d17732` |
| 上一版（可回滚） | `datasets/handwriting_notes/annotation_template.v1.bak.jsonl` / `b45d05ebd61432a0…` |
| 版本清单（含 23 张原图逐张 SHA-256） | `datasets/handwriting_notes/data_version_v2.json` + `.sha256` |
| 标注复核报告 | `annotation_review_v2.md`（逐页 23 条，修正 10 处 / 7 页） |
| 修正明细 | `outputs/annotation_fixes_v2.json` |

固定划分（13 / 5 / 5，互斥、覆盖全部 23 页，本阶段已重算确认）：

| 集合 | 页数 | 文件 | SHA-256 |
|---|---|---|---|
| train | 13 | `datasets/handwriting_notes/splits/train.jsonl` | `aabb1af92132e11b8c829a4613349f86b981341080c608a970a843c054be73da` |
| val | 5 | `.../splits/val.jsonl` | `aed108254fdfc01e8ee94c6ebb303f20acf3807a5a1a2e78f91d636316fddab4` |
| test | 5 | `.../splits/test.jsonl` | `67a5df075e8575e3d1e9005633783076a67c2ce681b0fed3025319e3f1563f0a` |
| train（Swift 格式） | 13 | `.../splits/train_swift.jsonl` | `86e17f9d9e184dfa42521e4c1b2b9c7705e073cf481876365ea7c751bdcee7db` |
| val（Swift 格式） | 5 | `.../splits/val_swift.jsonl` | `f959cbc51aa2242afa00a48d11431b4736ebb351cfe5a5ce2b4fc4a49bf050b9` |

- **test 成员**：000020、000028、000037、000044、000051；**val**：235932、235946、235955、000005、000012；**train**：其余 13 页。
- 合成数据：本轮 `synthetic.count=0`；策略仍是"只能由 `scripts/prepare_dataset.py` 拼进训练集，val/test 只用真实页"。
- 标注局限：`reviewed=true` **不是正确性证明**。`annotation_review_v2.md` 第 4 节列出仍存疑项（如 200854「标称/有效标距」中等置信、竖排小字页未逐字确认）。看不清处一律 `□`（5 页 32 处），评测对含 `□` 的页整页排除。
- **存在第二份标注**：`data/annotations/real.jsonl`（23 行，远程分支上由标注 bot 生成，已在仓库中）。它**未**被任何 split 引用，**未**参与训练或评测；逐页抽验显示它在 200907、200914、200858 等处读错，并对 200843 误判"图片损坏"。因此不能当作真值，也不能把"远程版 `□` 多/本地版 `□` 少"解读成谁更忠实（详见 `annotation_review_v2.md` 第 6 节、`datasets/handwriting_notes/remote_annotation_review.md`）。

---

## 4. 训练

### 4.1 命令

```bash
cd ocr_module
powershell -File scripts/train_qlora_v2.ps1 -OutputRoot outputs/qlora-v2
# 可选：-BaseModel <id|path> -Epochs 4 -GradAccum 1 -MaxSteps -1 -Inspect
```

脚本会先把完整实参写进 `<runDir>/command.txt`，再保存 `args.json`、`logging.jsonl`、`checkpoint-N/`。

### 4.2 真实实参（取自 `outputs/qlora-v2/v2data-20261115-134816/args.json`）

`model=SeerRay-Lab/Xiaomi-OCR-0`、`model_type=qwen3_5`、`template=qwen3_5`、`tuner_type=lora`、`target_modules=all-linear`（落地后仅 `model.language_model.*`，372 模块）、`quant_method=bnb`、`quant_bits=4`、`nf4` + `double_quant` + `compute_dtype=bfloat16`、`lora_rank=8`、`lora_alpha=32`、`lora_dropout=0.05`、`freeze_vit=true`、`freeze_aligner=true`、`dataset=train_swift.jsonl`、`val_dataset=val_swift.jsonl`（**显式接入**）、`split_dataset_ratio=0`、`max_length=8192`、`truncation_strategy=delete`、`max_pixels=1024000`、`batch=1`、`grad_accum=1`、`lr=1e-4`、`cosine`、`warmup_ratio=0.03`、`epochs=4` → **52 步**、`seed=42`、`eval_strategy=epoch`、`save_strategy=epoch`、`save_total_limit=4`。

`configs/train_qlora.yaml` 是**模板**（`save_total_limit: 2` 等字段与本跑不同），以 `args.json` 为准确认实际参数。

### 4.3 最佳 checkpoint 选择依据

| ckpt | step | eval_loss | eval_token_acc |
|---|---|---|---|
| 13 | 13 | 1.20785 | 0.7691 |
| **26** | **26** | **1.18883** | **0.7781** |
| 39 | 39 | 1.18942 | 0.7747 |
| 52 | 52 | 1.19147 | 0.7773 |

**选 checkpoint-26（验证集 eval_loss 最低）。测试集全程未参与任何选择。**
注意：验证 loss 下降**不等于** OCR 效果改善（本次恰是反例：loss 降了、含公式 CER 升了）。

### 4.4 Windows 已知坑（已修）

`ms-swift` 的 `update_last_checkpoint_symlink` 在无符号链接权限时（`WinError 1314`）回滚会对空目录调 `os.remove()` → `PermissionError [WinError 5]`，在第 1 个 save 处崩溃。脚本预先在 run dir 建同名空目录让其走早退分支，**一律用真实 `checkpoint-N` 目录**，不依赖 `last-checkpoint`。失败现场保留在 `outputs/qlora-v2/FAILED-134204-lastcheckpoint-bug/`（未当作结果使用）。

---

## 5. 推理与评测命令（基座 / 微调同口径）

```bash
cd ocr_module
PY=../.venv/Scripts/python.exe     # 本机；换机器改成自己的解释器

# 0) 校验数据版本与划分
$PY scripts/data_version.py

# 1) 基座（无 adapter）
$PY scripts/infer_test_baseline.py \
  --manifest datasets/handwriting_notes/splits/test.jsonl \
  --output outputs/eval_v3/baseline/test_pred_v3.jsonl \
  --max-pixels 1024000 --max-new-tokens 1024

# 2) 微调（checkpoint-26）
$PY scripts/infer_test_baseline.py \
  --manifest datasets/handwriting_notes/splits/test.jsonl \
  --output outputs/eval_v3/finetuned/test_pred_v3.jsonl \
  --adapter outputs/qlora-v2/v2data-20261115-134816/checkpoint-26 \
  --max-pixels 1024000 --max-new-tokens 1024

# 3) 同版本评估器（两侧同一份 evaluate.py）
$PY scripts/evaluate.py --pred outputs/eval_v3/baseline/test_pred_v3.jsonl \
  --ref datasets/handwriting_notes/splits/test.jsonl --out outputs/eval_v3/baseline
$PY scripts/evaluate.py --pred outputs/eval_v3/finetuned/test_pred_v3.jsonl \
  --ref datasets/handwriting_notes/splits/test.jsonl --out outputs/eval_v3/finetuned

# 4) 对比表 + 含公式体补充指标
$PY scripts/compare_runs.py \
  --base-pred outputs/eval_v3/baseline/test_pred_v3.jsonl \
  --base-metrics outputs/eval_v3/baseline/metrics.json \
  --ft-pred outputs/eval_v3/finetuned/test_pred_v3.jsonl \
  --ft-metrics outputs/eval_v3/finetuned/metrics.json
$PY scripts/eval_supplement.py

# 5) 当前入口单页冒烟（验证页 235955，两臂）
$PY scripts/infer_test_baseline.py --manifest outputs/smoke_v3/manifest_235955.jsonl \
  --output outputs/smoke_v3/base_235955.jsonl --max-pixels 1024000 --max-new-tokens 1024
$PY scripts/infer_test_baseline.py --manifest outputs/smoke_v3/manifest_235955.jsonl \
  --output outputs/smoke_v3/ft_235955.jsonl \
  --adapter outputs/qlora-v2/v2data-20261115-134816/checkpoint-26 \
  --max-pixels 1024000 --max-new-tokens 1024
$PY scripts/render_smoke_v3.py        # 读上面两个 jsonl，生成 smoke_v3.json / smoke_v3.md

# 6) adapter 权重校验
$PY scripts/verify_adapter.py --adapter outputs/qlora-v2/v2data-20261115-134816/checkpoint-26

# 7) 单测（默认 Temp 目录可能被拒，指定工作区内 basetemp）
$PY -m pytest tests -q -p no:cacheprovider --basetemp=.pytest_tmp
```

说明：
- `infer_test_baseline.py` 会在输出旁写 `.protocol.json` sidecar（运行时指纹）；若已有文件且指纹不同，脚本**拒绝覆写**——换配置请换输出路径，不要拼接不同配置的预测。
- 贪心解码、`max_new_tokens=1024`、无 system prompt、显式 `pad_token_id`、`torch.no_grad()`。
- 上限 1024 的来源是"非测试页最长参考答案 611 token"（`outputs/protocol/supervised_tokens.json`），**只用参考长度、不看测试成绩**。

---

## 6. 真实运行配置与结果文件

| 类别 | 路径 |
|---|---|
| 当前交付协议（checkpoint-26） | `inference_protocol_v3_delivery.json` |
| 历史协议（冻结于重训前，记录 checkpoint-4，原样保留） | `inference_protocol_v2.json` |
| 推理诊断 | `inference_diagnosis_v2.md` |
| 基座 5 页预测 + sidecar | `outputs/eval_v3/baseline/test_pred_v3.jsonl`(+`.protocol.json`) |
| 微调 5 页预测 + sidecar | `outputs/eval_v3/finetuned/test_pred_v3.jsonl`(+`.protocol.json`) |
| 逐页指标与错误明细 | `outputs/eval_v3/{baseline,finetuned}/metrics.json`、`err_*.md` |
| 对比 | `outputs/eval_v3/comparison_v3.json` / `.md` |
| 含公式体补充指标 | `outputs/eval_v3/supplement.json` |
| 复现性（同页两次重跑） | `outputs/eval_v3/repro_check/` |
| 当前入口冒烟（两臂） | `outputs/smoke_v3/`（json / md / 两臂 jsonl + sidecar） |
| 训练 run（实参/日志/4 ckpt） | `outputs/qlora-v2/v2data-20261115-134816/`（**权重与 optimizer 未入库**，见第 7 节） |
| adapter 校验 | `outputs/protocol/adapter_check_v2data_ckpt26.json` |
| 训练 smoke（显存探测） | `outputs/protocol/train_smoke_v2data.json` |
| 历史产物（未改动） | `outputs/qlora-v1/`、`outputs/baseline/`、`outputs/finetuned/` |

显存口径提醒：`memory(GiB)`（ms-swift 进程侧统计）与 `peak_cuda_allocated`（torch 分配器）都**不是物理显存**；`nvidia-smi memory.used` 是整卡占用（含桌面等其它进程）。本卡物理 8188 MiB，出现过的 8.40 GiB / 13010 MiB 都不能读作"需求"。

**哪些东西只在本地、没有入库**（不要以为 clone 下来就全都有）：
- `outputs/qlora-v1/`、`outputs/qlora-v2/` 下的权重与 optimizer（`*.safetensors`/`*.pt` 被忽略）；
- `outputs/modelscope_cache/`（基座模型缓存）、`outputs/baseline/`、`outputs/finetuned/`（旧 4096 尺度预测）、`outputs/eval_v3/crops/`（4.3 MB 错误分析裁切图）、`outputs/splits_backup_v1/`；
- `outputs/smoke_v3/*.log`（根 `.gitignore` 忽略 `*.log`）；
- `data/`（早期临时数据与远程标注对照）。
入库的是：代码、脚本、测试、标注与划分、协议、报告，以及 `outputs/` 下的小型文字产物（预测 jsonl、metrics、comparison、supplement、diagnosis、protocol 证据）。

---

## 7. Checkpoint 获取方式

**本次交付未上传权重。** 详见 `CHECKPOINT_MANIFEST.md`：

- 交付必需部分：`adapter_model.safetensors`（21,700,560 B，`a2be86a210a107e7db63c61e3bd180f0c4be7f46693fb2ffb2910ebc1f9abbb6`）+ `adapter_config.json`；
- 获取途径三选一：① 向交付方索要这两个文件；② 按第 4.1 节命令**自行重训**（数据与脚本齐全，52 步约 4 分钟）；③ 待 owner 确认后再配置 Git LFS 或私有模型仓库上传；
- 拿到后务必校验 SHA-256，并跑 `scripts/verify_adapter.py`（期望 `weights_verified=true`、372/372、`max_abs_diff=0.0`）；
- 基座 `SeerRay-Lab/Xiaomi-OCR-0` 需另行获取（ModelScope / HuggingFace 同名模型），LoRA 只能挂同一基座。

---

## 8. 当前局限（不要越过这些去下结论）

1. **3 个可评分页**（5 页测试集减 2 页含 `□` 排除）。样本量不支持任何泛化声明。
2. **测试集已被多次观察**：本项目的 `test.jsonl`（000020/000028/000037/000044/000051）在开发过程中被反复用于诊断与错误分析，**它已经不再是"从未见过的测试集"**。任何在该集上挑设置/挑 checkpoint 的做法都会污染结论；本轮虽未用它选 checkpoint（只按 val loss），但"没有污染"这一前提已不可恢复，下一轮必须补独立测试页。
3. **表格能力未被测试**：3 页参考里 0 张表，"表格数匹配 3/3"只说明不瞎编表格。
4. **公式能力未被证实**：只有"提取到/没提取到"；基座 3 页 `n_hyp=0` 记为"未提取到格式化公式"（既不证明对也不证明全错），微调提取到但 EM=0。
5. **上限 1024 的依据是参考长度，不是输出长度**：验证页 235955 两臂实际都命中 1024 并退化（`outputs/smoke_v3/`）。测试集恰好未触顶，不能外推。
6. **重复检测只是下限**：235955 微调臂触顶却未触发四条规则，说明规则抓不到"格式工整但内容全错"。
7. **可复现性表述收紧**：该页在记录的两次运行中输出逐字节一致（235955 基座 sha256 `f5095b19da0112d0…`；000037 微调两次 sha256 相同）。这不等于"贪心解码是确定性的"——换输入/GPU/驱动/库版本/批处理都可能改变数值路径。
8. 旧 `smoke_report.md` / `smoke_result.json` 是 4096 尺度的早期冒烟，**已被 `outputs/smoke_v3/` 取代**，仅作历史记录。

---

## 9. 下一轮实验安排（本次不执行）

> 训练侧的需求、边界、配置与本机核验见 **`training_plan_v4.md`**（含"当前 ckpt-26 已是最佳、同配置重跑只复现"的判定）。本节只描述实验因子。

**阶段 A：只在固定验证集（5 页）上做参数对比，不碰测试集。**

1. 因子一 —— 生成上限：`max_new_tokens` 1024 vs 2048（其余完全不变），逐页记录
   - 是否仍触顶、生成 token 数、`finish_reason`；
   - **区分"重复被延长"与"内容被恢复"**：触顶页要人工看新增部分是新内容还是同一模式的延长，光看长度增长不算恢复。
2. 因子二 —— 精简提示词：现行「请转写这页手写课程笔记为结构化 Markdown（保留标题、列表、表格与公式）。」vs 更短的纯转写指令（去掉"表格"等易诱发编造的词，并显式约束"只输出图中存在的标题"）。
   - 观察 235955 上 `## 山东大学` 这类臆造标题是否消失；
   - 注意提示词一改就必须换实验目录，**不得与旧预测拼接**。
3. 一次只改一个因子、GPU 串行、同进程监控；短暂无输出不等于崩溃，不要重启动。

**阶段 B：冻结设置后再做正式测试。**

4. 阶段 A 选定的设置写进新的协议文件（v4），双方（基座/微调）用**同一份新协议**重跑；
5. **补新的独立测试页**（当前 5 页已被反复观察）。建议按同样的复核标准标注 8–15 页从未参与任何诊断/调参的真实页，形成新的 test2 集合；旧 test 保留为"开发集"并在报告中明确标注；
6. 指标仍要同时报"正文 CER"与"含公式体 CER"，并报告可评分页数与每个排除原因。

**暂不做**：扩大训练数据、换更大模型、改 LoRA 结构。当前瓶颈是"密集手写页会退化"，不是容量不足的证据。

---

## 10. 隐私与上传状态（需 owner 确认）

- 本仓库为 **PUBLIC**。`datasets/handwriting_notes/images/` 的 23 张手写原图（约 103 MB）**已在更早的提交 `d046366` 中进入该公开仓库**（非 Git LFS）。这是本次交付之前既存的事实，本轮没有再新增任何原图或 PDF。
- 这些页面是个人课程笔记（含姓名/学号类信息未逐页核查）。**若要继续公开或转公开镜像，请 owner 明确确认**；如需撤回，需要重写历史（`git filter-repo`）并强推，属于破坏性操作，本次未做。
- 标注文件 `annotation_template.jsonl` 是原图的转写结果，与原图同等敏感；`data/annotations/real.jsonl`（标注 bot 版本）同样已在公开仓库中，敏感度相同。
- 未上传：模型权重（见第 7 节）、`outputs/eval_v3/crops/`（4.3 MB 裁切图，本地证据，可用于复核，未入库）、`outputs/modelscope_cache/`、`.venv/`、任何缓存。

---

## 11. 复现检查清单

- [ ] `python scripts/data_version.py` 通过，annotation sha256 = `34f262c2…`，13/5/5 互斥
- [ ] `sha256sum datasets/handwriting_notes/splits/test.jsonl` = `67a5df07…`
- [ ] `scripts/verify_adapter.py` 返回 `weights_verified=true`
- [ ] 基座 5 页指标 = 宏 CER 1.4872、字符加权 1.4386
- [ ] 微调 5 页指标 = 宏 CER 0.7725、字符加权 0.7433
- [ ] `supplement.json` = 1.1217 / 1.7952（含公式体宏平均）
- [ ] `comparison_v3.json` 的失败率 = 0.6667（两侧）
- [ ] `pytest tests` 全绿（**42 passed**，需 `--basetemp`；2026-11-24 新增 4 项指令覆盖测试后从 33 变为 42）

---

## 12. 2026-11-24 交付：重训复现 + 阶段A（上限 / 提示词）

> 本节的原始数据全部未入库（在 `outputs/` 下）；入库的是报告、代码与计划。

### 12.1 提交链（`feat/ocr-xiaomi-qlora`，远端 head `5a2ef4b`）

| 提交 | 内容 |
|---|---|
| `7f97876` | `training_plan_v4.md`：需求 / 边界 / 配置 / 本机核验 |
| `cc06528` | 重训复现记录（见 12.2） |
| `e55359f` | `phase_a_cap_report_v4.md`：因子一，上限 1024 vs 2048 |
| `3c1692d` | 指令可覆盖（`--instruction`）+ 4 项单测 + 因子二 A 报告 |
| `5a2ef4b` | 因子二扩到 A/B 三条指令并收口 |

### 12.2 重训（复现性验证，不是新结果）

- run：`outputs/qlora-v2/v2data-20261124-124959`，52 步 / **147.5 s** / `train_loss` 0.7831；val 1.208→**1.189**→1.189→1.191，best = ckpt-26。
- `checkpoint-26/adapter_model.safetensors` 与 11-15 那次 **sha256 逐字节相同**（`a2be86a2…`，21,700,560 B）；同 run 的 13/39/52 哈希互不相同，排除拷贝。
- adapter 校验 372/372、`max_abs_diff=0`、`weights_verified=true`。
- 结论限定：该机器 + 该依赖版本 + 该 seed 下可逐字节复现；不外推为"训练确定性"。

### 12.3 因子一：上限 1024 vs 2048（val 5 页）

- 5 页中 3 页两档**逐字节相同**（都自然 eos）；唯一自然结束的 235955 微调臂，追加的 323 字是**同一套臆造模板续写**，含公式 CER 反而 0.830→0.844。
- 000005 微调臂：2048 仍触顶，追加 1491 字纯模式复制，CER 0.888→0.943，耗时 77.7→160.6 s。
- **结论：上限不是瓶颈；2048 不设为默认。**

### 12.4 因子二：冻结 / A / B 三条指令（val 5 页，cap 1024）

| 指标（微调臂） | 冻结 | A | B |
|---|---|---|---|
| 宏 正文 CER | 1.248 | 1.086 | **1.017** |
| 宏 含公式 CER | **0.775** | 1.085 | 1.041 |
| 字数/参考（均值） | 1.64 | 0.68 | 0.70 |
| 小标题总数 / 重复条目 | 139 / 26.0% | 7 / 0.9% | 7 / 0.9% |
| 逐页胜（正文 CER） | **3 页** | 2 页 | 0 页 |

- **臆造模板由提示词触发**（两次复现）：`## 山东大学`、`1.输入/2.输出/3~5.处理` 全部消失。
- 但 A/B 都以**欠生成 + 公式丢失**为代价；B 的"不要省略"对微调臂几乎无效（字数比 0.68→0.70，235955 更短），只对基座臂有效（宏 CER 0.915→0.857，触顶页 3→1）。
- **结论：三条指令之间是混合变化，A/B 都不能成为默认；提示词这条线到此为止。**

### 12.5 待对方安排（按优先级）

1. **解码侧**：重复惩罚 / `min_new_tokens` / 按页自适应上限，冻结成协议后在 val 对比——无需重训，直接针对"过度重复 + 欠生成"。
2. **训练侧**：8 epoch（104 步）或扩训练页，重训后重走 adapter 校验。
3. **补独立测试页**：按 v2 标准标注 8–15 页从未参与调参的页（解决 5 页样本过小）。
4. **数据治理**：是否统一行尾使 sha256 跨平台可复现（会改 `splits/*` 字节）；权重是否托管（当前 adapter 21.7 MB 不入库）。

### 12.6 边界重申

本轮未碰 test、未改默认指令、未改协议文件、未上传权重；所有实验产物都在 Git 忽略的 `outputs/` 下。

---

## 13. 因子三：解码侧（重复惩罚 1.1 + `min_new_tokens` 256）

> 远程侧 2026-11-24 决策：PR #1 先合入 main（`5c8721a`），解码侧作为新实验在独立分支继续，出结果后另开 PR。本节即该实验结果。
> 分支 `feat/ocr-decode-params`（基于 `origin/main`），产物 `outputs/phaseA/decode/`（未入库）。

### 13.1 代码支撑

- `ocr/engine.py` 新增 `repetition_penalty`（默认 1.0）与 `min_new_tokens`（默认 0）；抽出 `_generation_kwargs()`，**默认值不传参**，历史数值路径不变；`describe()`、`last_run['generation']` 均记录两者。
- `scripts/infer_test_baseline.py` 新增 `--repetition-penalty` / `--min-new-tokens`，并写入 sidecar `fingerprint`。
- 新增 `tests/test_decoding_knobs.py`（4 项）：默认不传参、显式传参、1.0 视为未设置、describe 记录。`pytest` → **46 passed**。

### 13.2 配置

冻结指令、`max_pixels=1024000`、`max_new_tokens=1024`、`do_sample=False`，唯一变化是 `repetition_penalty=1.1` + `min_new_tokens=256`（**两者同时改，属混杂**，见 13.5）。

### 13.3 结果（val 5 页，5 页全可评分）

微调臂逐页：

| 页 | 冻结 tok/字数/CER | 解码 tok/字数/CER | ΔCER | 冻结 CERmath | 解码 CERmath |
|---|---|---|---|---|---|
| 235932 | 464 / 745 / 0.624 | 506 / 949 / 0.691 | +0.067 | 0.411 | 0.582 |
| 235946 | 448 / 650 / 0.962 | 358 / 573 / **0.722** | −0.240 | 0.962 | 0.722 |
| 235955 | **1024** / 1486 / 2.981 | 711 / 1247 / **0.921** | **−2.060** | 0.830 | 0.917 |
| 000005 | **1024** / 1523 / 0.888 | 323 / 525 / 0.903 | +0.014 | 0.888 | 0.903 |
| 000012 | 639 / 1055 / 0.785 | 335 / 555 / 0.888 | +0.103 | 0.785 | 0.876 |

宏观：

| 臂 | 正文宏 CER | 含公式宏 CER | 字数/参考 | 小标题 | 重复条目 | 触顶页 | 判重复页 | 耗时 |
|---|---|---|---|---|---|---|---|---|
| 微调·冻结 | 1.248 | **0.775** | 1.64 | 139 | 26.0% | 2 | 2 | 275 s |
| **微调·解码** | **0.825** | 0.800 | **1.07** | 9 | **0.0%** | **0** | **0** | **195 s** |
| 基座·冻结 | 15.955 | 16.139 | 1.00 | 0 | 0% | 1 | 1 | 109 s |
| 基座·解码 | 1.181 | 1.217 | 0.81 | 0 | 0% | 0 | 1 | 89 s |

### 13.4 判读

1. **解码侧是目前唯一同时压住两个病灶的设置**：正文宏 CER 1.248→0.825（−0.42），重复条目 26%→0%，触顶 2→0，字数比 1.64→1.07（既不虚长也不欠生成），且更快（275→195 s）。
2. **逐页胜负仍是冻结 3 : 解码 2**，但解码赢的两页是大幅（−2.060、−0.240），输的三页是小幅（+0.067、+0.014、+0.103）。**看宏平均会高估、看胜负数会低估**，两者必须同报。
3. **含公式宏 CER 略降**（0.775→0.800）：抑制重复会压掉部分公式，与因子二同一枚硬币的两面。
4. **基座臂是"去掉灾难、换来普遍中等"**：冻结节 000005 只有 45 token（CER 72.7），解码把它拉到 270 token，但 235932（0.544→0.915）与 000012（0.753→1.311）变差。基座宏平均同样不可单独引用。

### 13.5 局限（必须与结论一起引用）

1. **两个参数同时改**，违反"一次一个因子"：无法区分是重复惩罚抑制了循环，还是最小长度把早停页强行拉长。**下一步必须做消融**（只 rp 1.1 / 只 min 256）。
2. 5 页样本，235955 与 000005 两页主导差异。
3. `min_new_tokens=256` 会强制短页继续写，新增内容的正确性**未逐字核对**。
4. 因此**不由此改默认协议**：先消融，再决定是否写 `inference_protocol_v4.json`。

### 13.6 待办

1. 消融两臂（只 rp / 只 min），确认增益来源；
2. 消融确认后冻结协议 v4；
3. 行尾归一化（单独提交，已按远程决策执行）；
4. adapter 托管（`CHECKPOINT_MANIFEST.md` 已补 SHA-256 与下载路径说明，上传待凭据）；
5. 阶段B：新测试页 + 正式评测。

---

## 14. 解码侧消融：增益到底来自哪个旋钮（2026-11-25）

> 目的：§13 的组合是"rp 1.1 + min 256"同时改，违反一次一因子。本节把两档拆开，回答"重复被压住"与"输出被拉长"分别是谁的功劳。
> 设置、清单、上限、指令与 §13 完全一致；新增四臂 `rp_only`（只 rp 1.1）/ `min_only`（只 min 256），产物在 `outputs/phaseA/decode/`。

### 14.1 微调臂（5 页）

| 配置 | 正文宏 CER | 含公式宏 CER | 字数/参考 | 小标题 | 重复条目 | 触顶 | 判重复 | 耗时 |
|---|---|---|---|---|---|---|---|---|
| 冻结 | 1.248 | **0.775** | 1.64 | 139 | 26.0% | 2 | 2 | 275 s |
| **只 rp 1.1** | 0.875 | 0.854 | **1.01** | 9 | 0% | 0 | 0 | 180 s |
| 只 min 256 | 1.248 | **0.775** | 1.64 | 139 | 26.0% | 2 | 2 | 300 s |
| 组合 | **0.825** | 0.800 | 1.07 | 9 | 0% | 0 | 0 | 195 s |

**逐字比对（关键证据）**：

- `min_only` 在**全部 5 页**与冻结**逐字节相同** → `min_new_tokens=256` 在微调臂上**完全是空操作**（该臂不加上限时没有任何一页会在 256 token 前停）。
- `rp_only` 在 **4/5 页**与组合**逐字节相同**；唯一不同的是 000005（组合多 169 字，因为 rp 单独会在 217 token 早停，min 把它托到 323 token）。
- 因此：**微调臂的收益几乎全部来自重复惩罚**，`min` 只在"rp 让模型早停"的那一页起了兜底作用。

### 14.2 基座臂（5 页）

| 配置 | 正文宏 CER | 含公式宏 CER | 字数/参考 | 触顶 | 判重复 |
|---|---|---|---|---|---|
| 冻结 | 15.955 | 16.139 | 1.00 | 1 | 1 |
| **只 rp 1.1** | 2.747 | 3.133 | **0.51** | 0 | 0 |
| 只 min 256 | 1.688 | 1.677 | 1.14 | 1 | 1 |
| 组合 | **1.181** | 1.217 | 0.81 | 0 | 1 |

**基座臂的交互完全相反**：`rp` 单独会把输出压塌（235955 只剩 **65 token / CER 8.533**，235932、000012 同样过短），`min` 单独除了把 000005 的 45-token 崩坏托到 260 token 外基本等于冻结；**只有组合可用**——即 `min` 在这里是 `rp` 的安全地板。

### 14.3 结论

1. **微调臂的增益归因清楚：重复惩罚是主因，`min_new_tokens` 是兜底**。`rp_only` 已经把正文宏 CER 从 1.248 降到 0.875、重复条目清零、字数比拉回 1.01；组合只在 000005 一页再改善（1.153→0.903）。
2. **`min=256` 对微调臂能造成风险却不带来收益**：它逐字节等于冻结（4/5 页不受影响），却会在模型"正常早停"时强行续写；而它的唯一正收益是防 `rp` 早停。若采用 `rp`，建议把 `min` 视为可选项而不是默认。
3. **基座臂不能只用 `rp`**：会塌。基座若要接入，必须 `rp + min` 一起用。
4. **仍不改默认协议**：`rp=1.1` 的证据只覆盖 5 页、且"重复惩罚会让部分公式被压掉"（含公式 CER 0.775→0.854）没有解决。建议在冻结 v4 前先决定：是接受公式维度的小幅退让，还是继续调 `rp` 档位（如 1.05）。

### 14.4 复现

```powershell
cd ocr_module
$env:OCR_BASE_MODEL = "<本地 ModelScope 快照>"
# 只 rp
python scripts/infer_test_baseline.py --manifest datasets/handwriting_notes/splits/val.jsonl `
  --output outputs/phaseA/decode/rp_only_ft/val_pred.jsonl `
  --adapter outputs/qlora-v2/v2data-20261124-124959/checkpoint-26 `
  --max-new-tokens 1024 --repetition-penalty 1.1
# 只 min
python scripts/infer_test_baseline.py --manifest datasets/handwriting_notes/splits/val.jsonl `
  --output outputs/phaseA/decode/min_only_ft/val_pred.jsonl `
  --adapter outputs/qlora-v2/v2data-20261124-124959/checkpoint-26 `
  --max-new-tokens 1024 --min-new-tokens 256
```

数字复核：`E:\ocr\_analyze_ablation.py`（四配置逐页指标 + 逐字比对），临时脚本未入库。

---

## 15. `rp` 档位扫描：1.05 vs 1.1（2026-11-25）

> 目的：§14.3 指出 `rp=1.1` 让含公式 CER 从 0.775 退到 0.854，因此试更温和的 1.05，看能否既压住重复又少压公式。
> 对比方式（同旋钮比同旋钮）：微调臂 `rp_only` 1.05 vs 1.1（该臂 min 为空操作）；基座臂 `rp+min256` 1.05 vs 1.1（该臂必须带 min）。

### 15.1 微调臂

| 配置 | 正文宏 CER | 含公式宏 CER | 字数/参考 | 小标题 | 重复条目 | 触顶 | 判重复 | 耗时 |
|---|---|---|---|---|---|---|---|---|
| 冻结 | 1.248 | **0.775** | 1.64 | 139 | 26.0% | 2 | 2 | 275 s |
| **rp 1.1** | **0.875** | 0.854 | **1.01** | **9** | 0% | 0 | 0 | 180 s |
| rp 1.05 | 1.064 | 0.810 | 1.11 | 27 | 0% | 0 | 1 | 208 s |

逐页（正文 CER）：235932 `0.624 → 0.691 (1.1) → 1.115 (1.05)`；235946 `0.962 → 0.722 → 0.893`；235955 `2.981 → 0.921 → 1.442`；000005 `0.888 → 1.153 → 1.099`；000012 `0.785 → 0.888 → 0.772`。

- 1.05 **确实把公式指标拉回**（0.854→0.810，接近冻结的 0.775），但**正文 CER 退到 1.064**（比 1.1 差 0.19），且小标题从 9 回到 27（模板化内容回流）、235946 重新被判重复。
- 也就是说：**1.05 不是"更保守但同样有效"，而是换了一种退化方式**。

### 15.2 基座臂

| 配置 | 正文宏 CER | 含公式宏 CER | 字数/参考 | 触顶 |
|---|---|---|---|---|
| 冻结 | 15.955 | 16.139 | 1.00 | 1 |
| rp 1.1 + min | 1.181 | 1.217 | 0.81 | 0 |
| **rp 1.05 + min** | **0.989** | **1.029** | **1.14** | 1 |

基座臂上 1.05 反而更好（两项指标都升），但代价是 000005 又触顶（1024 token、判重复）。基座只作参照，不作为交付路径。

### 15.3 结论（v4 建议）

1. **交付路径（微调臂）用 `rp=1.1`**：正文 CER 0.875 vs 1.064、字数比 1.01 vs 1.11、小标题 9 vs 27，1.1 明显更稳；公式指标的 0.044 差距不足以抵消 0.19 的正文差距。
2. **`rp=1.05` 判定为不采纳**，但保留记录：它证明"公式退让"可以用更弱的惩罚换回来，只是代价转移到了正文与模板化上。
3. **对比协议必须两臂同参**：基座臂离不开 `min=256`（否则 rp 会塌），因此**基座 vs 微调的对比必须用 `rp1.1 + min256`**；只在纯微调交付时才可以去掉 `min`。这条决定 v4 需要两套 profile，而不是一套。
4. 因此 v4 的形态建议：
   - `compare_v4`（评测/对比）：`rp=1.1`、`min_new_tokens=256`、`max_new_tokens=1024`、`max_pixels=1024000`、冻结指令；
   - `deliver_v4`（微调侧生产）：同上但 `min_new_tokens=0`。

### 15.4 复现

```powershell
python scripts/infer_test_baseline.py --manifest datasets/handwriting_notes/splits/val.jsonl `
  --output outputs/phaseA/decode/rp105_ft/val_pred.jsonl `
  --adapter outputs/qlora-v2/v2data-20261124-124959/checkpoint-26 `
  --max-new-tokens 1024 --repetition-penalty 1.05
```

数字复核：`E:\ocr\_analyze_rpsweep.py`（同旋钮对比 + 逐页），临时脚本未入库。

---

## 16. 协议 v4 冻结（2026-11-25，远程已批准）

### 16.1 两套 profile

| profile | `repetition_penalty` | `min_new_tokens` | `max_new_tokens` | `max_pixels` | 用途 |
|---|---|---|---|---|---|
| `compare_v4` | 1.1 | **256** | 1024 | 1024000 | 基座 vs 微调的任何对比（两臂同参） |
| `deliver_v4` | 1.1 | **0（不传）** | 1024 | 1024000 | 微调侧生产（单臂） |

冻结在 `inference_protocol_v4.json`；`ocr/engine.py` 的 `DEFAULT_REPETITION_PENALTY = 1.1`、`DEFAULT_MIN_NEW_TOKENS = 0` 即 `deliver_v4`，`DEFAULT_MAX_NEW_TOKENS/MAX_PIXELS` 沿用。

### 16.2 为什么是两套而不是一套

- **对比必须两臂同参**：`rp` 单独会把基座臂压塌（§14.2：235955 只剩 65 token、CER 8.533），基座离不开 `min=256` 这层地板。
- **生产不需要这层地板**：微调臂上 `min=256` 与冻结协议逐字节相同（§14.1，5/5 页），却会在模型正常早停时强行续写；它的唯一正收益是防 `rp` 早停。

### 16.3 v3 仍然可复现

引擎只有在 `repetition_penalty != 1.0` 时才把该参数交给 `generate`，所以 **显式传 `repetition_penalty=1.0` 的数值路径与 v3 逐字节一致**：

```powershell
python scripts/infer_test_baseline.py --manifest datasets/handwriting_notes/splits/val.jsonl `
  --output <out> [--adapter <ckpt-26>] --repetition-penalty 1.0
```

`scripts/infer_test_baseline.py` 的两个解码参数默认改为 `None`（省略即用引擎默认），因此 CLI 不再硬编码 1.0/0 而掩盖协议版本。

### 16.4 防漂移

`tests/test_decoding_knobs.py` 新增两项：读 `inference_protocol_v4.json` 与 `engine.DEFAULT_*` 逐项比对（不一致即失败），以及协议里的指令文本/哈希必须等于 `ocr/prompts.py::INSTRUCTION`。当前 `pytest` → **48 passed**。

### 16.5 仍未解决（写进协议 known_limits）

1. 证据只覆盖固定验证集 5 页；test 未参与、也未用于选参。
2. 含公式宏 CER 从 0.775 退到 0.854（`deliver_v4`）——抑制重复会压掉部分公式，报告必须同时给两列。
3. `min_new_tokens=256` 会强制短页继续写，稀疏页风险未单独验证。
4. 5 页样本下宏平均方差大，逐页胜负必须同报（`deliver_v4` 对冻结是 2 胜 3 负，但赢的幅度远大于输的幅度）。
5. 阶段B 仍缺 8–15 页从未参与调参的独立测试页。
