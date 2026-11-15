# 最终验收报告 v2：Xiaomi-OCR-0 手写笔记 OCR — 修复、标注、训练与评测

- 生成时间：2026-11-15
- 工作目录（下文所有相对路径以此为根）：`E:\ocr\wheatfield_braintrust\ocr_module`
- 本文件取代此前结论；`final_comparison_report.md` 保留为**历史只读审计**，不再更新，其结论（旧 4096 尺度 / 512 上限 / v1 标签 / 4 步训练）已被本报告替代。
- 验收方式：先核查已有产物，**只对发现的缺口补做工作**；本阶段补做的是「当前代码状态下的单页冒烟（两臂）」与「单元测试复跑」，**未重训、未重评**。

---

## 0. 结论摘要

| 问题 | 结论 |
|---|---|
| 推理入口是否可用（图文消息、图像 token、UTF-8、Markdown） | **可用**。prompt 1008 token 中 972 个是 `image_pad`，两臂 prompt sha256 完全相同；输出 UTF-8 解码 0 个替换字符，Markdown 可读 |
| 单页冒烟（当前代码） | **通过但暴露新事实**：验证页 235955 上**两臂都命中 1024 上限**并退化；基座编造表头，微调编造标题/公式。冒烟能证明入口链路正确，**不能**证明生成质量 |
| 标注是否可信 | **23 页全部对照原图复核**，v2 修正 10 处 / 7 页，看不清一律 `□`（5 页 32 个），无凭空补写；`reviewed=true` 只表示"已复核"，**不是正确性证明** |
| 训练是否完成 | **完成**。4-bit QLoRA、52 步、4 个真实 checkpoint，验证集显式接入（`split_dataset_ratio=0`），选中 `checkpoint-26`（val eval_loss 1.18883 最低） |
| 基座 vs 微调是否有可比结论 | **有，但为"混合变化（mixed）"**：主指标 CER 1.4872 → 0.7725 改善；把公式体计入后 1.1217 → **1.7952** 反而更差。**不能**宣称"微调让 OCR 变好" |
| 证据强度 | 可评分页 **3 页**（5 页中 2 页因参考含 `□` 整页排除），失败率两臂同为 66.7%，公式 EM 与标题匹配率两臂同为 0.0 → **不支持泛化结论** |

一句话：**链路是真的、训练是真的、对比也是同口径的，但"微调更好"这个结论不成立**——它只是把一个失败模式（基座编造宽表）换成了另一个（微调把内容塞进重复公式 / 编造标题），而主指标因为会把 `$…$` 整段删除，系统性地偏袒后者。

---

## 1. 五项验收逐条核验

### 1.1 推理入口

| 检查项 | 证据 | 判定 |
|---|---|---|
| 图文消息格式 | `inference_protocol_v2.json → prompt.messages`：单 user 轮，`[{image},{text:"\n"+instruction}]` | 通过 |
| 是否真的存在图像 token | 冒烟两臂 `prompt_tokens=1008`，`n_image_pad_tokens=972`；`image_grid_thw=[[1,54,72]]` → 3888 patch ÷ 2² = 972 视觉 token（**patch 数与视觉 token 数分开记，未混称**） | 通过 |
| 训练/推理模板一致 | `prompt.alignment_proof`：训练侧渲染 1012 token、推理侧 1008 token，`prompt_identical=true`，差的 4 个 token 恰是"答案首字 + `<|im_end|>` + 换行"；assistant 头 = `<|im_start|>assistant\n<think>\n\n</think>\n\n`（空 think = 非思考模式） | 通过 |
| 旧流程问题 | 旧仓库推理流程带 SYSTEM_PROMPT，从第 1 个 token 起不一致且多 240 token → 已废弃 | 已修正 |
| UTF-8 / Markdown | `outputs/smoke_v3/smoke_v3.json`：两臂 `decode_ok=true`、`replacement_chars=0`、`readable_markdown=true` | 通过 |
| 后处理两侧一致 | 同一 `normalize_output`：只剥 1 层外层围栏（内部 ``` 数为奇数则不剥）+ `\[..\]`/`\(..\)` → `$$..$$`；raw 与归一化文本都落盘 | 通过 |

结论：入口本身没有问题。**"加载成功"没有被当作"配置正确"**——adapter 另有五项独立校验（见 1.4）。

### 1.2 单页冒烟（当前代码、两臂、同一页）

代码在本阶段确有改动（`ocr/engine.py`、`ocr/postprocess.py`），因此**旧冒烟不能单独证明当前入口有效**，故补做一次：

- 页面：`IMG_20261104_235955.jpg`（**验证集页**，非测试页，不参与任何选择）
- 产物：`outputs/smoke_v3/smoke_v3.json`、`smoke_v3.md`、`base_235955.jsonl(.protocol.json)`、`ft_235955.jsonl(.protocol.json)`、`base.log`、`ft.log`

| 项 | 基座 | 微调 checkpoint-26 |
|---|---|---|
| 原图 / sha256 前 16 位 | 4096×3072 / `373f55638b7b77e6` | 同（同一文件） |
| 处理器实际尺寸 | 1152×864 | 同 |
| `max_pixels` / grid | 1024000 / `[[1,54,72]]` | 同 |
| patch / 视觉 token | 3888 / 972 | 同 |
| prompt / image_pad | 1008 / 972 | 同 |
| prompt sha256 | `557b0658a32f1602…` | **同** |
| 生成 token / 上限 | **1024 / 1024** | **1024 / 1024** |
| 结束原因 | `length`（末 token 735，EOS=248046） | `length`（末 token 3） |
| 重复检测 | True（`unit_run`,`ngram`） | False（四规则均未触发） |
| 耗时 | 54.2 s | 96.7 s |
| 进程峰值 allocated / reserved | 989.2 / 1030.0 MiB | 1009.4 / 1050.0 MiB |
| 整卡占用 前→后（含其它进程） | [1713, 8188] → [1914, 8188] MiB | [1757, 8188] → [1936, 8188] MiB |
| markdown 字符 / 文件字节 | 1791 / 10518 | 1486 / 8237 |

逐项检查（参考 532 个纯文本字符、2 个标题、10 个列表项、8 个公式）：

| 元素 | 基座 | 微调 | 结论 |
|---|---|---|---|
| 文字 | 1024 字符，CER 1.9192 | 162 字符，CER 0.9079 | 都有输出、可读；**微调覆盖严重不足** |
| 标题 | 0 个（参考 2 个 `##`） | 6 个全为 `##` | 基座不输出标题；微调输出层级与内容都不对 |
| 列表 | 0 项 | 30 项（参考 10） | 基座无列表；微调过度列表化 |
| 公式 | 未提取到格式化公式（0） | 102 个（参考 8） | 基座公式被写进纯文本；微调大量臆造公式 |

**冒烟暴露的两个新事实（必须写进结论）：**

1. **1024 上限不足以覆盖最密的验证页**。上限是按"最长非测试参考答案 611 token"定的（`cap_decision`），但**参考答案长度 ≠ 模型输出长度**：微调把内容展开成列表 + 公式后，在 235955 上同样触顶（1486 字符 / 1024 token）。测试集 5 页恰好没有触顶，但这不能外推成"1024 永远够"。
2. **基座的退化形态是编造表头**：输出退化为 `<md> | 课程名称 | 课程代码 | 课程名称 | 课程代码 …` 反复循环；微调的退化形态是编造 `## 山东大学` 标题 + `输入/输出/转换/子集` 模板化条目 + `$S_{PC}\rightarrow$` 类臆造公式。**两者都不是"识别"，但都被计了分**（主指标把公式整段删掉后，微调剩下的 162 字碰巧 CER 更低）。

**可复现性旁证（限定表述）**：本次基座冒烟与几天前独立诊断运行在同一页（235955）上产出的 markdown **逐字节相同**（1791 字符，sha256 均为 `f5095b19da0112d0…`）。准确的说法是：**该页在记录的这两次运行中输出逐字节一致**。这**不**外推为"贪心解码是确定性的"——不同输入、不同 GPU/驱动/CUDA 版本、不同 transformers 或 PyTorch 版本、不同 batch/padding 都可能改变数值路径。
注意两次运行的**重复规则标记不同**（诊断记为 `template_line_run` 命中，冒烟记为 `unit_run`+`ngram`），因为检测器在这两次运行之间改过——所以只把"生成文本一致"当作复现性证据，不把规则标记当证据。微调侧另有 `outputs/eval_v3/repro_check/`（000037 两次运行 markdown sha256 相同，同样只说"这两次一致"）。

**冒烟的有效边界**：这一页只证明"入口链路正确、证据齐全"，**不证明**文字/公式/结构识别质量。用它当质量证据会是误用。

### 1.3 真实标注

| 检查项 | 证据 | 判定 |
|---|---|---|
| 23 页是否都有原图核查记录 | `annotation_review_v2.md` 第 2 节逐页表（23 行，含 train/val/test 归属与处置） | 通过 |
| 是否用 PDF 补写 | 复核仅用 `datasets/handwriting_notes/images/` 23 张原图；`pdf/` 与远程版 `data/annotations/real.jsonl` 均未参与修正 | 通过 |
| 看不清是否用 `□` | 5 页共 32 个 `□`，逐处列出（000020 整页 28 个、200839/200903/235900/000028 各 1） | 通过 |
| 是否保留原图错误 | 235955 保留 `SUB` 写作 `SRC+DST→DST`；000012 保留 JNB"高于"等原图不严谨处 | 通过 |
| 是否自造结构 | v1 在 235916 自造表头、在 235922 自造"段寄存器"层 → v2 已去除 | 已修正 |
| 划分与 Swift 数据一致 | `datasets/handwriting_notes/data_version_v2.json`：annotation sha256 `34f262c2…`；train/val/test = 13/5/5，`splits_disjoint=True`、`splits_cover_all_images=True`、`membership_unchanged_vs_v1=True`；`train_swift.jsonl`/`val_swift.jsonl` 与对应 split 页集合一致 | 通过 |
| 独立复核（本阶段） | 重算三份 split：13 / 5 / 5，两两交集为空，合计 23 页去重 23 | 通过 |

**必须一并声明的性质**：`reviewed=true` 只是流程标记，**不是正确性证明**。报告第 4 节自列"仍存疑"：200854「标称/有效标距」为中等置信、200907/200911/200914 竖排小字仅整体通读未逐字确认。人工/助手转写仍可能存在误差，尤其是 12MP 原图单字不足 3px 的页面。

### 1.4 训练可行性与数据隔离

| 检查项 | 证据（全部来自运行产物，非手写清单） | 判定 |
|---|---|---|
| Windows 可行性 | 真实跑通 52 步；曾因 `os.symlink` 权限失败一次（`PermissionError [WinError 5] '…\last-checkpoint.tmp'`），修复方式：run dir 预置同名空目录让 `update_last_checkpoint_symlink` 走早退分支，**不依赖符号链接**，一律用真实 `checkpoint-N` 目录 | 通过（含失败记录） |
| 依赖版本 | python 3.12.13 / torch 2.6.0+cu124 / transformers 5.14.1 / peft 0.20.0 / ms-swift 4.6.0 / bitsandbytes 0.50.2 / qwen-vl-utils 0.0.14 / RTX 4060 Laptop 8188 MiB | 通过 |
| 模型与 template | `model_type=qwen3_5`、`template=qwen3_5`（非猜测，取自 `args.json` 并有 `prompt_identical` 证明） | 通过 |
| 4-bit 实参 | `quant_method=bnb`、`quant_bits=4`、`nf4`、`double_quant=true`、`compute_dtype=bfloat16`、`torch_dtype=bfloat16` | 通过 |
| LoRA | r=8 / α=32 / dropout 0.05，`target=all-linear` 落地后正则仅 `model.language_model.*`，372 个模块、**0 个越界**；`freeze_vit=true`、`freeze_aligner=true`；可训练 5,411,328（0.8877%） | 通过 |
| 数据隔离 | `dataset=train_swift.jsonl`(13)、`val_dataset=val_swift.jsonl`(5) **显式接入**、`split_dataset_ratio=0`（禁止自动切分）；`synthetic.count=0`（本轮无合成样本，策略仍为"合成只进 train"） | 通过 |
| checkpoint 实际保存 | `checkpoint-13/26/39/52` 四个真实目录（各含 `adapter_model.safetensors`、`optimizer.pt`、`trainer_state.json`）；日志 `best_model_checkpoint=checkpoint-26` | 通过 |
| checkpoint 可加载且权重对得上 | `outputs/protocol/adapter_check_v2data_ckpt26.json`：372/372 键匹配、`max_abs_diff=0.0`、`peft_loader_missing_lora_keys=0`、`lora_modules_outside_language_model=0`、`lora_B_all_zero=0`、`trainable_params=0`、`inference_uses_no_grad=true`、`weights_verified=true` | 通过 |

**训练记录的两个纠正（此前被误读过）：**

- `warmup_steps=0` 不等于没有 warmup。`warmup_ratio=0.03` × 52 步 ≈ 1.56 步，实际 LR 日志：**step1 = 5e-05、step2 = 1e-04**（峰值），随后余弦衰减至 step52 = 0.0。即以"真实 LR 日志"为准，不以参数默认值推断。
- `memory(GiB)=8.40` **不是物理显存**。本卡物理 8188 MiB = 8.0 GiB，8.40 已超物理上限 → WDDM 走共享内存；训练侧与推理侧（曾出现 `peak_cuda_allocated=13010 MiB`）均只能同机横向比较，不能当显存需求结论。

### 1.5 微调前后对比

| 检查项 | 证据 | 判定 |
|---|---|---|
| 同一参考集 | 双方都跑 `datasets/handwriting_notes/splits/test.jsonl`（5 页，sha256 `67a5df07…`），参考来自 v2 标注 `34f262c2…` | 通过 |
| 同一输入尺度 | 双方 `max_pixels=1024000`，grid 全部落在 `[1,54,72]`/`[1,72,54]` → 3888 patch / 972 视觉 token | 通过 |
| 同一提示词 | 无 system、单 user 轮；两臂 `instruction_sha256` 相同 | 通过 |
| 同一生成设置 | 贪心、`max_new_tokens=1024`、显式 `pad_token_id`、`torch.no_grad()` | 通过 |
| 同一后处理 / 评估器 | 同一 `normalize_output`；同一份 `scripts/evaluate.py`（本次仅加字段，未改指标定义） | 通过 |
| 双方都重跑完整 5 页 | `outputs/eval_v3/{baseline,finetuned}/test_pred_v3.jsonl` 各 5 条，逐页落盘 + 元数据 sidecar；旧 4096/512 预测保留在 `outputs/baseline/`、`outputs/finetuned/` 未改动 | 通过 |
| 可评分页与排除原因 | 5 页中 3 页可评分；2 页因参考含 `□`（`unknown_reference`）整页排除，**双方同规则**，排除页预测仍保留 | 通过 |
| checkpoint 与数据版本一致 | 训练用 v2 标注 + v2 split；评测参考同为 v2 标注（sha256 相同），运行 sidecar 记录 `adapter_weights_sha256=a2be86a2…` 与训练产物一致 | 通过 |

**关于 `inference_protocol_v2.json` 的 `adapter` 字段（已处理，未篡改历史）**：该文件冻结于重训之前，其 `adapter` 段落记录的是当时的 `checkpoint-4`（含该 checkpoint 的权重 sha256 与加载校验证据），**属于历史协议记录**，因此**原样保留、不修改**。

- 它不参与任何历史指纹计算：没有任何脚本读取该文件（`grep` 全仓确认，仅 `ocr/engine.py` 注释与测试 docstring 提到文件名）；每个运行目录的 `.protocol.json` sidecar 里的 `fingerprint` 由 `scripts/infer_test_baseline.py::fingerprint(args, engine_desc)` 在**运行时用实参**计算，不从该文件读取。所以不改它也不会让旧指纹失真，改它反而会让"旧指纹 ↔ 旧内容"的对应关系断裂。
- 本次交付的**实际配置**另记为 `inference_protocol_v3_delivery.json`（`adapter = checkpoint-26`，含权重 sha256 `a2be86a2…`、数据版本哈希、评测产物路径与指标摘要）。
- 实际评测以运行 sidecar 为准：`outputs/eval_v3/finetuned/test_pred_v3.jsonl.protocol.json` 记录 `adapter_dir=…/checkpoint-26`、`adapter_weights_verified=true`。

---

## 2. 23 页划分与标注版本

- 标注文件：`datasets/handwriting_notes/annotation_template.jsonl`，sha256 `34f262c270b2bf955a9d039b7d80f355c13e20bd87b8e7709c97201d19d17732`，23 页
- 上一版：`annotation_template.v1.bak.jsonl`，sha256 `b45d05ebd61432a0…`（保留可回滚）
- 逐处修正：`outputs/annotation_fixes_v2.json`（10 处 / 7 页）

| 集合 | 页数 | 页名 |
|---|---|---|
| train | 13 | 200839、200843、200854、200858、200903、200907、200911、200914、200917、235900、235908、235916、235922 |
| val | 5 | 235932、235946、235955、000005、000012 |
| test | 5 | 000020、000028、000037、000044、000051 |

- 互斥且覆盖全量：合计 23，去重 23，两两交集为空（本阶段重算确认）。
- 修正分布：train 4 页 7 处、val 1 页 1 处、test 2 页 2 处 —— 即**测试参考也被改过**，这一点必须在结论里体现（旧 checkpoint 与旧预测都不能代表 v2 参考）。
- 合成数据：`synthetic.count=0`，本轮未生成；策略仍为"合成只允许由 `prepare_dataset.py` 拼进 train，val/test 只用真实页"。

---

## 3. 模型、训练参数、checkpoint

| 项 | 值 |
|---|---|
| 基座 | `SeerRay-Lab/Xiaomi-OCR-0`（`Qwen3_5ForConditionalGeneration`，model_type `qwen3_5`）；HF 快照与 ModelScope 快照**逐文件 sha256 相同** |
| run dir | `outputs/qlora-v2/v2data-20261115-134816` |
| 量化 | bnb 4-bit NF4 + double quant，compute dtype bfloat16 |
| LoRA | r=8、α=32、dropout 0.05，仅语言模型 372 模块；ViT 与 aligner 冻结 |
| 优化 | lr 1e-4、cosine、warmup_ratio 0.03（实测 1–2 步）、batch 1、**grad_accum 1**、4 epoch = **52 步**、seed 42 |
| 长度 | `max_length=8192`、`truncation_strategy=delete`、`max_pixels=1024000` |
| 评估/保存 | `eval_strategy=epoch`、`save_strategy=epoch`、`save_total_limit=4` |
| 产出 | 238 s，train_loss 0.7831，4 个真实 checkpoint |
| **选中** | **`checkpoint-26`**（`outputs/qlora-v2/v2data-20261115-134816/checkpoint-26`，63 MB） |

验证曲线（**仅用于选 checkpoint，测试集全程未参与**）：

| ckpt | step | eval_loss | eval_token_acc |
|---|---|---|---|
| 13 | 13 | 1.20785 | 0.7691 |
| **26** | **26** | **1.18883** | **0.7781** |
| 39 | 39 | 1.18942 | 0.7747 |
| 52 | 52 | 1.19147 | 0.7773 |

**为什么重训（选 B）而不是沿用旧 checkpoint-4**：训练标签有 7 处已证实错误（其中 train 4 页 7 处、val 1 页），且旧跑 `grad_accum=16` + 13 样本 → 每 epoch 仅 1 步、4 epoch 共 4 步，属实现缺陷而非超参偏好。改动只有 `grad_accum 16 → 1`，其余超参与旧跑一致，避免夹带调参。旧 checkpoint-4 与 val loss 1.2456 保留为历史产物，**未冒充 v2 数据结果**。

---

## 4. 双方推理配置（冻结）

| 项 | 基座 | 微调 |
|---|---|---|
| adapter | 无（sidecar `adapter_dir=null`） | `checkpoint-26`，`weights_verified=true` |
| max_pixels | 1024000 | 1024000 |
| 实际输入尺寸 | 1152×864 等（全部非原图尺度） | 同 |
| 视觉 token | 972 / 页 | 972 / 页 |
| 提示词 | 无 system，`"\n"+INSTRUCTION` | 同（sha256 相同） |
| 生成 | greedy，max_new_tokens 1024，`torch.no_grad()` | 同 |
| 后处理 | `normalize_output`（剥 1 层包装 + 统一公式定界符） | 同 |
| 评估器 | `scripts/evaluate.py`（`□` 页整页排除） | 同 |

上限决策（`cap_decision`）：非测试页最长参考答案 611 token → 512 会截断正常页、1024 留约 1.6 倍余量。**该决策只用参考长度，未看测试成绩**。但如 1.2 所示，参考长度不保证覆盖模型输出长度。

---

## 5. 汇总指标与逐页结果

### 5.1 页数与排除

| 项 | 值 |
|---|---|
| 总页数 | 5 |
| 可评分页 | 3（000037、000044、000051） |
| 排除页 | 2（000020：28 个 `□`、整页失焦；000028：1 个 `□`、红字单位前缀不可辨） |
| 排除原因 | `unknown_reference`，双方同规则；预测已生成并落盘（000020: 139/152 token，000028: 417/648） |

### 5.2 主指标（冻结评估器，公式文本**不**计入）

| 指标 | 基座 | 微调 | 差值 |
|---|---|---|---|
| 宏平均 CER（每页等权） | 1.4872 | **0.7725** | −0.7147 |
| 字符加权 CER（Σ编辑距离/Σ参考字符） | 1.4386 | 0.7433 | −0.6953 |
| Σ编辑距离 / Σ参考字符 | 1289 / 896 | 666 / 896 | −623 / 0 |
| 公式精确匹配率 | 0.0 | 0.0 | 0 |
| 公式 CER | 1.0 | 3.0894 | +2.0894 |
| 标题序列匹配率 | 0.0 | 0.0 | 0 |
| 命中生成上限页数 | 0/3 | 0/3 | 0 |
| 检出重复页数 | 2/3 | 2/3 | 0 |
| 失败率（触上限或重复） | 66.7% | 66.7% | 0 |
| 总耗时 | 132.9 s | 231.8 s | +98.9 s（1.74×） |
| 单页峰值 allocated | 989.2 MiB | 1009.4 MiB | +20 |
| 参考中"未提取到格式化公式"的页数 | 3 | 0 | −3 |

CER > 100% 是正常的：分母只有 896 个参考字符，而基座大量插入内容（000037 输出 781 字对参考 355），插入与重复使编辑距离超过参考长度。

### 5.3 补充指标（把公式体计入文本）——**这一栏改变结论**

同一份编辑距离代码、同一排除规则，只把 `plain_text()` 删除的 `$…$`/`$$…$$` 内容保留：

| 指标 | 基座 | 微调 |
|---|---|---|
| 宏平均 CER（含公式体） | **1.1217** | **1.7952** |
| 字符加权 CER（含公式体） | 1.1125 | 1.9119 |

主指标的改善有相当部分是"内容被公式定界符吞掉"造成的（000044 微调输出 1986 字符里 1909 字符是重复公式，删掉后只剩 77 字符，于是 CER 看起来正常）。

### 5.4 逐页结果

| 页 | 参考字符 | 基座 CER | 微调 CER | Δ | 基座生成/结束/重复 | 微调生成/结束/重复 |
|---|---|---|---|---|---|---|
| 000020 | 排除 | — | — | — | 139 / eos / 否 | 152 / eos / 是 |
| 000028 | 排除 | — | — | — | 417 / eos / 否 | 648 / eos / 是 |
| 000037 | 355 | 1.7155 | **0.4732** | −1.2423 | 687 / eos / 是 | 307 / eos / 否 |
| 000044 | 329 | 0.8359 | 0.9149 | +0.0790 | 326 / eos / 是 | 880 / eos / 是 |
| 000051 | 212 | 1.9104 | 0.9292 | −0.9811 | 403 / eos / 是 | 344 / eos / 是 |

含公式体的逐页 CER：000037 1.3453→0.6102；000044 0.8228→**3.9599**；000051 1.1969→0.8154。

结构（参考 headings/lists/tables）：000037 = 2/10/**0**、000044 = 5/10/**0**、000051 = 4/11/**0**；基座输出 0/0/1、0/0/1、0/0/1；微调 5/12/0、23/0/0、1/0/0。

---

## 6. 代表性案例（对照原图，不只凭指标）

裁切图：`outputs/eval_v3/crops/`。

**000037（置位/复位 + 21. 8253）— 真实改善，但仍错。**
原图实读：`置位/复位`：`D7=0`、`D6D5 任意值`、`D3D2D1: 000~111 ⇒ PC0~PC7`、`D0: 1 SET / 0 RESET`；工作方式 6 条，`方式2(分频器)…(f出=f输入÷N初值)`。
基座把整页压成一张 `<md>| … |` 表、塞入上百个 `<br>`（所以 781 字符），`分频器→分电器`、`单稳触发器→单档继电器`。微调恢复"标题 + 列表"，`分频器`、`方波发生器` 读对；但整块"置位/复位"丢失、`D7D6`/`D5D4` 细节丢失、`方式1` 写成"单格计数器"、`GATE→CATE`、`方式2` 的公式被换成臆造的 `$1=\frac12+\frac12+\frac12$`。
→ CER 1.72→0.47 是真实改善（主要来自"不再编造大表 + 结构对了"），但**不叫识别正确**。

**000044（22. 8259 ICW1~ICW4）— 双方都失败，Δ 不能读成退化。**
基座输出与页面无关的编造表（`定义|IOW1|IOW2…`、`统计模型`、`P0,P0,P0`）；微调输出 `$$I_0 \rightarrow I_0 \rightarrow …$$` 约 100 次。主指标 0.9149 之所以"还行"，是因为重复全在公式里被删掉了。真正的信号是"主指标隐藏了一大段重复内容"。

**000051（OCW1~OCW3）— 双方都失败，微调崩得慢一点。**
基座退化成 `P1 P2 … P99` 计数；微调开头抓到 `OCU2=1 / A0=1`、`Di=1; D0 中断,被解构`（"被屏蔽/被开放"读成"被解构"），随后 `$D_0=0$;` 连续重复。1.91→0.93 反映"崩得没那么彻底"。

**235955（验证页，冒烟）— 两臂都退化，形态不同。**
基座：`<md>| 课程名称 | 课程代码 | …` 重复 1024 token；微调：`## 山东大学` + `输入/输出/转换/子集` 模板条目 + 102 个臆造公式，纯文本只剩 162 字符。这一页不进任何指标，但它说明**当前两臂在密集页面上都不能交付可用转写**。

---

## 7. 限制（必须与结论一起读）

1. **样本量**：可评分页 3 页（5 页测试集减 2 页排除），单页变化即可主导宏平均，不做显著性推断。**3 个可评分页不支持任何泛化结论**。
2. **2 页被整页排除**：因为参考含 `□`（`unknown_reference`）。排除规则在评测前定好、双方同规则、未按成绩删页；但代价是"最难的失焦页"没有客观分数。
3. **公式指标**：提取器只认 `$…$`/`$$…$$`（`\[..\]`/`\(..\)` 经归一化后才可提取）。基座 3 页 `n_hyp=0` → 记为"**未提取到格式化公式**"，既不证明正确也不证明全错；微调"有公式但 EM=0"，公式 CER 3.0894 > 1 是因为输出了大量参考中没有的公式。基座那一栏的 1.0 是"输出为空、全按删除计"，不是"好"。
4. **结构指标**：3 页参考**一张表都没有**。基座编造 3 张表、微调 0 张，因此"表格数匹配 3/3"只说明"不瞎编表格"，**不能说明具备表格识别能力**。标题/列表匹配率双方均为 0.0，说明结构输出仍不可用。
5. **截断**：测试集 5 页无一触顶（双方 0/3），所以"截断"不是本次失败原因；但**验证页 235955 两臂都触顶**，说明 1024 对密集页不足。命中上限本身不等于失败，本报告只把"触顶 + 重复/发散"计为失败。
6. **验证 loss 下降 ≠ OCR 效果改善**：eval_loss 1.20785→1.18883（token 级 teacher forcing）与 CER 是两套口径；本次正是"loss 降了、含公式 CER 反而升了"的反例。
7. **显存口径**：`memory(GiB)` 与 `peak_cuda_allocated` 都不是物理显存；`nvidia-smi memory.used` 是整卡占用（含桌面）。三者不可互换。
8. **重复检测是下限而非质量度量**：235955 微调臂触顶却"四规则均未触发"，说明规则只能抓机械重复，抓不到"结构工整但内容全错"。

---

## 8. 复现命令与产物路径

```bash
cd E:\ocr\wheatfield_braintrust\ocr_module
PY=..\.venv\Scripts\python.exe

# 0) 数据版本与划分校验
$PY scripts/data_version.py

# 1) 训练（唯一改动 grad_accum=1；Windows 用真实 checkpoint 目录）
powershell -File scripts/train_qlora_v2.ps1 -OutputRoot outputs/qlora-v2

# 2) 评测（双方同一协议、新目录）
$PY scripts/infer_test_baseline.py --manifest datasets/handwriting_notes/splits/test.jsonl \
    --output outputs/eval_v3/baseline/test_pred_v3.jsonl --max-pixels 1024000 --max-new-tokens 1024
$PY scripts/infer_test_baseline.py --manifest datasets/handwriting_notes/splits/test.jsonl \
    --output outputs/eval_v3/finetuned/test_pred_v3.jsonl \
    --adapter outputs/qlora-v2/v2data-20261115-134816/checkpoint-26 \
    --max-pixels 1024000 --max-new-tokens 1024

$PY scripts/evaluate.py --pred outputs/eval_v3/baseline/test_pred_v3.jsonl \
    --ref datasets/handwriting_notes/splits/test.jsonl --out outputs/eval_v3/baseline
$PY scripts/evaluate.py --pred outputs/eval_v3/finetuned/test_pred_v3.jsonl \
    --ref datasets/handwriting_notes/splits/test.jsonl --out outputs/eval_v3/finetuned

$PY scripts/compare_runs.py --base-pred outputs/eval_v3/baseline/test_pred_v3.jsonl \
    --base-metrics outputs/eval_v3/baseline/metrics.json \
    --ft-pred outputs/eval_v3/finetuned/test_pred_v3.jsonl \
    --ft-metrics outputs/eval_v3/finetuned/metrics.json
$PY scripts/eval_supplement.py

# 3) 单页冒烟（当前入口，两臂，验证页 235955）
$PY scripts/infer_test_baseline.py --manifest outputs/smoke_v3/manifest_235955.jsonl \
    --output outputs/smoke_v3/base_235955.jsonl --max-pixels 1024000 --max-new-tokens 1024
$PY scripts/infer_test_baseline.py --manifest outputs/smoke_v3/manifest_235955.jsonl \
    --output outputs/smoke_v3/ft_235955.jsonl \
    --adapter outputs/qlora-v2/v2data-20261115-134816/checkpoint-26 \
    --max-pixels 1024000 --max-new-tokens 1024
$PY scripts/render_smoke_v3.py     # 读上面两个 jsonl，生成 outputs/smoke_v3/smoke_v3.{json,md}

# 4) 单元测试（默认 Temp 被沙箱拒绝，需指定工作区内 basetemp）
$PY -m pytest tests -q -p no:cacheprovider --basetemp=.pytest_tmp
```

| 类别 | 路径 |
|---|---|
| 最终验收报告（本文件） | `final_comparison_report_v2.md` |
| 历史审计（只读） | `final_comparison_report.md` |
| 标注复核 | `annotation_review_v2.md`、`outputs/annotation_fixes_v2.json` |
| 数据版本 / 哈希 | `datasets/handwriting_notes/data_version_v2.json`(+`.sha256`) |
| 产物清单（前序） | `artifacts_inventory_v2.json` |
| 协议（历史，冻结于重训前） | `inference_protocol_v2.json`、`inference_diagnosis_v2.md` |
| 协议（当前交付） | `inference_protocol_v3_delivery.json` |
| 交接与复现说明 | `HANDOFF.md`、`CHECKPOINT_MANIFEST.md` |
| 冒烟 | `outputs/smoke_v3/`（json / md / 两臂 jsonl + protocol / 日志） |
| 评测预测 | `outputs/eval_v3/{baseline,finetuned}/test_pred_v3.jsonl`(+sidecar) |
| 指标 | `outputs/eval_v3/{baseline,finetuned}/metrics.json`、`err_*.md` |
| 对比 | `outputs/eval_v3/comparison_v3.{json,md}`、`supplement.json`、`eval_report_v3.md` |
| 复现性 | `outputs/eval_v3/repro_check/` |
| 训练 run | `outputs/qlora-v2/v2data-20261115-134816/`（args/logging/4 ckpt） |
| 选中 checkpoint | `outputs/qlora-v2/v2data-20261115-134816/checkpoint-26/` |
| adapter 校验 | `outputs/protocol/adapter_check_v2data_ckpt26.json` |
| 失败尝试证据 | `outputs/qlora-v2/FAILED-134204-lastcheckpoint-bug/`、`outputs/protocol/train_attempt_failed_134204.json` |
| 历史产物（未改动） | `outputs/qlora-v1/`、`outputs/baseline/`、`outputs/finetuned/` |

---

## 9. 未完成事项与尚不支持的结论

**尚未做（明确列出，不当作已完成）：**

1. **未做更大样本评测**。要判断微调是否真有净收益，需要把可评分页从 3 页扩到十几页量级（补拍、或对现有 23 页之外的页面做同样严格的标注）。
2. **未验证表格识别能力**。测试集参考 0 张表，需要专门的含表页面与含表参考。
3. **公式能力未被证实**。当前只有"提取到/没提取到"，缺"公式规范化后逐符号比对"的评估器。
4. **未做上限提升实验**。235955 显示 1024 不足，但没跑 2048/3072 看"加大上限能否救回密集页"——这会改变协议，必须先在验证页定好再双方同跑。
5. **未做提示词/后处理改进**。`## 山东大学` 这类臆造标题提示可能需要显式约束（如"只输出图中存在的标题"），本轮未尝试。
6. **`inference_protocol_v2.json` 的 `adapter` 字段是旧 checkpoint-4**——已按历史协议原样保留（详见 1.5），当前交付配置见 `inference_protocol_v3_delivery.json`。

**尚不支持的结论（不要说）：**

- ❌「微调让 OCR 变好了」——含公式 CER 反而升高，且两臂失败率相同。
- ❌「微调能泛化到手写笔记」——3 页、5 页样本，且两页整页排除。
- ❌「基座不具备公式能力」——基座的公式是"没被提取到"，不是"识别错"。
- ❌「表格指标 3/3 说明会认表」——参考里没有表。
- ❌「验证 loss 更低说明效果更好」——本次恰是反例。
- ❌「训练用了 8.4 GiB 显存」——那是进程侧统计，物理上限 8.0 GiB。

**可以支持的结论：**

- ✅ 推理链路已对齐且可复现（模板逐 token 一致、adapter 权重逐张量相等、贪心解码跨运行一致）。
- ✅ 数据侧 13/5/5 固定划分、互斥、覆盖 23 页，v2 标注逐页对照原图、`□` 用得克制、无 PDF 补写、无自造结构。
- ✅ 4-bit QLoRA 在 Windows 单卡 8 GB 上可行，52 步训练与 4 个 checkpoint 全部落盘且可加载。
- ✅ 在同口径下，**000037 这一页有实质改善**（不再编造宽表、结构与术语更接近原图）。
- ✅ 微调把失败模式从"编造宽表 `<br>`"换成了"编造标题 + 重复公式"，**主指标偏好后者，内容层净收益不成立**。
