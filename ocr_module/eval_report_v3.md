# 基座 vs 微调 冻结评测报告（v3）

- 日期：2026-11-15
- 测试集：`datasets/handwriting_notes/splits/test.jsonl`（5 页真实页，v2 标注，sha256 见 `data_version_v2.json`）
- 双方协议：`inference_protocol_v2.json`（`max_pixels=1024000`、`max_new_tokens=1024`、无 system、同一后处理）
- 结论一句话：**混合变化**。主指标（不含公式文本）大幅改善（CER 1.4872 → 0.7725），但把公式体计入后微调版反而更差（1.1217 → 1.7952）；可评分页只有 3 页且其中 2 页双方都退化，**不足以支撑"微调整体更好"的结论**。

---

## 1. 评测哪个 checkpoint：选 B（重训一次）

### 1.1 判定依据（不是为了让结果好看）

| 判定项 | 事实 | 是否构成"已证实且必须修正" |
|---|---|---|
| 训练标签 | v2 复核修正 10 处 / 7 页，其中**训练集 4 页 7 处**（200854、200858×3、235916×2、235922）、验证集 1 页 1 处（235946）、测试 2 页 2 处 | **是**：旧 checkpoint 是用已证实错误的标签训的 |
| 训练实现 | `gradient_accumulation_steps=16` + 13 个训练样本 → 每 epoch 仅 1 个 optimizer step，4 epoch 共 **4 步**；`warmup_ratio=0.03` 退化为 1 步，LR 第 4 步归零 | **是**：不是超参偏好问题，是"累积步数 > 每 epoch 样本数"导致训练几乎没发生 |
| 训练模板 | `template=qwen3_5` / `model_type=qwen3_5`，与推理侧已核验逐 token 一致（`prompt_identical=true`） | 否，模板本身无需改 |
| 数据划分 | 13/5/5 固定、互斥、覆盖全部 23 页、无合成数据、无测试页泄漏（本阶段重新校验通过） | 否 |

因此：**保留 `outputs/qlora-v1/v2-20261115-114258/checkpoint-4` 作为历史产物**（v1 数据 + 4 步训练），在 v2 数据上重训一次。改动只有一处：`gradient_accumulation_steps: 16 → 1`，其余超参与旧跑完全一致，避免夹带调参。

### 1.2 旧 checkpoint 不是"错的结果"，只是不能代表 v2

旧 checkpoint-4 的 val eval_loss = 1.2456（v1 验证标签、4 步）；新 checkpoint-26 的 val eval_loss = 1.1888（v2 验证标签、26 步）。**两者不可直接比较**（标签版本与步数都不同），这里只作为"新跑确实在学"的旁证。

---

## 2. 重训记录

### 2.1 实际参数（全部来自运行产物 `args.json`，非手写清单）

| 项 | 值 |
|---|---|
| run dir | `outputs/qlora-v2/v2data-20261115-134816` |
| base model | `SeerRay-Lab/Xiaomi-OCR-0`（ModelScope 快照；与推理用的 HF 快照**逐文件 sha256 相同**，含 1.7 GB 权重） |
| model_type / template | `qwen3_5` / `qwen3_5` |
| tuner / target | `lora` / `all-linear`（保存后的正则只覆盖 `model.language_model.*`，372 个模块，0 个落在语言模型之外） |
| 量化 | `quant_method=bnb`、`quant_bits=4`、`bnb_4bit_quant_type=nf4`、`bnb_4bit_use_double_quant=true`、`bnb_4bit_compute_dtype=bfloat16`、`torch_dtype=bfloat16` |
| LoRA | r=8、alpha=32、dropout=0.05，可训练 5,411,328（0.8877%） |
| 冻结 | `freeze_vit=true`、`freeze_aligner=true` |
| 数据 | `dataset=train_swift.jsonl`(13)、**`val_dataset=val_swift.jsonl`(5) 显式接入**、`split_dataset_ratio=0`（禁止自动切分） |
| 长度 | `max_length=8192`、`truncation_strategy=delete`、`max_pixels=1024000` |
| 优化 | lr=1e-4、cosine、`warmup_ratio=0.03`、batch=1、**grad_accum=1**、epochs=4 → **52 步**、seed=42 |
| 保存/评估 | `eval_strategy=epoch`、`save_strategy=epoch`、`save_total_limit=4` → checkpoint-13/26/39/52 全部保留 |
| 产出 | 238 s、train_loss 0.7831、4 个真实 checkpoint 目录（不依赖 `last-checkpoint` 符号链接） |

### 2.2 验证曲线（只用它选 checkpoint）

| ckpt | step | epoch | eval_loss | eval_token_acc |
|---|---|---|---|---|
| checkpoint-13 | 13 | 1 | 1.20785 | 0.7691 |
| **checkpoint-26** | 26 | 2 | **1.18883** | **0.7781** |
| checkpoint-39 | 39 | 3 | 1.18942 | 0.7747 |
| checkpoint-52 | 52 | 4 | 1.19147 | 0.7773 |

**选 checkpoint-26（验证集 eval_loss 最低）**，测试集全程未参与任何选择。

### 2.3 内存与显存

| 阶段 | `memory(GiB)` |
|---|---|
| 1 步 smoke（训练） | 2.53 |
| 1 步 smoke（含 eval） | 3.51 |
| 2 步 smoke | 4.91 |
| 正式训练（52 步，含 4 次 eval） | 8.40 |

注意：这是 ms-swift 的进程侧统计，**不是物理显存**。本卡物理 8188 MiB = 8.0 GiB，8.40 GiB 已超过物理上限，说明 WDDM 借用了共享内存；同样地，推理侧 d3 用例曾出现 `peak_cuda_allocated=13010 MiB`。数字只能同机横向比较。

### 2.4 一次失败尝试（如实记录，已修复）

第一次重训在 **epoch-1 保存 checkpoint 时崩溃**：

```
PermissionError: [WinError 5] 拒绝访问。 '...\last-checkpoint.tmp'
```

根因链：`swift.utils.io_utils.update_last_checkpoint_symlink` 尝试 `os.symlink` → Windows 无创建符号链接权限（`WinError 1314`）→ 回滚分支对一个**空目录** `last-checkpoint.tmp` 调 `os.remove()` → Windows 不允许 `os.remove` 删目录。

修复：在 run dir 预置一个同名空目录，让该函数走 "already exists and is not a symlink; skip" 的早退分支，从而完全不依赖符号链接（本项目一律用真实 `checkpoint-N` 目录名）。
失败产物保留在 `outputs/qlora-v2/FAILED-134204-lastcheckpoint-bug/`，证据 `outputs/protocol/train_attempt_failed_134204.json`（含当时的 loss/LR 日志）。它**没有**被当作训练结果使用。

---

## 3. 评测冻结口径

| 项 | 设定 |
|---|---|
| 输入尺度 | 双方 `max_pixels=1024000`，grid 全部落在 `[1,54,72]` / `[1,72,54]` → 3888 patch / 972 视觉 token |
| 提示词 | 无 system、单 user 轮、`"\n"+INSTRUCTION`，与训练逐 token 一致 |
| 生成 | 贪心、`max_new_tokens=1024`、`pad_token_id` 显式设置、`torch.no_grad()` |
| 后处理 | 同一 `normalize_output`（剥一层外层包装 + 统一 `\[..\]`→`$$..$$`），两侧相同 |
| 评估器 | 同一份 `scripts/evaluate.py`（本次只新增字段，不改变既有指标定义） |
| 排除规则 | 参考含 `□` 的页整页排除，双方同规则 |
| 复现性 | 单页重跑验证：`000037` 微调版两次运行 markdown **sha256 完全相同**（307 token、eos），见 `outputs/eval_v3/repro_check/` |
| 历史产物 | 旧 4096 / 512 预测保留在 `outputs/baseline/`、`outputs/finetuned/`，未改动、未参与本次口径 |

新实验目录：`outputs/eval_v3/{baseline,finetuned}/`（每页一条 jsonl，含完整 `run` 元数据 + 原始生成 + 归一化后 markdown）。

---

## 4. 指标

### 4.1 页数与排除

| 项 | 值 |
|---|---|
| 总页数 | 5 |
| 可评分页 | 3（000037、000044、000051） |
| 排除 | 2 |
| 排除原因 | `unknown_reference`：参考含 `□` —— 000020（28 个 □，整页失焦）、000028（1 个 □，红字单位前缀不可辨） |

两页的预测**已生成且已落盘**（000020: 基座 139 / 微调 152 token；000028: 基座 417 / 微调 648 token），只是按规则不计分。

### 4.2 主指标（冻结评估器，公式文本**不**计入）

| 指标 | 基座 | 微调 | 差值 |
|---|---|---|---|
| 宏平均 CER（每页等权） | **1.4872** | **0.7725** | −0.7147 |
| 字符加权 CER（Σ编辑距离 / Σ参考字符） | 1.4386 | 0.7433 | −0.6953 |
| Σ编辑距离 | 1289 | 666 | −623 |
| Σ参考字符（分母） | 896 | 896 | 0 |

CER 可以超过 100%：分母只有 896 个参考字符，而基座多插入了大量内容（000037 输出 781 字符对 355 参考、000051 输出 433 对 212），插入与重复使编辑距离超过参考长度，故 CER > 1。

### 4.3 补充指标（把公式体计入文本）——**这一栏改变了结论**

主指标的 `plain_text()` 会把 `$...$` / `$$...$$` 整段删除。微调版恰恰把大量内容包进公式，于是被"从分母里抹掉"，看起来更好。用同一套归一化与同一份编辑距离代码、但保留公式体后：

| 指标 | 基座 | 微调 |
|---|---|---|
| 宏平均 CER（含公式体） | **1.1217** | **1.7952** |
| 字符加权 CER（含公式体） | 1.1125 | 1.9119 |

逐页（含公式体）：

| 页 | 基座 | 微调 | 说明 |
|---|---|---|---|
| 000037 | 1.3453 | **0.6102** | 微调确实更好 |
| 000044 | 0.8228 | **3.9599** | 微调严重退化（重复公式） |
| 000051 | 1.1969 | 0.8154 | 微调略好 |

即：**主指标的改善有相当一部分来自"内容被公式定界符吞掉"，把公式体算进来后微调整体更差。**（`outputs/eval_v3/supplement.json`）

### 4.4 生成失败与截断

| 项 | 基座 | 微调 |
|---|---|---|
| 命中 1024 上限 | 0/3 | 0/3 |
| 正常 eos 结束 | 3/3 | 3/3 |
| 检出重复 | **2/3**（000037、000051，均 `ngram`） | **2/3**（000044：unit+template+ngram；000051：line+unit+template） |
| 失败率（触上限或重复） | 66.7% | 66.7% |
| 总耗时 | 132.9 s | 231.8 s（1.74×） |
| 单页峰值 allocated | 989.2 MiB | 1009.4 MiB |
| 整卡占用（nvidia-smi，含桌面等其它进程） | 1877–1883 / 8188 MiB | 1885–1897 / 8188 MiB |

整卡占用只有约 1.9 GiB，而进程峰值 allocated 约 1 GiB —— 两者语义不同：前者是整卡（含其它进程）、后者是本进程经 torch 分配器；在 WDDM 下二者既不能相互替代，也不能当作物理解释。

没有任何一页触到上限，所以**"截断"不是本次的失败原因**；两边的失败都是"自己决定停下但内容在退化"。

### 4.5 结构指标（必须连同"参考有没有该结构"一起看）

| 页 | 参考 headings / lists / tables | 基座 | 微调 |
|---|---|---|---|
| 000037 | 2 / 10 / **0** | 0 / 0 / 1 | 5 / 12 / 0 |
| 000044 | 5 / 10 / **0** | 0 / 0 / 1 | 23 / 0 / 0 |
| 000051 | 4 / 11 / **0** | 0 / 0 / 1 | 1 / 0 / 0 |

- 标题序列匹配率：双方均 **0.0**（基座完全不输出 `#` 标题；微调的层级与参考不一致，000044 更是把 23 行都写成 `##`）。
- **表格指标无区分度**：3 页参考里**一张表都没有**，基座编造了 3 张表（全部错误），微调一张都没输出（因此"表格数匹配"是 3/3）。这只说明"不瞎编表格"，**不能说明具备表格识别能力**。
- 列表项：参考 10/10/11，基座 0/0/0（完全不输出列表），微调 12/0/0。

### 4.6 公式指标

- 提取器只认 `$...$` / `$$...$$`（`ocr/postprocess._FORMULA_RE`）；`\[..\]` / `\(..\)` 经 `normalize_output` 转换后才可被提取。
- 基座：3 页参考共 47 个公式，基座**一个都没提取到**（`n_hyp=0`，全部页面）→ 记为"**未提取到格式化公式**"。这不证明公式识别错误，但也不证明正确；实际输出里公式内容是被塞进编造表格的纯文本（如 `| C = C0/C0' |`）。
- 微调：`n_hyp` 分别为 2 / 1 / 25，即"有公式了"，但**精确匹配率 0.0**，公式 CER 3.0894（>1，因为输出了大量参考中没有的公式）。基座这一栏的 1.0 也不是"好"，而是"输出为空、全部按删除计"。

---

## 5. 对照原图的错误分析（不只凭指标）

三张可评分页均重新打开原图核对（裁切图见 `outputs/eval_v3/crops/`）。

### 5.1 000037（页顶"置位/复位" + "21. 8253"）——**真实改善，但仍错**

原图实读：`置位/复位`：`D7=0`；`D6D5 任意值`；`D3D2D1: 000~111 ⇒ PC0~PC7`；`D0: 1 SET 置位 / 0 RESET 复位`。工作方式 6 条，其中 `方式2(分频器)…(f出=f输入÷N初值)`。

- 基座：把整页压成一张 `<md> | … |` 表，用了上百个 `<br>` 占位（这就是 781 字符与 2.20 倍长度的来源），`分频器`→`分电器`、`单稳触发器`→`单档继电器`，结构完全错。
- 微调：恢复成"标题 + 列表"，`分频器`、`方波发生器` 读对（比基座好）；但**整块"置位/复位"丢失**，控制字 `D7D6`/`D5D4` 两行细节丢失，`方式1` 写成"单格计数器"、`GATE`→`CATE`，`方式2` 的公式被替换成凭空生成的 `$1=\frac12+\frac12+\frac12$`。
- 结论：该页 CER 1.72→0.47（含公式 1.35→0.61）是**真实改善**，主要来自"不再编造大表 + 结构对了"；但内容仍有多处错误与整块遗漏，不能称为识别正确。

### 5.2 000044（22. 8259，ICW1~ICW4）——**双方都失败，ΔCER 不能解读为退化**

原图为 ICW1 的 `D7D6D5 8080/8085 有用、8086 任意`、`D4 固定 1`、`D3 1 电平/0 边沿` 等，参考标注与图一致，页面无表格。

- 基座：输出与页面无关的编造表（`定义|IOW1|IOW2…`、`统计模型`、`P0,P0,P0`），基本没有可用内容。
- 微调：输出 `$$I_0 \rightarrow I_0 \rightarrow …$$` 重复约 100 次（含公式体 1986 字符，去公式体只剩 77 字符）。主指标 CER 0.9149 之所以"看起来还行"，是因为这些重复全在公式里、被 `plain_text()` 删掉了。
- 结论：该页微调的 CER 从 0.8359 升到 0.9149，**但两侧都没转写**，这个差值不构成"退化"的证据；真正的信号是"主指标把一大段重复内容隐藏了"。

### 5.3 000051（操作命令字 OCW1~OCW3）——**双方都失败，微调略好**

原图：`OCW1: A0=1`、`Di=1⇒IRi 中断被屏蔽`；`OCW2: A0=0, D4D3=00`，`D7D6D5` 八种组合；`OCW3: A0=0, D7=X, D4D3=01`，`D6D5/D2/D1D0` 各项。

- 基座：`P1 P2 … P99` 计数式退化，无有效内容。
- 微调：开头几行抓到了 `OCU2 = 1 / A0 = 1`、`Di=1; D0 中断,被解构`（把"被屏蔽/被开放"读成"被解构"），随后退化成 `$D_0=0$;` 连续重复。
- 结论：微调在开头比基座更接近原图，但同样崩掉；CER 1.91→0.93（含公式 1.20→0.82）反映的是"崩得没那么彻底"，不是"识别对了"。

### 5.4 不可评分页的观察（不计分）

- 000020（整页失焦、28 个 □）：基座 139 / 微调 152 token，两侧都只输出极短内容，微调触发 `template_line_run`。
- 000028（红字 + 8255A）：基座 417 / 微调 648 token，微调触发 `template_line_run`。两页参考本身含 `□`，无法客观评分，故不参与对比。

---

## 6. 结论

**混合变化（mixed），且不足以宣称"微调更好"。**

1. 主指标（不含公式文本）显著改善：宏 CER 1.4872 → 0.7725，字符加权 1.4386 → 0.7433。
2. 但把公式体计入同一套 CER 后，微调反而更差：1.1217 → 1.7952。差异主要来自 000044 一页的公式重复（被主指标删掉 1900 字符）。
3. 可评分页只有 3 页，其中 **2 页双方都退化**（失败率同为 66.7%），公式精确匹配率双方都是 0.0，标题序列匹配率双方都是 0.0。
4. 唯一可确认的实质改善在 000037：结构恢复为"标题+列表"、`分频器/方波发生器` 等术语读对、不再编造 `<br>` 大表。同页仍有整块遗漏与公式臆造。
5. 因此：**不能**说"微调使 OCR 变好"，只能说"在这 3 页上，微调把一个失败模式（编造宽表）换成了另一个失败模式（公式/标题重复），主指标偏好后者，而内容层面的净收益不成立"。

### 需要写清的口径限制

- 主指标对"把内容塞进 `$…$`"的模型有系统性优待（公式体从分母中消失）。任何后续结论都应同时看第 4.3 节的含公式版本。
- 表格能力**未被测试**（参考 3 页无表格）。
- 公式能力**未被证实**（EM 0.0；基座是"没提取到"，微调是"提取到但不匹配"）。
- 样本量 3 页，单页变化即主导均值，不做统计显著性推断。

---

## 7. 产物清单

| 类别 | 路径 |
|---|---|
| 基座 5 页预测 + 元数据 | `outputs/eval_v3/baseline/test_pred_v3.jsonl`（+ `.protocol.json` 指纹） |
| 微调 5 页预测 + 元数据 | `outputs/eval_v3/finetuned/test_pred_v3.jsonl`（+ `.protocol.json`） |
| 同版本评测结果 | `outputs/eval_v3/{baseline,finetuned}/metrics.json`、`err_*.md` |
| 对比表（机器可读 / 人读） | `outputs/eval_v3/comparison_v3.json`、`comparison_v3.md` |
| 含公式体补充指标 | `outputs/eval_v3/supplement.json` |
| 复现性验证 | `outputs/eval_v3/repro_check/` |
| 原图裁切（错误分析用） | `outputs/eval_v3/crops/` |
| 新训练 run（参数/日志/曲线/4 个 checkpoint） | `outputs/qlora-v2/v2data-20261115-134816/` |
| 选中 checkpoint | `outputs/qlora-v2/v2data-20261115-134816/checkpoint-26/` |
| adapter 权重校验 | `outputs/protocol/adapter_check_v2data_ckpt26.json`（372/372、max_abs_diff=0、`weights_verified=true`） |
| 训练 smoke（显存探测） | `outputs/protocol/train_smoke_v2data.json` |
| 失败尝试证据 | `outputs/qlora-v2/FAILED-134204-lastcheckpoint-bug/`、`outputs/protocol/train_attempt_failed_134204.json` |
| 历史产物（未改动） | `outputs/qlora-v1/`、`outputs/baseline/`、`outputs/finetuned/` |

### 复现命令

```bash
# 训练（唯一改动 grad_accum=1；其余与 v1 相同）
powershell -File scripts/train_qlora_v2.ps1 -OutputRoot outputs/qlora-v2

# 评测（双方同一协议、新目录）
python scripts/infer_test_baseline.py --manifest datasets/handwriting_notes/splits/test.jsonl \
    --output outputs/eval_v3/baseline/test_pred_v3.jsonl --max-pixels 1024000 --max-new-tokens 1024
python scripts/infer_test_baseline.py --manifest datasets/handwriting_notes/splits/test.jsonl \
    --output outputs/eval_v3/finetuned/test_pred_v3.jsonl \
    --adapter outputs/qlora-v2/v2data-20261115-134816/checkpoint-26 \
    --max-pixels 1024000 --max-new-tokens 1024

python scripts/evaluate.py --pred outputs/eval_v3/baseline/test_pred_v3.jsonl \
    --ref datasets/handwriting_notes/splits/test.jsonl --out outputs/eval_v3/baseline
python scripts/evaluate.py --pred outputs/eval_v3/finetuned/test_pred_v3.jsonl \
    --ref datasets/handwriting_notes/splits/test.jsonl --out outputs/eval_v3/finetuned

python scripts/compare_runs.py --base-pred outputs/eval_v3/baseline/test_pred_v3.jsonl \
    --base-metrics outputs/eval_v3/baseline/metrics.json \
    --ft-pred outputs/eval_v3/finetuned/test_pred_v3.jsonl \
    --ft-metrics outputs/eval_v3/finetuned/metrics.json
python scripts/eval_supplement.py
```
